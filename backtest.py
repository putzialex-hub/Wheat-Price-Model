from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import ModelSpec
import os
from target_utils import from_target, to_target
from collections import deque

from calibration import (
    apply_conformal,
    apply_residual_interval,
    conformal_qhat,
    conformal_qhat_residual,
    rolling_pool_append,
)
from model import predict, train_model, train_quantile_models, predict_quantiles


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame  # meta + y_true + y_pred
    metrics: Dict[str, float]


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-9, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _nonconformity_scores(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> np.ndarray:
    scores = np.maximum(p10 - y_true, y_true - p90)
    return np.maximum(scores, 0.0)


def _coverage_width(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    coverage = float(np.mean((y_true >= low) & (y_true <= high)))
    width = float(np.mean(high - low))
    return coverage, width


def select_features_for_model(X: pd.DataFrame, model_type: str) -> pd.DataFrame:
    if model_type != "ridge":
        return X
    drop_cols = [c for c in ["close_c2", "settlement_c2", "close_c2__is_missing"] if c in X.columns]
    return X.drop(columns=drop_cols)


def _choose_ridge_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    model_spec: ModelSpec,
) -> float:
    grid = model_spec.ridge_alpha_grid
    if not grid:
        return model_spec.ridge_alpha
    df = meta.copy()
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["qend_date", "asof_date"]).reset_index(drop=True)
    unique_qends = sorted(df["qend_date"].unique())
    split_idx = max(1, int(len(unique_qends) * 0.7))
    train_qends = unique_qends[:split_idx]
    val_qends = unique_qends[split_idx:]
    if not val_qends:
        return model_spec.ridge_alpha

    train_idx = df[df["qend_date"].isin(train_qends)]["row_id"].values
    val_idx = df[df["qend_date"].isin(val_qends)]["row_id"].values
    X_train = select_features_for_model(X.iloc[train_idx], "ridge")
    X_val = select_features_for_model(X.iloc[val_idx], "ridge")
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    y_train_delta = y_train.to_numpy() - X_train["asof_close"].astype(float).to_numpy()
    y_train_delta = pd.Series(y_train_delta, index=y_train.index)
    asof_close_val = X_val["asof_close"].astype(float).to_numpy()
    best_alpha = model_spec.ridge_alpha
    best_mae = float("inf")
    for alpha in grid:
        ridge_model = train_model(X_train, y_train_delta, model_spec, model_type="ridge", ridge_alpha=alpha)
        delta_pred = predict(ridge_model, X_val)
        y_pred = asof_close_val + delta_pred
        mae = _mae(y_val.to_numpy(), y_pred)
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return best_alpha


def walk_forward_by_quarter(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    model_spec: ModelSpec,
    min_train_quarters: int = 12,
    delta_clip: float | None = 80.0,
    hybrid_threshold: float = 10.0,
    enable_hybrid: bool = True,
    hybrid_model: str = "ridge",
    target_mode: str = "level",
) -> BacktestResult:
    """
    Walk-forward:
      train on all rows with qend_date < current qend_date
      test on current quarter (all asof rows mapping to that quarter-end)
    """
    if hybrid_model not in {"ridge", "tree", "both"}:
        raise ValueError("hybrid_model must be one of: ridge, tree, both")
    if model_spec.calibration_mode not in {"per_fold", "pooled", "rolling"}:
        raise ValueError("calibration_mode must be one of: per_fold, pooled, rolling")

    df = meta.copy()
    df["y_true"] = df["y_true_price"].values
    df["y_true_target"] = y.values
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["qend_date", "asof_date"]).reset_index(drop=True)

    unique_qends = sorted(df["qend_date"].unique())
    preds_rows: List[pd.DataFrame] = []
    ridge_alpha = _choose_ridge_alpha(X, y, meta, model_spec)
    pooled_scores: List[float] = []
    rolling_scores = deque(maxlen=model_spec.rolling_folds)
    rolling_pool: List[float] = []
    y_price_all = df["y_true_price"].to_numpy()

    for i, qend in enumerate(unique_qends):
        train_qends = unique_qends[:i]
        if len(train_qends) < min_train_quarters:
            continue

        train_idx = df[df["qend_date"].isin(train_qends)]["row_id"].values
        test_idx = df[df["qend_date"] == qend]["row_id"].values

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        X_train_ridge = select_features_for_model(X_train, "ridge")
        X_test_ridge = select_features_for_model(X_test, "ridge")
        y_price_train = y_price_all[train_idx]
        y_price_test = y_price_all[test_idx]

        if "asof_close" not in X_test.columns or "ret_20d" not in X_test.columns:
            raise ValueError("Features must include asof_close and ret_20d for baselines.")

        asof_close = X_test["asof_close"].astype(float).to_numpy()
        ret_20d = X_test["ret_20d"].fillna(0.0).astype(float).to_numpy()

        y_train_target = y_train
        tree_model = train_model(X_train, y_train_target, model_spec, model_type="tree")
        pred_target_tree = predict(tree_model, X_test)

        y_train_target_ridge = y_train_target
        ridge_model = train_model(
            X_train_ridge,
            y_train_target_ridge,
            model_spec,
            model_type="ridge",
            ridge_alpha=ridge_alpha,
        )
        pred_target_ridge = predict(ridge_model, X_test_ridge)

        calib_size = max(20, int(len(X_train) * 0.2))
        if len(X_train) > calib_size:
            fit_X = X_train.iloc[:-calib_size]
            fit_y = y_train_target.iloc[:-calib_size]
            cal_X = X_train.iloc[-calib_size:]
            cal_y = y_train_target.iloc[-calib_size:]
            quantile_models = train_quantile_models(fit_X, fit_y)
            cal_pred = predict_quantiles(quantile_models, cal_X)
            cal_scores = _nonconformity_scores(
                cal_y.to_numpy(),
                cal_pred["delta_p10"].to_numpy(),
                cal_pred["delta_p90"].to_numpy(),
            )
            cal_asof = cal_X["asof_close"].astype(float).to_numpy()
            cal_y_target = cal_y.to_numpy()
            cal_naive_target = to_target(cal_asof, cal_asof, target_mode)
            cal_p50_target = cal_pred["delta_p50"].to_numpy()
            q_hat_naive = conformal_qhat_residual(cal_y_target, cal_naive_target, model_spec.interval_alpha)
            q_hat_p50 = conformal_qhat_residual(cal_y_target, cal_p50_target, model_spec.interval_alpha)
            pooled_scores.extend(cal_scores.tolist())
            rolling_scores.append(cal_scores)
            rolling_pool = rolling_pool_append(rolling_pool, cal_scores, model_spec.rolling_calibration_size)
            alpha = model_spec.interval_alpha
            if model_spec.calibration_mode == "pooled" and len(pooled_scores) >= model_spec.min_pool_size:
                q_hat = conformal_qhat(np.asarray(pooled_scores), alpha)
            elif model_spec.calibration_mode == "rolling" and rolling_scores:
                window_scores = np.concatenate(list(rolling_scores))
                if len(window_scores) >= model_spec.min_pool_size:
                    q_hat = conformal_qhat(window_scores, alpha)
                else:
                    q_hat = conformal_qhat(cal_scores, alpha)
            else:
                q_hat = conformal_qhat(cal_scores, alpha)
            if len(rolling_pool) >= model_spec.min_pool_size:
                q_hat_roll = conformal_qhat(np.asarray(rolling_pool), alpha)
            else:
                q_hat_roll = q_hat
        else:
            quantile_models = train_quantile_models(X_train, y_train_target)
            q_hat = 0.0
            q_hat_naive = 0.0
            q_hat_p50 = 0.0
            q_hat_roll = 0.0
        delta_quantiles = predict_quantiles(quantile_models, X_test)

        if delta_clip is not None and target_mode in {"delta", "log_return"}:
            pred_target_tree = np.clip(pred_target_tree, -delta_clip, delta_clip)
            pred_target_ridge = np.clip(pred_target_ridge, -delta_clip, delta_clip)
            for col in ["delta_p10", "delta_p50", "delta_p90"]:
                delta_quantiles[col] = np.clip(delta_quantiles[col], -delta_clip, delta_clip)

        asof_close_arr = asof_close
        y_pred_tree = from_target(pred_target_tree, asof_close_arr, target_mode)
        y_pred_ridge = from_target(pred_target_ridge, asof_close_arr, target_mode)
        y_pred_p10_target = delta_quantiles["delta_p10"].to_numpy()
        y_pred_p50_target = delta_quantiles["delta_p50"].to_numpy()
        y_pred_p90_target = delta_quantiles["delta_p90"].to_numpy()
        y_pred_p10 = from_target(y_pred_p10_target, asof_close_arr, target_mode)
        y_pred_p50 = from_target(y_pred_p50_target, asof_close_arr, target_mode)
        y_pred_p90 = from_target(y_pred_p90_target, asof_close_arr, target_mode)
        p10_cal_t, p90_cal_t = apply_conformal(y_pred_p10_target, y_pred_p90_target, q_hat)
        p10_cal_roll_t, p90_cal_roll_t = apply_conformal(y_pred_p10_target, y_pred_p90_target, q_hat_roll)
        y_pred_p10_cal = from_target(p10_cal_t, asof_close_arr, target_mode)
        y_pred_p90_cal = from_target(p90_cal_t, asof_close_arr, target_mode)
        p10_cal_roll = from_target(p10_cal_roll_t, asof_close_arr, target_mode)
        p90_cal_roll = from_target(p90_cal_roll_t, asof_close_arr, target_mode)
        naive_target = to_target(asof_close_arr, asof_close_arr, target_mode)
        low_naive_t, high_naive_t = apply_residual_interval(naive_target, q_hat_naive)
        low_p50_t, high_p50_t = apply_residual_interval(y_pred_p50_target, q_hat_p50)
        low_naive = from_target(low_naive_t, asof_close_arr, target_mode)
        high_naive = from_target(high_naive_t, asof_close_arr, target_mode)
        low_p50 = from_target(low_p50_t, asof_close_arr, target_mode)
        high_p50 = from_target(high_p50_t, asof_close_arr, target_mode)
        y_pred_naive = asof_close
        y_pred_mom = asof_close * (1.0 + ret_20d)

        use_model_tree = None
        use_model_ridge = None
        y_pred_hybrid_tree = None
        y_pred_hybrid_ridge = None
        delta_pred_tree_price = y_pred_tree - asof_close
        delta_pred_ridge_price = y_pred_ridge - asof_close
        if enable_hybrid:
            if hybrid_model in {"tree", "both"}:
                use_model_tree = np.abs(delta_pred_tree_price) >= hybrid_threshold
                y_pred_hybrid_tree = np.where(use_model_tree, y_pred_tree, y_pred_naive)
            if hybrid_model in {"ridge", "both"}:
                use_model_ridge = np.abs(delta_pred_ridge_price) >= hybrid_threshold
                y_pred_hybrid_ridge = np.where(use_model_ridge, y_pred_ridge, y_pred_naive)
        else:
            if hybrid_model in {"tree", "both"}:
                use_model_tree = np.ones_like(y_pred_tree, dtype=bool)
                y_pred_hybrid_tree = y_pred_tree
            if hybrid_model in {"ridge", "both"}:
                use_model_ridge = np.ones_like(y_pred_ridge, dtype=bool)
                y_pred_hybrid_ridge = y_pred_ridge

        out = df[df["row_id"].isin(test_idx)].copy()
        out["y_pred_tree"] = y_pred_tree
        out["y_pred_ridge"] = y_pred_ridge
        out["y_pred_p10"] = y_pred_p10
        out["y_pred_p50"] = y_pred_p50
        out["y_pred_p90"] = y_pred_p90
        out["y_pred_p10_raw"] = y_pred_p10
        out["y_pred_p90_raw"] = y_pred_p90
        out["y_pred_p10_cal"] = y_pred_p10_cal
        out["y_pred_p90_cal"] = y_pred_p90_cal
        out["q_hat"] = q_hat
        out["q_hat_roll"] = q_hat_roll
        out["p10_cal_roll"] = p10_cal_roll
        out["p90_cal_roll"] = p90_cal_roll
        out["y_pred_target_p50"] = y_pred_p50_target
        out["low_naive"] = low_naive
        out["high_naive"] = high_naive
        out["q_hat_naive"] = q_hat_naive
        out["low_p50"] = low_p50
        out["high_p50"] = high_p50
        out["q_hat_p50"] = q_hat_p50
        if y_pred_hybrid_tree is not None:
            out["y_pred_hybrid_tree"] = y_pred_hybrid_tree
            out["use_model_tree"] = use_model_tree.astype(int)
        if y_pred_hybrid_ridge is not None:
            out["y_pred_hybrid_ridge"] = y_pred_hybrid_ridge
            out["use_model_ridge"] = use_model_ridge.astype(int)
        out["y_pred_naive"] = y_pred_naive
        out["y_pred_mom"] = y_pred_mom
        out["asof_close"] = asof_close
        out["ret_20d"] = ret_20d
        out["y_true_price"] = y_price_test
        out["y_true_target"] = y_test.to_numpy()
        if "weeks_to_qend" in X_test.columns:
            out["weeks_to_qend"] = X_test["weeks_to_qend"].to_numpy()
        preds_rows.append(out)

    if not preds_rows:
        raise ValueError("Backtest produced no predictions; check min_train_quarters or data length")

    preds = pd.concat(preds_rows, ignore_index=True)
    y_true = preds["y_true_price"].to_numpy()
    y_true_target = preds["y_true_target"].to_numpy()
    y_pred_tree = preds["y_pred_tree"].to_numpy()
    y_pred_ridge = preds["y_pred_ridge"].to_numpy()
    y_pred_p10 = preds["y_pred_p10"].to_numpy()
    y_pred_p50 = preds["y_pred_p50"].to_numpy()
    y_pred_target_p50 = preds["y_pred_target_p50"].to_numpy()
    y_pred_p90 = preds["y_pred_p90"].to_numpy()
    y_pred_p10_cal = preds["y_pred_p10_cal"].to_numpy()
    y_pred_p90_cal = preds["y_pred_p90_cal"].to_numpy()
    p10_cal_roll = preds["p10_cal_roll"].to_numpy()
    p90_cal_roll = preds["p90_cal_roll"].to_numpy()
    low_naive = preds["low_naive"].to_numpy()
    high_naive = preds["high_naive"].to_numpy()
    low_p50 = preds["low_p50"].to_numpy()
    high_p50 = preds["high_p50"].to_numpy()
    y_pred_naive = preds["y_pred_naive"].to_numpy()
    y_pred_mom = preds["y_pred_mom"].to_numpy()

    coverage_roll, width_roll = _coverage_width(y_true, p10_cal_roll, p90_cal_roll)
    coverage_cal, width_cal = _coverage_width(y_true, y_pred_p10_cal, y_pred_p90_cal)
    if preds["qend_date"].nunique() >= 40:
        recent_qends = sorted(preds["qend_date"].unique())[-40:]
        recent_mask = preds["qend_date"].isin(recent_qends).to_numpy()
        coverage_roll_recent, width_roll_recent = _coverage_width(
            y_true[recent_mask],
            p10_cal_roll[recent_mask],
            p90_cal_roll[recent_mask],
        )
        coverage_cal_recent, width_cal_recent = _coverage_width(
            y_true[recent_mask],
            y_pred_p10_cal[recent_mask],
            y_pred_p90_cal[recent_mask],
        )
    else:
        coverage_roll_recent, width_roll_recent = float("nan"), float("nan")
        coverage_cal_recent, width_cal_recent = float("nan"), float("nan")

    metrics = {
        "MAE_tree": _mae(y_true, y_pred_tree),
        "RMSE_tree": _rmse(y_true, y_pred_tree),
        "MAPE_tree": _mape(y_true, y_pred_tree),
        "MAE_ridge": _mae(y_true, y_pred_ridge),
        "RMSE_ridge": _rmse(y_true, y_pred_ridge),
        "MAPE_ridge": _mape(y_true, y_pred_ridge),
        "MAE_p50": _mae(y_true, y_pred_p50),
        "MAPE_p50": _mape(y_true, y_pred_p50),
        "MAE_target_p50": _mae(y_true_target, y_pred_target_p50),
        "coverage_80_raw": float(np.mean((y_true >= y_pred_p10) & (y_true <= y_pred_p90))),
        "avg_width_80_raw": float(np.mean(y_pred_p90 - y_pred_p10)),
        "coverage_80_cal": float(np.mean((y_true >= y_pred_p10_cal) & (y_true <= y_pred_p90_cal))),
        "avg_width_80_cal": float(np.mean(y_pred_p90_cal - y_pred_p10_cal)),
        "coverage_cal_roll": coverage_roll,
        "width_cal_roll": width_roll,
        "coverage_cal_roll_recent40": coverage_roll_recent,
        "width_cal_roll_recent40": width_roll_recent,
        "coverage_cal_recent40": coverage_cal_recent,
        "width_cal_recent40": width_cal_recent,
        "coverage_resid_naive": float(np.mean((y_true >= low_naive) & (y_true <= high_naive))),
        "width_resid_naive": float(np.mean(high_naive - low_naive)),
        "coverage_resid_p50": float(np.mean((y_true >= low_p50) & (y_true <= high_p50))),
        "width_resid_p50": float(np.mean(high_p50 - low_p50)),
        "pinball_p10": _pinball(y_true, y_pred_p10, 0.1),
        "pinball_p50": _pinball(y_true, y_pred_p50, 0.5),
        "pinball_p90": _pinball(y_true, y_pred_p90, 0.9),
        "MAE_naive": _mae(y_true, y_pred_naive),
        "RMSE_naive": _rmse(y_true, y_pred_naive),
        "MAPE_naive": _mape(y_true, y_pred_naive),
        "MAE_mom": _mae(y_true, y_pred_mom),
        "RMSE_mom": _rmse(y_true, y_pred_mom),
        "MAPE_mom": _mape(y_true, y_pred_mom),
    }
    if "y_pred_hybrid_tree" in preds.columns:
        y_pred_hybrid_tree = preds["y_pred_hybrid_tree"].to_numpy()
        metrics.update(
            {
                "MAE_hybrid_tree": _mae(y_true, y_pred_hybrid_tree),
                "RMSE_hybrid_tree": _rmse(y_true, y_pred_hybrid_tree),
                "MAPE_hybrid_tree": _mape(y_true, y_pred_hybrid_tree),
            }
        )
    if "y_pred_hybrid_ridge" in preds.columns:
        y_pred_hybrid_ridge = preds["y_pred_hybrid_ridge"].to_numpy()
        metrics.update(
            {
                "MAE_hybrid_ridge": _mae(y_true, y_pred_hybrid_ridge),
                "RMSE_hybrid_ridge": _rmse(y_true, y_pred_hybrid_ridge),
                "MAPE_hybrid_ridge": _mape(y_true, y_pred_hybrid_ridge),
            }
        )

    if os.environ.get("WHEAT_DEBUG_TARGET") == "1":
        print("Debug y_pred_target_p50:\n", pd.Series(y_pred_target_p50).describe())
        print("Debug y_pred_p50 (price):\n", pd.Series(y_pred_p50).describe())
        if target_mode == "level" and pd.Series(y_pred_p50).nunique() <= 1:
            raise ValueError("level target_mode produced constant price predictions; check target mapping.")

    return BacktestResult(predictions=preds, metrics=metrics)
