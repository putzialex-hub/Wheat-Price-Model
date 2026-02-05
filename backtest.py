from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import ModelSpec
from model import train_model, predict


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
) -> BacktestResult:
    """
    Walk-forward:
      train on all rows with qend_date < current qend_date
      test on current quarter (all asof rows mapping to that quarter-end)
    """
    if hybrid_model not in {"ridge", "tree", "both"}:
        raise ValueError("hybrid_model must be one of: ridge, tree, both")

    df = meta.copy()
    df["y_true"] = y.values
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["qend_date", "asof_date"]).reset_index(drop=True)

    unique_qends = sorted(df["qend_date"].unique())
    preds_rows: List[pd.DataFrame] = []
    ridge_alpha = _choose_ridge_alpha(X, y, meta, model_spec)

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

        if "asof_close" not in X_test.columns or "ret_20d" not in X_test.columns:
            raise ValueError("Features must include asof_close and ret_20d for baselines.")

        asof_close = X_test["asof_close"].astype(float).to_numpy()
        ret_20d = X_test["ret_20d"].fillna(0.0).astype(float).to_numpy()

        y_train_delta = y_train.to_numpy() - X_train["asof_close"].astype(float).to_numpy()
        y_train_delta = pd.Series(y_train_delta, index=y_train.index)

        tree_model = train_model(X_train, y_train_delta, model_spec, model_type="tree")
        delta_pred_tree = predict(tree_model, X_test)

        y_train_delta_ridge = y_train.to_numpy() - X_train_ridge["asof_close"].astype(float).to_numpy()
        y_train_delta_ridge = pd.Series(y_train_delta_ridge, index=y_train.index)
        ridge_model = train_model(
            X_train_ridge,
            y_train_delta_ridge,
            model_spec,
            model_type="ridge",
            ridge_alpha=ridge_alpha,
        )
        delta_pred_ridge = predict(ridge_model, X_test_ridge)

        if delta_clip is not None:
            delta_pred_tree = np.clip(delta_pred_tree, -delta_clip, delta_clip)
            delta_pred_ridge = np.clip(delta_pred_ridge, -delta_clip, delta_clip)

        y_pred_tree = asof_close + delta_pred_tree
        y_pred_ridge = asof_close + delta_pred_ridge
        y_pred_naive = asof_close
        y_pred_mom = asof_close * (1.0 + ret_20d)

        use_model_tree = None
        use_model_ridge = None
        y_pred_hybrid_tree = None
        y_pred_hybrid_ridge = None
        if enable_hybrid:
            if hybrid_model in {"tree", "both"}:
                use_model_tree = np.abs(delta_pred_tree) >= hybrid_threshold
                y_pred_hybrid_tree = np.where(use_model_tree, y_pred_tree, y_pred_naive)
            if hybrid_model in {"ridge", "both"}:
                use_model_ridge = np.abs(delta_pred_ridge) >= hybrid_threshold
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
        if "weeks_to_qend" in X_test.columns:
            out["weeks_to_qend"] = X_test["weeks_to_qend"].to_numpy()
        preds_rows.append(out)

    if not preds_rows:
        raise ValueError("Backtest produced no predictions; check min_train_quarters or data length")

    preds = pd.concat(preds_rows, ignore_index=True)
    y_true = preds["y_true"].to_numpy()
    y_pred_tree = preds["y_pred_tree"].to_numpy()
    y_pred_ridge = preds["y_pred_ridge"].to_numpy()
    y_pred_naive = preds["y_pred_naive"].to_numpy()
    y_pred_mom = preds["y_pred_mom"].to_numpy()

    metrics = {
        "MAE_tree": _mae(y_true, y_pred_tree),
        "RMSE_tree": _rmse(y_true, y_pred_tree),
        "MAPE_tree": _mape(y_true, y_pred_tree),
        "MAE_ridge": _mae(y_true, y_pred_ridge),
        "RMSE_ridge": _rmse(y_true, y_pred_ridge),
        "MAPE_ridge": _mape(y_true, y_pred_ridge),
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

    return BacktestResult(predictions=preds, metrics=metrics)
