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


def walk_forward_by_quarter(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    model_spec: ModelSpec,
    min_train_quarters: int = 12,
    delta_clip: float | None = 80.0,
    hybrid_threshold: float = 10.0,
    enable_hybrid: bool = True,
) -> BacktestResult:
    """
    Walk-forward:
      train on all rows with qend_date < current qend_date
      test on current quarter (all asof rows mapping to that quarter-end)
    """
    df = meta.copy()
    df["y_true"] = y.values
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["qend_date", "asof_date"]).reset_index(drop=True)

    unique_qends = sorted(df["qend_date"].unique())
    preds_rows: List[pd.DataFrame] = []

    for i, qend in enumerate(unique_qends):
        train_qends = unique_qends[:i]
        if len(train_qends) < min_train_quarters:
            continue

        train_idx = df[df["qend_date"].isin(train_qends)]["row_id"].values
        test_idx = df[df["qend_date"] == qend]["row_id"].values

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if "asof_close" not in X_test.columns or "ret_20d" not in X_test.columns:
            raise ValueError("Features must include asof_close and ret_20d for baselines.")

        asof_close = X_test["asof_close"].astype(float).to_numpy()
        ret_20d = X_test["ret_20d"].fillna(0.0).astype(float).to_numpy()

        y_train_delta = y_train.to_numpy() - X_train["asof_close"].astype(float).to_numpy()
        y_train_delta = pd.Series(y_train_delta, index=y_train.index)

        tree_model = train_model(X_train, y_train_delta, model_spec, model_type="tree")
        delta_pred_tree = predict(tree_model, X_test)

        ridge_model = train_model(X_train, y_train_delta, model_spec, model_type="ridge")
        delta_pred_ridge = predict(ridge_model, X_test)

        if delta_clip is not None:
            delta_pred_tree = np.clip(delta_pred_tree, -delta_clip, delta_clip)
            delta_pred_ridge = np.clip(delta_pred_ridge, -delta_clip, delta_clip)

        y_pred_tree = asof_close + delta_pred_tree
        y_pred_ridge = asof_close + delta_pred_ridge
        y_pred_naive = asof_close
        y_pred_mom = asof_close * (1.0 + ret_20d)

        if enable_hybrid:
            use_model_tree = np.abs(delta_pred_tree) >= hybrid_threshold
            use_model_ridge = np.abs(delta_pred_ridge) >= hybrid_threshold
            y_pred_hybrid_tree = np.where(use_model_tree, y_pred_tree, y_pred_naive)
            y_pred_hybrid_ridge = np.where(use_model_ridge, y_pred_ridge, y_pred_naive)
        else:
            use_model_tree = np.ones_like(y_pred_tree, dtype=bool)
            use_model_ridge = np.ones_like(y_pred_ridge, dtype=bool)
            y_pred_hybrid_tree = y_pred_tree
            y_pred_hybrid_ridge = y_pred_ridge

        out = df[df["row_id"].isin(test_idx)].copy()
        out["y_pred_tree"] = y_pred_tree
        out["y_pred_ridge"] = y_pred_ridge
        out["y_pred_hybrid_tree"] = y_pred_hybrid_tree
        out["y_pred_hybrid_ridge"] = y_pred_hybrid_ridge
        out["y_pred_naive"] = y_pred_naive
        out["y_pred_mom"] = y_pred_mom
        out["use_model_tree"] = use_model_tree.astype(int)
        out["use_model_ridge"] = use_model_ridge.astype(int)
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
    y_pred_hybrid_tree = preds["y_pred_hybrid_tree"].to_numpy()
    y_pred_hybrid_ridge = preds["y_pred_hybrid_ridge"].to_numpy()
    y_pred_naive = preds["y_pred_naive"].to_numpy()
    y_pred_mom = preds["y_pred_mom"].to_numpy()

    metrics = {
        "MAE_tree": _mae(y_true, y_pred_tree),
        "RMSE_tree": _rmse(y_true, y_pred_tree),
        "MAPE_tree": _mape(y_true, y_pred_tree),
        "MAE_ridge": _mae(y_true, y_pred_ridge),
        "RMSE_ridge": _rmse(y_true, y_pred_ridge),
        "MAPE_ridge": _mape(y_true, y_pred_ridge),
        "MAE_hybrid_tree": _mae(y_true, y_pred_hybrid_tree),
        "RMSE_hybrid_tree": _rmse(y_true, y_pred_hybrid_tree),
        "MAPE_hybrid_tree": _mape(y_true, y_pred_hybrid_tree),
        "MAE_hybrid_ridge": _mae(y_true, y_pred_hybrid_ridge),
        "RMSE_hybrid_ridge": _rmse(y_true, y_pred_hybrid_ridge),
        "MAPE_hybrid_ridge": _mape(y_true, y_pred_hybrid_ridge),
        "MAE_naive": _mae(y_true, y_pred_naive),
        "RMSE_naive": _rmse(y_true, y_pred_naive),
        "MAPE_naive": _mape(y_true, y_pred_naive),
        "MAE_mom": _mae(y_true, y_pred_mom),
        "RMSE_mom": _rmse(y_true, y_pred_mom),
        "MAPE_mom": _mape(y_true, y_pred_mom),
    }

    return BacktestResult(predictions=preds, metrics=metrics)
