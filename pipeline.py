from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import AppConfig
from data_loader import load_bl2c1_csv, load_bl2c2_csv
import features
from features import add_market_features, add_macro_return_features, latest_available_merge
from dataset import build_quarter_end_dataset
from target_utils import from_target
from calibration import (
    apply_conformal,
    apply_residual_interval,
    conformal_qhat,
    conformal_qhat_residual,
    rolling_pool_append,
)
from model import (
    TrainedModel,
    QuantileModels,
    predict_quantiles,
    train_model,
    train_quantile_models,
)
from backtest import BacktestResult, walk_forward_by_quarter


@dataclass(frozen=True)
class PipelineArtifacts:
    prices: pd.DataFrame
    features_daily: pd.DataFrame
    dataset_meta: pd.DataFrame
    dataset_X: pd.DataFrame
    dataset_y: pd.Series
    quantile_models: QuantileModels | None
    target_mode: str


def load_optional_table(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.rename(columns=lambda c: str(c).lstrip("\ufeff").strip())
    return df


def run_training_pipeline(
    cfg: AppConfig,
    price_csv_path: str,
    price_csv_c2_path: str | None = None,
    primary_only: bool = True,
    hybrid_threshold: float = 10.0,
    enable_hybrid: bool = True,
    hybrid_model: str = "ridge",
    target_mode: str = "level",
) -> tuple[TrainedModel, PipelineArtifacts, BacktestResult]:
    """
    End-to-end:
      - load data
      - build features (market + optional lag-safe macro/fundamentals)
      - build supervised dataset
      - train final model on all rows
      - run walk-forward backtest for diagnostics
    """
    prices = load_bl2c1_csv(price_csv_path)

    trading_days = pd.DatetimeIndex(prices.index).sort_values()
    feats = add_market_features(prices)

    if price_csv_c2_path:
        c2_path = Path(price_csv_c2_path)
        if c2_path.exists():
            c2 = load_bl2c2_csv(c2_path)
            feats = feats.join(c2[["close_c2", "settlement_c2"]], how="left")
            feats["close_c2"] = pd.to_numeric(feats["close_c2"], errors="coerce")
            feats["close_c2__is_missing"] = feats["close_c2"].isna().astype(float)

    macro = load_optional_table(cfg.data.macro_path)
    if macro is not None:
        if "available_at" not in macro.columns:
            print("Warning: macro CSV missing 'available_at' column; skipping macro merge.")
        else:
            feats = latest_available_merge(feats, macro, asof_col="available_at")

    fundamentals = load_optional_table(cfg.data.fundamentals_path)
    if fundamentals is not None:
        if "available_at" not in fundamentals.columns:
            print("Warning: fundamentals CSV missing 'available_at' column; skipping merge.")
        else:
            feats = latest_available_merge(feats, fundamentals, asof_col="available_at")

    feats = add_macro_return_features(feats)
    add_term_structure = getattr(features, "add_term_structure_features", None)
    if add_term_structure is not None:
        feats = add_term_structure(feats)

    ds = build_quarter_end_dataset(
        features_daily=feats,
        cont_daily=prices,
        trading_days=trading_days,
        spec=cfg.forecast,
        primary_only=primary_only,
        target_mode=target_mode,
    )

    # Backtest
    bt = walk_forward_by_quarter(
        ds.X,
        ds.y,
        ds.meta,
        cfg.model,
        hybrid_threshold=hybrid_threshold,
        enable_hybrid=enable_hybrid,
        hybrid_model=hybrid_model,
        target_mode=target_mode,
    )

    # Train final model on full dataset (target-mode)
    model = train_model(ds.X, ds.y, cfg.model, model_type="tree")
    quantile_models = train_quantile_models(ds.X, ds.y)

    artifacts = PipelineArtifacts(
        prices=prices,
        features_daily=feats,
        dataset_meta=ds.meta,
        dataset_X=ds.X,
        dataset_y=ds.y,
        quantile_models=quantile_models,
        target_mode=target_mode,
    )

    return model, artifacts, bt


def forecast_next_quarter_end(
    model: TrainedModel,
    artifacts: PipelineArtifacts,
    cfg: AppConfig,
) -> pd.DataFrame:
    """
    Produce a forecast for the next available quarter-end target using the latest Friday cut.
    This follows the same dataset construction logic, but uses the most recent feature row.
    """
    feats = artifacts.features_daily.copy()
    last_date = feats.index.max()
    friday_dates = feats.index[feats.index.weekday == 4]
    if friday_dates.empty:
        raise ValueError("No Friday dates available in features; cannot compute live forecast.")
    asof_date = friday_dates[friday_dates <= last_date].max()
    if (last_date - asof_date).days > 7:
        print(
            "Warning: latest date is more than 7 days after the last available Friday "
            f"({asof_date.date()})."
        )

    # Minimal input row
    X_live = feats.loc[[asof_date]].copy()
    # weeks_to_qend is unknown here without computing quarter-end; for live, keep 0 or compute externally
    X_live["weeks_to_qend"] = 0

    target_mode = artifacts.target_mode
    pred_target = model.pipeline.predict(X_live)[0]
    asof_close = float(X_live["asof_close"].iloc[0])
    forecast = float(from_target(pred_target, asof_close, target_mode))
    if artifacts.quantile_models is None:
        return pd.DataFrame({"asof_date": [asof_date], "forecast_qend": [forecast]})
    y_all_target = artifacts.dataset_y.to_numpy()
    calib_size = max(20, int(len(artifacts.dataset_X) * 0.2))
    if len(artifacts.dataset_X) > calib_size:
        fit_X = artifacts.dataset_X.iloc[:-calib_size]
        cal_X = artifacts.dataset_X.iloc[-calib_size:]
        fit_y = pd.Series(y_all_target[:-calib_size], index=fit_X.index)
        cal_y = pd.Series(y_all_target[-calib_size:], index=cal_X.index)
        quantile_models = train_quantile_models(fit_X, fit_y)
        cal_pred = predict_quantiles(quantile_models, cal_X)
        cal_scores = np.maximum(
            cal_pred["delta_p10"].to_numpy() - cal_y.to_numpy(),
            cal_y.to_numpy() - cal_pred["delta_p90"].to_numpy(),
        )
        cal_scores = np.maximum(cal_scores, 0.0)
        q_hat = conformal_qhat(cal_scores, cfg.model.interval_alpha)
        q_hat_roll = q_hat
        rolling_pool: list[float] = []
        rolling_pool = rolling_pool_append(rolling_pool, cal_scores, cfg.model.rolling_calibration_size)
        if len(rolling_pool) >= cfg.model.min_pool_size:
            q_hat_roll = conformal_qhat(np.asarray(rolling_pool), cfg.model.interval_alpha)
        else:
            q_hat_roll = q_hat
        cal_asof = cal_X["asof_close"].astype(float).to_numpy()
        cal_y_level = artifacts.dataset_y.to_numpy()[-calib_size:]
        cal_p50_target = cal_pred["delta_p50"].to_numpy()
        cal_naive_target = np.zeros_like(cal_p50_target)
        if target_mode == "level":
            cal_naive_target = cal_asof
        q_hat_naive = conformal_qhat_residual(cal_y.to_numpy(), cal_naive_target, cfg.model.interval_alpha)
        q_hat_p50 = conformal_qhat_residual(cal_y.to_numpy(), cal_p50_target, cfg.model.interval_alpha)
    else:
        quantile_models = artifacts.quantile_models
        q_hat = 0.0
        q_hat_roll = 0.0
        q_hat_naive = 0.0
        q_hat_p50 = 0.0
    delta_q = predict_quantiles(quantile_models, X_live).iloc[0]
    p10_target = float(delta_q["delta_p10"])
    p50_target = float(delta_q["delta_p50"])
    p90_target = float(delta_q["delta_p90"])
    p10_raw = float(from_target(p10_target, asof_close, target_mode))
    p50 = float(from_target(p50_target, asof_close, target_mode))
    p90_raw = float(from_target(p90_target, asof_close, target_mode))
    p10_cal_t, p90_cal_t = apply_conformal(np.array([p10_target]), np.array([p90_target]), q_hat)
    p10_cal_roll_t, p90_cal_roll_t = apply_conformal(
        np.array([p10_target]),
        np.array([p90_target]),
        q_hat_roll,
    )
    p10_cal = float(from_target(p10_cal_t[0], asof_close, target_mode))
    p90_cal = float(from_target(p90_cal_t[0], asof_close, target_mode))
    p10_cal_roll = float(from_target(p10_cal_roll_t[0], asof_close, target_mode))
    p90_cal_roll = float(from_target(p90_cal_roll_t[0], asof_close, target_mode))
    risk_score_raw = p90_raw - p10_raw
    risk_score_cal = p90_cal - p10_cal
    risk_score_roll = p90_cal_roll - p10_cal_roll
    naive_target = 0.0 if target_mode in {"delta", "log_return"} else asof_close
    low_naive_t, high_naive_t = apply_residual_interval(np.array([naive_target]), q_hat_naive)
    low_naive = float(from_target(low_naive_t[0], asof_close, target_mode))
    high_naive = float(from_target(high_naive_t[0], asof_close, target_mode))
    low_p50_t, high_p50_t = apply_residual_interval(np.array([p50_target]), q_hat_p50)
    low_p50 = float(from_target(low_p50_t[0], asof_close, target_mode))
    high_p50 = float(from_target(high_p50_t[0], asof_close, target_mode))
    return pd.DataFrame(
        {
            "asof_date": [asof_date],
            "forecast_qend": [forecast],
            "forecast_p50": [p50],
            "forecast_target_p50": [p50_target],
            "p10_raw": [p10_raw],
            "p90_raw": [p90_raw],
            "p10_cal": [p10_cal],
            "p90_cal": [p90_cal],
            "risk_score_raw": [risk_score_raw],
            "risk_score_cal": [risk_score_cal],
            "q_hat": [q_hat],
            "p10_cal_roll": [p10_cal_roll],
            "p90_cal_roll": [p90_cal_roll],
            "risk_score_roll": [risk_score_roll],
            "q_hat_roll": [q_hat_roll],
            "forecast_point_naive": [asof_close],
            "low_naive": [low_naive],
            "high_naive": [high_naive],
            "risk_score_naive": [high_naive - low_naive],
            "q_hat_naive": [q_hat_naive],
            "low_p50": [low_p50],
            "high_p50": [high_p50],
            "risk_score_p50": [high_p50 - low_p50],
            "q_hat_p50": [q_hat_p50],
        }
    )
