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
    )

    # Train final model on full dataset (delta target)
    y_delta = ds.y.to_numpy() - ds.X["asof_close"].astype(float).to_numpy()
    model = train_model(ds.X, pd.Series(y_delta, index=ds.y.index), cfg.model, model_type="tree")
    quantile_models = train_quantile_models(ds.X, pd.Series(y_delta, index=ds.y.index))

    artifacts = PipelineArtifacts(
        prices=prices,
        features_daily=feats,
        dataset_meta=ds.meta,
        dataset_X=ds.X,
        dataset_y=ds.y,
        quantile_models=quantile_models,
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

    delta_pred = model.pipeline.predict(X_live)[0]
    asof_close = float(X_live["asof_close"].iloc[0])
    forecast = asof_close + float(delta_pred)
    if artifacts.quantile_models is None:
        return pd.DataFrame({"asof_date": [asof_date], "forecast_qend": [forecast]})
    y_all_delta = artifacts.dataset_y.to_numpy() - artifacts.dataset_X["asof_close"].astype(float).to_numpy()
    calib_size = max(20, int(len(artifacts.dataset_X) * 0.2))
    if len(artifacts.dataset_X) > calib_size:
        fit_X = artifacts.dataset_X.iloc[:-calib_size]
        cal_X = artifacts.dataset_X.iloc[-calib_size:]
        fit_y = pd.Series(y_all_delta[:-calib_size], index=fit_X.index)
        cal_y = pd.Series(y_all_delta[-calib_size:], index=cal_X.index)
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
        cal_p50 = cal_asof + cal_pred["delta_p50"].to_numpy()
        q_hat_naive = conformal_qhat_residual(cal_y_level, cal_asof, cfg.model.interval_alpha)
        q_hat_p50 = conformal_qhat_residual(cal_y_level, cal_p50, cfg.model.interval_alpha)
    else:
        quantile_models = artifacts.quantile_models
        q_hat = 0.0
        q_hat_roll = 0.0
        q_hat_naive = 0.0
        q_hat_p50 = 0.0
    delta_q = predict_quantiles(quantile_models, X_live).iloc[0]
    p10_raw = asof_close + float(delta_q["delta_p10"])
    p50 = asof_close + float(delta_q["delta_p50"])
    p90_raw = asof_close + float(delta_q["delta_p90"])
    p10_cal, p90_cal = apply_conformal(np.array([p10_raw]), np.array([p90_raw]), q_hat)
    p10_cal_roll, p90_cal_roll = apply_conformal(
        np.array([p10_raw]),
        np.array([p90_raw]),
        q_hat_roll,
    )
    risk_score_raw = p90_raw - p10_raw
    risk_score_cal = float(p90_cal[0] - p10_cal[0])
    low_naive, high_naive = apply_residual_interval(np.array([asof_close]), q_hat_naive)
    low_p50, high_p50 = apply_residual_interval(np.array([p50]), q_hat_p50)
    return pd.DataFrame(
        {
            "asof_date": [asof_date],
            "forecast_qend": [forecast],
            "forecast_p50": [p50],
            "p10_raw": [p10_raw],
            "p90_raw": [p90_raw],
            "p10_cal": [float(p10_cal[0])],
            "p90_cal": [float(p90_cal[0])],
            "risk_score_raw": [risk_score_raw],
            "risk_score_cal": [risk_score_cal],
            "q_hat": [q_hat],
            "p10_cal_roll": [float(p10_cal_roll[0])],
            "p90_cal_roll": [float(p90_cal_roll[0])],
            "risk_score_roll": [float(p90_cal_roll[0] - p10_cal_roll[0])],
            "q_hat_roll": [q_hat_roll],
            "forecast_point_naive": [asof_close],
            "low_naive": [float(low_naive[0])],
            "high_naive": [float(high_naive[0])],
            "risk_score_naive": [float(high_naive[0] - low_naive[0])],
            "q_hat_naive": [q_hat_naive],
            "low_p50": [float(low_p50[0])],
            "high_p50": [float(high_p50[0])],
            "risk_score_p50": [float(high_p50[0] - low_p50[0])],
            "q_hat_p50": [q_hat_p50],
        }
    )
