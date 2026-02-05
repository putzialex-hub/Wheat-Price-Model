from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from config import AppConfig
from data_loader import load_bl2c1_csv
from features import add_market_features, latest_available_merge
from dataset import build_quarter_end_dataset
from model import train_model, TrainedModel
from backtest import BacktestResult, walk_forward_by_quarter


@dataclass(frozen=True)
class PipelineArtifacts:
    prices: pd.DataFrame
    features_daily: pd.DataFrame
    dataset_meta: pd.DataFrame


def load_optional_table(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    return pd.read_csv(path)


def run_training_pipeline(
    cfg: AppConfig,
    price_csv_path: str,
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

    macro = load_optional_table(cfg.data.macro_path)
    if macro is not None:
        feats = latest_available_merge(feats, macro, asof_col="available_at")

    fundamentals = load_optional_table(cfg.data.fundamentals_path)
    if fundamentals is not None:
        feats = latest_available_merge(feats, fundamentals, asof_col="available_at")

    ds = build_quarter_end_dataset(
        features_daily=feats,
        cont_daily=prices,
        trading_days=trading_days,
        spec=cfg.forecast,
    )

    # Backtest
    bt = walk_forward_by_quarter(ds.X, ds.y, ds.meta, cfg.model)

    # Train final model on full dataset
    model = train_model(ds.X, ds.y, cfg.model)

    artifacts = PipelineArtifacts(
        prices=prices,
        features_daily=feats,
        dataset_meta=ds.meta,
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

    y_pred = model.pipeline.predict(X_live)[0]
    return pd.DataFrame({"asof_date": [asof_date], "forecast_qend": [float(y_pred)]})
