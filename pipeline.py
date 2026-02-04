from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import AppConfig
from .continuous import build_backadjusted_continuous
from .features import add_market_features, latest_available_merge
from .dataset import build_quarter_end_dataset
from .model import train_model, TrainedModel
from .backtest import walk_forward_by_quarter


@dataclass(frozen=True)
class PipelineArtifacts:
    continuous: pd.DataFrame
    roll_events: pd.DataFrame
    features_daily: pd.DataFrame


def load_contracts_table(path: Path) -> pd.DataFrame:
    """
    Load contract-level table.
    Supports CSV and Parquet.
    """
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_optional_table(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def run_training_pipeline(
    cfg: AppConfig,
    expiry_calendar: Dict[str, pd.Timestamp],
) -> tuple[TrainedModel, PipelineArtifacts, dict]:
    """
    End-to-end:
      - load data
      - build continuous
      - build features (market + optional lag-safe macro/fundamentals)
      - build supervised dataset
      - train final model on all rows
      - run walk-forward backtest for diagnostics
    """
    contracts = load_contracts_table(cfg.data.contracts_path)

    # Derive trading days from available dates in contracts
    trading_days = pd.DatetimeIndex(pd.to_datetime(contracts["date"]).dt.normalize().unique()).sort_values()

    cont_out = build_backadjusted_continuous(
        contracts=contracts,
        trading_days=trading_days,
        expiry_calendar=expiry_calendar,
        roll_rule=cfg.roll,
    )

    cont = cont_out.continuous
    feats = add_market_features(cont)

    macro = load_optional_table(cfg.data.macro_path)
    if macro is not None:
        feats = latest_available_merge(feats, macro, asof_col="available_at")

    fundamentals = load_optional_table(cfg.data.fundamentals_path)
    if fundamentals is not None:
        feats = latest_available_merge(feats, fundamentals, asof_col="available_at")

    ds = build_quarter_end_dataset(
        features_daily=feats,
        cont_daily=cont,
        trading_days=trading_days,
        spec=cfg.forecast,
    )

    # Backtest
    bt = walk_forward_by_quarter(ds.X, ds.y, ds.meta, cfg.model)

    # Train final model on full dataset
    model = train_model(ds.X, ds.y, cfg.model)

    artifacts = PipelineArtifacts(
        continuous=cont,
        roll_events=cont_out.roll_events,
        features_daily=feats,
    )

    return model, artifacts, bt.metrics


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

    # Only run on Fridays for the weekly update job
    if last_date.weekday() != 4:
        raise ValueError(f"Latest date {last_date.date()} is not a Friday; weekly job should run after Friday close.")

    # Minimal input row
    X_live = feats.loc[[last_date]].copy()
    # weeks_to_qend is unknown here without computing quarter-end; for live, keep 0 or compute externally
    X_live["weeks_to_qend"] = 0

    y_pred = model.pipeline.predict(X_live)[0]
    return pd.DataFrame({"asof_date": [last_date], "forecast_qend": [float(y_pred)]})
