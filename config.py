from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataPaths:
    """
    Default: file-based inputs (CSV/Parquet).
    You can replace with vendor adapters later.
    """
    # Contract-level daily data (recommended)
    # Expected columns: date, contract, close, settlement(optional), volume, open_interest
    contracts_path: Path

    # Optional: front/next series already provided by vendor (less ideal than contract-level)
    chain_path: Optional[Path] = None

    # Optional: macro/fundamentals feature tables (already lag-safe / release-dated)
    macro_path: Optional[Path] = None
    fundamentals_path: Optional[Path] = None


@dataclass(frozen=True)
class RollRule:
    """
    Deterministic roll rule:
    - prefer volume crossover
    - fallback to open interest crossover
    - require consecutive days to avoid noise
    - guardrail around expiry to prevent early rolls
    """
    consecutive_days: int = 2
    expiry_guard_business_days: int = 15  # don't roll earlier than this unless forced
    force_roll_business_days_before_expiry: int = 3  # must roll by this point


@dataclass(frozen=True)
class ForecastSpec:
    """
    Quarter-end target and as-of definition.
    """
    # Quarter-end = last trading day of the quarter
    # As-Of = last Friday close before the trading day that is 10 trading days before quarter-end
    trading_days_before_qend: int = 10

    # Use "settlement" for target when available, otherwise fall back to "close"
    prefer_settlement_target: bool = True

    # Weekly updates: include multiple Fridays per quarter for training (recommended)
    include_weekly_updates: bool = True
    max_weeks_before_qend: int = 8  # include Fridays up to N weeks before q-end
    min_business_days_before_qend: int = 3  # avoid too-close-to-close rows


@dataclass(frozen=True)
class ModelSpec:
    random_state: int = 42
    # Small, sane default model; tune later.
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 3
    ridge_alpha: float = 10.0
    ridge_alpha_grid: Optional[list[float]] = None
    interval_alpha: float = 0.2
    calibration_mode: str = "pooled"
    min_pool_size: int = 50
    rolling_folds: int = 12
    rolling_calibration_size: int = 200


@dataclass(frozen=True)
class AppConfig:
    data: DataPaths
    roll: RollRule = RollRule()
    forecast: ForecastSpec = ForecastSpec()
    model: ModelSpec = ModelSpec()
