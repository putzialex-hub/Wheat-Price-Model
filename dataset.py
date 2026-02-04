from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .calendar_utils import (
    last_trading_day_of_quarter,
    nth_trading_day_before,
    previous_friday,
    quarters_from_dates,
)
from .config import ForecastSpec


@dataclass(frozen=True)
class DatasetOutput:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame  # quarter, asof_date, qend_date, weeks_to_qend


def _fridays_between(dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    mask = (dates >= start) & (dates <= end) & (dates.weekday == 4)
    return list(dates[mask])


def build_quarter_end_dataset(
    features_daily: pd.DataFrame,
    cont_daily: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    spec: ForecastSpec,
) -> DatasetOutput:
    """
    Build supervised dataset:
      X(as-of Friday) -> y(quarter-end settlement or close)

    features_daily: daily features (index=date)
    cont_daily: continuous series (index=date) with settlement_badj/close_badj
    """
    dates = features_daily.index.intersection(trading_days).sort_values()
    if len(dates) == 0:
        raise ValueError("No trading-day-aligned features")

    quarters = pd.period_range(quarters_from_dates(dates).min(), quarters_from_dates(dates).max(), freq="Q")
    rows_X: List[pd.Series] = []
    rows_y: List[float] = []
    rows_meta: List[Tuple] = []

    for q in quarters:
        try:
            qend = last_trading_day_of_quarter(trading_days, q)
        except ValueError:
            continue

        # Determine anchor day: 10 trading days before qend
        try:
            anchor = nth_trading_day_before(trading_days, qend, spec.trading_days_before_qend)
        except ValueError:
            continue

        # Primary as-of date
        try:
            asof_primary = previous_friday(trading_days, anchor)
        except ValueError:
            continue

        # Optionally include multiple Fridays up to quarter-end for weekly updates
        if spec.include_weekly_updates:
            start = qend - pd.Timedelta(days=7 * spec.max_weeks_before_qend)
            fridays = _fridays_between(trading_days, start, qend)
            # Keep only Fridays that are on/after primary as-of (so live pattern matches) and not too close to qend
            candidates = []
            for f in fridays:
                bdays_to_qend = len(trading_days[(trading_days > f) & (trading_days <= qend)])
                if f >= asof_primary and bdays_to_qend >= spec.min_business_days_before_qend:
                    candidates.append(f)
            asof_dates = candidates if candidates else [asof_primary]
        else:
            asof_dates = [asof_primary]

        # Determine target value at qend
        if spec.prefer_settlement_target and pd.notna(cont_daily.loc[qend, "settlement_badj"]):
            y_val = float(cont_daily.loc[qend, "settlement_badj"])
        else:
            y_val = float(cont_daily.loc[qend, "close_badj"])

        for asof in asof_dates:
            if asof not in features_daily.index:
                continue
            x = features_daily.loc[asof].copy()
            weeks_to_qend = int(round((qend - asof).days / 7.0))
            x["weeks_to_qend"] = weeks_to_qend

            rows_X.append(x)
            rows_y.append(y_val)
            rows_meta.append((str(q), asof, qend, weeks_to_qend))

    if not rows_X:
        raise ValueError("No dataset rows generated; check calendar/inputs")

    X = pd.DataFrame(rows_X)
    y = pd.Series(rows_y, name="target_qend_settlement")
    meta = pd.DataFrame(rows_meta, columns=["quarter", "asof_date", "qend_date", "weeks_to_qend"])

    # Basic cleanup: drop columns with all-missing
    X = X.dropna(axis=1, how="all")

    return DatasetOutput(X=X, y=y, meta=meta)
