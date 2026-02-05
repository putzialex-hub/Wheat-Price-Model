from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    prices index=date; requires close (or settlement).
    Creates weekly-style features (can be sampled later on Fridays).
    """
    df = prices.copy()

    px = df["close"]
    df["ret_1d"] = px.pct_change()
    df["ret_5d"] = px.pct_change(5)
    df["ret_20d"] = px.pct_change(20)

    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["mom_20d"] = px / px.shift(20) - 1.0

    return df


def latest_available_merge(
    base: pd.DataFrame,
    feature_table: pd.DataFrame,
    asof_col: str = "available_at",
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Merge a release-dated feature table onto daily base index using 'latest available' policy.

    feature_table must have:
      - available_at (datetime): when the value became available
      - value columns
    """
    ft = feature_table.copy()
    ft[asof_col] = pd.to_datetime(ft[asof_col]).dt.normalize()
    ft = ft.sort_values(asof_col)

    if value_cols is None:
        value_cols = [c for c in ft.columns if c not in {asof_col}]

    # Reindex by available_at, then forward-fill onto base dates
    ft = ft.set_index(asof_col)[value_cols]
    merged = base.join(ft.reindex(base.index).ffill())

    # Missing flags
    for c in value_cols:
        merged[f"{c}__is_missing"] = merged[c].isna().astype(float)

    return merged
