from __future__ import annotations

import numpy as np
import pandas as pd


def add_market_features(cont: pd.DataFrame) -> pd.DataFrame:
    """
    cont index=date; requires close_badj and volume/open_interest.
    Creates weekly-style features (can be sampled later on Fridays).
    """
    df = cont.copy()

    px = df["close_badj"]
    df["ret_1d"] = px.pct_change()
    df["ret_5d"] = px.pct_change(5)
    df["ret_20d"] = px.pct_change(20)

    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["mom_20d"] = px / px.shift(20) - 1.0

    df["vol_chg_5d"] = df["volume"].pct_change(5).replace([np.inf, -np.inf], np.nan)
    df["oi_chg_5d"] = df["open_interest"].pct_change(5).replace([np.inf, -np.inf], np.nan)

    # Simple roll-proximity proxy: big changes in cum_adjustment flag roll regime
    df["roll_event"] = df["cum_adjustment"].diff().abs() > 1e-9
    df["roll_event_20d"] = df["roll_event"].rolling(20).max().astype(float)

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
