from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import RollRule


@dataclass(frozen=True)
class ContinuousOutput:
    """
    Continuous series with metadata about roll and adjustments.
    """
    continuous: pd.DataFrame  # index=date; cols: close, settlement(optional), volume, open_interest, active_contract
    roll_events: pd.DataFrame  # date, from_contract, to_contract, price_diff_close, cum_adjustment


def _business_days_between(trading_days: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> int:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    i0 = trading_days.searchsorted(start, side="left")
    i1 = trading_days.searchsorted(end, side="left")
    return max(0, i1 - i0)


def build_backadjusted_continuous(
    contracts: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    expiry_calendar: Dict[str, pd.Timestamp],
    roll_rule: RollRule,
) -> ContinuousOutput:
    """
    Build a backadjusted continuous series from contract-level data.

    Expected contracts columns:
      - date (datetime)
      - contract (str)
      - close (float)
      - settlement (float, optional)
      - volume (float/int)
      - open_interest (float/int)

    expiry_calendar: contract -> expiry (last trading day / last trade date)

    Roll logic (deterministic):
      - candidate next contract is the nearest later expiry among available contracts
      - roll when next dominates current by volume OR OI for `consecutive_days`
      - and we are within `expiry_guard_business_days` of current expiry (unless forced)
      - forced roll if within `force_roll_business_days_before_expiry`
    """
    df = contracts.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "settlement" not in df.columns:
        df["settlement"] = np.nan

    required = {"date", "contract", "close", "volume", "open_interest"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"contracts missing columns: {sorted(missing)}")

    df = df.sort_values(["date", "contract"]).reset_index(drop=True)

    # Ensure we only use trading days
    df = df[df["date"].isin(trading_days)]
    if df.empty:
        raise ValueError("No contract rows align with trading days")

    # Pre-group by date
    by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        d: g.set_index("contract") for d, g in df.groupby("date", sort=True)
    }

    # Determine contract ordering by expiry
    def expiry_of(c: str) -> pd.Timestamp:
        if c not in expiry_calendar:
            raise ValueError(f"Missing expiry for contract '{c}' in expiry_calendar")
        return pd.Timestamp(expiry_calendar[c]).normalize()

    all_contracts = sorted(df["contract"].unique(), key=expiry_of)

    # Start with earliest contract that has data
    first_date = min(by_date.keys())
    start_slice = by_date[first_date]
    start_contract = start_slice.index.intersection(all_contracts)
    if len(start_contract) == 0:
        raise ValueError("No known contracts on start date")
    active = start_contract[0]

    cum_adjust = 0.0
    consec_counter = 0

    cont_rows: List[Tuple] = []
    roll_rows: List[Tuple] = []

    def next_contract(current: str) -> Optional[str]:
        exp = expiry_of(current)
        later = [c for c in all_contracts if expiry_of(c) > exp]
        return later[0] if later else None

    for d in trading_days:
        if d not in by_date:
            # no data day; skip (or could forward-fill but dangerous for prices)
            continue

        day_slice = by_date[d]

        if active not in day_slice.index:
            # If active missing, try to advance to next available contract with data
            nc = active
            while nc is not None and nc not in day_slice.index:
                nc = next_contract(nc)
            if nc is None:
                continue
            active = nc
            consec_counter = 0

        nc = next_contract(active)
        active_exp = expiry_of(active)
        bdays_to_exp = _business_days_between(trading_days, d, active_exp)

        # Forced roll very close to expiry if next exists and has data
        forced = (nc is not None) and (bdays_to_exp <= roll_rule.force_roll_business_days_before_expiry)

        should_roll = False
        if nc is not None and nc in day_slice.index:
            curr = day_slice.loc[active]
            nxt = day_slice.loc[nc]

            dominates = (nxt["volume"] > curr["volume"]) or (nxt["open_interest"] > curr["open_interest"])
            if dominates:
                consec_counter += 1
            else:
                consec_counter = 0

            guard_ok = bdays_to_exp <= roll_rule.expiry_guard_business_days
            should_roll = forced or (guard_ok and consec_counter >= roll_rule.consecutive_days)

        if should_roll and nc is not None:
            # Roll at end of day: compute adjustment using close
            curr_close = float(day_slice.loc[active]["close"])
            next_close = float(day_slice.loc[nc]["close"])
            price_diff = curr_close - next_close  # additive backadjust

            cum_adjust += price_diff
            roll_rows.append((d, active, nc, price_diff, cum_adjust))

            active = nc
            consec_counter = 0

            # refresh slice after roll (same day data for new active)
            if active not in day_slice.index:
                continue

        row = day_slice.loc[active]
        cont_rows.append(
            (
                d,
                float(row["close"]) + cum_adjust,
                (float(row["settlement"]) + cum_adjust) if pd.notna(row["settlement"]) else np.nan,
                float(row["volume"]),
                float(row["open_interest"]),
                active,
                cum_adjust,
            )
        )

    cont = pd.DataFrame(
        cont_rows,
        columns=["date", "close_badj", "settlement_badj", "volume", "open_interest", "active_contract", "cum_adjustment"],
    ).set_index("date").sort_index()

    rolls = pd.DataFrame(
        roll_rows,
        columns=["date", "from_contract", "to_contract", "price_diff_close", "cum_adjustment"],
    ).sort_values("date")

    return ContinuousOutput(continuous=cont, roll_events=rolls)
