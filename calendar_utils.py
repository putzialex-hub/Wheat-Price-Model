from __future__ import annotations

import pandas as pd


def last_trading_day_of_quarter(trading_days: pd.DatetimeIndex, quarter: pd.Period) -> pd.Timestamp:
    """
    Quarter-end = last available trading day within the quarter.
    """
    q_start = quarter.start_time.normalize()
    q_end = quarter.end_time.normalize()
    mask = (trading_days >= q_start) & (trading_days <= q_end)
    if not mask.any():
        raise ValueError(f"No trading days found for quarter {quarter}")
    return trading_days[mask][-1]


def nth_trading_day_before(trading_days: pd.DatetimeIndex, date: pd.Timestamp, n: int) -> pd.Timestamp:
    """
    Return the trading day that is n trading days before 'date'.
    """
    date = pd.Timestamp(date).normalize()
    idx = trading_days.searchsorted(date, side="left")
    if idx == 0:
        raise ValueError("Date is before first trading day")
    # If date is not a trading day, idx points to next trading day; we want prior reference point for "before"
    if idx >= len(trading_days) or trading_days[idx] != date:
        idx = idx - 1
    target_idx = idx - n
    if target_idx < 0:
        raise ValueError("Not enough trading days before date")
    return trading_days[target_idx]


def previous_friday(trading_days: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    """
    Last trading day that is a Friday on or before 'date'.
    If 'date' itself is a Friday trading day, it returns 'date'.
    """
    date = pd.Timestamp(date).normalize()
    idx = trading_days.searchsorted(date, side="right") - 1
    while idx >= 0:
        d = trading_days[idx]
        if d.weekday() == 4:  # Friday
            return d
        idx -= 1
    raise ValueError("No Friday trading day found before date")


def quarters_from_dates(dates: pd.DatetimeIndex) -> pd.PeriodIndex:
    return dates.to_period("Q")
