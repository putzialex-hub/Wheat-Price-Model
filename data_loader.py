from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


_GERMAN_MONTH_MAP: Dict[str, str] = {
    "Jän": "Jan",
    "Feb": "Feb",
    "Mär": "Mar",
    "Mrz": "Mar",
    "Apr": "Apr",
    "Mai": "May",
    "Jun": "Jun",
    "Jul": "Jul",
    "Aug": "Aug",
    "Sep": "Sep",
    "Okt": "Oct",
    "Nov": "Nov",
    "Dez": "Dec",
}


@dataclass(frozen=True)
class ValidationResult:
    warnings: List[str]


def _normalize_german_months(series: pd.Series) -> pd.Series:
    out = series.astype(str)
    for de, en in _GERMAN_MONTH_MAP.items():
        out = out.str.replace(f"-{de}-", f"-{en}-", regex=False)
    return out


def _validate_price_series(df: pd.DataFrame) -> ValidationResult:
    warnings: List[str] = []

    if df["date"].isna().any():
        raise ValueError("Found empty dates in input data.")

    if df["date"].duplicated().any():
        dupes = df.loc[df["date"].duplicated(), "date"].dt.date.unique()
        raise ValueError(f"Duplicate dates found in input data: {dupes[:5]}")

    if not df["date"].is_monotonic_increasing:
        raise ValueError("Dates are not sorted in ascending order.")

    if df[["close", "settlement"]].isna().any().any():
        raise ValueError("Found empty Close/Settlement Price values.")

    if (df[["close", "settlement"]] <= 0).any().any():
        raise ValueError("Found non-positive Close/Settlement Price values.")

    day_gaps = df["date"].diff().dt.days.fillna(1)
    if (day_gaps > 7).any():
        gap_dates = df.loc[day_gaps > 7, "date"].dt.date.astype(str).tolist()
        warnings.append(
            f"Detected gaps larger than 7 days at: {', '.join(gap_dates[:5])}."
        )

    return ValidationResult(warnings=warnings)


def _load_continuation_csv(path: str | Path) -> pd.DataFrame:
    """
    Load single-series LSEG continuation (Close/Settlement).
    Returns a DataFrame indexed by date with columns: close, settlement.
    """
    csv_path = Path(path)
    raw = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

    required_cols = {"Exchange Date", "Close", "Settlement Price"}
    missing = required_cols - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dates = _normalize_german_months(raw["Exchange Date"])
    parsed = pd.to_datetime(dates, format="%d-%b-%Y", errors="coerce")
    if parsed.isna().any():
        bad = raw.loc[parsed.isna(), "Exchange Date"].head(5).tolist()
        raise ValueError(f"Failed to parse Exchange Date values: {bad}")

    df = pd.DataFrame(
        {
            "date": parsed.dt.normalize(),
            "close": pd.to_numeric(raw["Close"], errors="coerce"),
            "settlement": pd.to_numeric(raw["Settlement Price"], errors="coerce"),
        }
    ).sort_values("date")

    result = _validate_price_series(df)
    if result.warnings:
        warning_text = "\n".join(f"Warning: {w}" for w in result.warnings)
        print(warning_text)

    return df.set_index("date")


def load_bl2c1_csv(path: str | Path) -> pd.DataFrame:
    """
    Load single-series LSEG BL2c1 continuation (Close/Settlement).
    Returns a DataFrame indexed by date with columns: close, settlement.
    """
    return _load_continuation_csv(path)


def load_bl2c2_csv(path: str | Path) -> pd.DataFrame:
    """
    Load single-series LSEG BL2c2 continuation (Close/Settlement).
    Returns a DataFrame indexed by date with columns: close_c2, settlement_c2.
    """
    df = _load_continuation_csv(path)
    return df.rename(columns={"close": "close_c2", "settlement": "settlement_c2"})
