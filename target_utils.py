from __future__ import annotations

import numpy as np


def to_target(y_price: float | np.ndarray, asof_close: float | np.ndarray, target_mode: str) -> np.ndarray:
    y_price_arr = np.asarray(y_price, dtype=float)
    asof_arr = np.asarray(asof_close, dtype=float)
    if target_mode == "level":
        return y_price_arr
    if target_mode == "delta":
        return y_price_arr - asof_arr
    if target_mode == "log_return":
        if np.any(asof_arr <= 0):
            raise ValueError("asof_close must be > 0 for log_return target.")
        return np.log(y_price_arr / asof_arr)
    raise ValueError("target_mode must be one of: level, delta, log_return")


def from_target(y_target: float | np.ndarray, asof_close: float | np.ndarray, target_mode: str) -> np.ndarray:
    y_target_arr = np.asarray(y_target, dtype=float)
    asof_arr = np.asarray(asof_close, dtype=float)
    if target_mode == "level":
        return y_target_arr
    if target_mode == "delta":
        return asof_arr + y_target_arr
    if target_mode == "log_return":
        if np.any(asof_arr <= 0):
            raise ValueError("asof_close must be > 0 for log_return target.")
        return asof_arr * np.exp(y_target_arr)
    raise ValueError("target_mode must be one of: level, delta, log_return")
