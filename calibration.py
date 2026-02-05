from __future__ import annotations

import numpy as np


def conformal_qhat(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray, alpha: float) -> float:
    s = np.maximum(p10 - y_true, y_true - p90)
    s = np.maximum(s, 0.0)
    try:
        return float(np.quantile(s, 1 - alpha, method="higher"))
    except TypeError:
        return float(np.quantile(s, 1 - alpha, interpolation="higher"))


def apply_conformal(p10: np.ndarray, p90: np.ndarray, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    return p10 - q_hat, p90 + q_hat
