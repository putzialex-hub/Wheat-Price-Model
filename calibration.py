from __future__ import annotations

import numpy as np


def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    s = np.maximum(s, 0.0)
    n = s.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(s, k - 1)[k - 1])


def apply_conformal(p10: np.ndarray, p90: np.ndarray, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    return p10 - q_hat, p90 + q_hat


def conformal_qhat_residual(y_true: np.ndarray, y_center: np.ndarray, alpha: float) -> float:
    s = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_center, dtype=float))
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    n = s.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(s, k - 1)[k - 1])


def apply_residual_interval(center: np.ndarray, q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    return center - q_hat, center + q_hat
