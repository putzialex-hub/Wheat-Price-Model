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


def compute_bucket_edges(vol: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vol, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return (float("inf"), float("inf"))
    t1 = float(np.quantile(v, 1 / 3))
    t2 = float(np.quantile(v, 2 / 3))
    return (t1, t2)


def assign_vol_bucket(vol: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    v = np.asarray(vol, dtype=float)
    t1, t2 = edges
    if not np.isfinite(t1) or not np.isfinite(t2):
        return np.full(v.shape, 1, dtype=int)
    buckets = np.full(v.shape, 1, dtype=int)
    valid = np.isfinite(v)
    buckets[(valid) & (v <= t1)] = 0
    buckets[(valid) & (v > t2)] = 2
    return buckets


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


def rolling_pool_append(pool: list[float], new_scores: np.ndarray, max_size: int) -> list[float]:
    scores = np.asarray(new_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    pool.extend(scores.tolist())
    if max_size > 0 and len(pool) > max_size:
        pool = pool[-max_size:]
    return pool
