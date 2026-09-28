"""Interval estimates used throughout the study."""
from __future__ import annotations

import numpy as np
from scipy import stats


def clopper_pearson(k: int, size: int, conf: float = 0.99):
    """One-sided exact binomial limits (lower, upper) at confidence `conf`."""
    alpha = 1.0 - conf
    lower = 0.0 if k == 0 else float(stats.beta.ppf(alpha, k, size - k + 1))
    upper = 1.0 if k == size else float(stats.beta.ppf(1.0 - alpha, k + 1, size - k))
    return lower, upper


def bootstrap_ci(data, statistic=np.median, n_boot: int = 2000, conf: float = 0.95,
                 rng: np.random.Generator | None = None):
    """Point estimate and percentile bootstrap interval of `statistic`."""
    data = np.asarray(data)
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(data), (n_boot, len(data)))
    boot = np.array([statistic(data[i]) for i in idx])
    half = (1.0 - conf) / 2.0
    return (float(statistic(data)), float(np.quantile(boot, half)),
            float(np.quantile(boot, 1.0 - half)))
