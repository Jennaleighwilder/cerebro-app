#!/usr/bin/env python3
"""
CHIMERA BAYES — Bayesian utilities (credible intervals, posteriors)
==================================================================
Minimal: credible intervals from quantiles, posterior updates for simple cases.
"""

import numpy as np
from typing import Optional


def credible_interval(samples: list[float], alpha: float = 0.8) -> tuple[float, float]:
    """Equal-tailed credible interval at level alpha."""
    if not samples:
        return 0.0, 1.0
    arr = np.array(samples)
    lo = np.percentile(arr, (1 - alpha) / 2 * 100)
    hi = np.percentile(arr, (1 + alpha) / 2 * 100)
    return float(lo), float(hi)


def dirichlet_posterior(counts: np.ndarray, prior_alpha: float = 1.0) -> np.ndarray:
    """Posterior mean of Dirichlet: (counts + alpha) / sum."""
    K = len(counts)
    post = counts + prior_alpha
    return post / post.sum()
