"""Signed split-CQR, with a minimal nonempty interval repair if necessary."""

from math import ceil

import numpy as np


def scores(y, lower, upper):
    return np.maximum(np.asarray(lower) - y, np.asarray(y) - upper)


def correction(nonconformity, nominal):
    values = np.asarray(nonconformity, dtype=float)
    n = len(values)
    k = ceil((n + 1) * nominal)
    if not (1 <= k <= n and np.isfinite(values).all()):
        raise ValueError("Calibration sample cannot support the finite-sample quantile")
    return float(np.partition(values, k - 1)[k - 1]), k


def interval(lower, upper, adjustment):
    lo = np.maximum(0, np.asarray(lower) - adjustment)
    raw_hi = np.asarray(upper) + adjustment
    # A negative signed correction can create an empty interval. Keep its raw
    # upper endpoint for audit, then conservatively replace the empty set by {lo}.
    hi = np.maximum(lo, raw_hi)
    return lo, hi, raw_hi
