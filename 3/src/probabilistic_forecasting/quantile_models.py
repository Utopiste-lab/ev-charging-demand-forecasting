"""Direct gradient boosting for quantiles and a fitting-matched point control."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

QUANTILES = {"P10": 0.1, "P50": 0.5, "P90": 0.9}


def pinball(y, p, q):
    e = np.asarray(y, dtype=float) - np.asarray(p, dtype=float)
    return np.maximum(q * e, (q - 1) * e)


def estimator(label, options, seed):
    kw = {"loss": "quantile", "alpha": QUANTILES[label]} if label in QUANTILES else {"loss": "squared_error"}
    return GradientBoostingRegressor(**kw, **options, random_state=seed, n_iter_no_change=None)


def quantile_rearrangement(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError("P10/P50/P90 must be a finite three-column matrix")
    return np.sort(np.maximum(raw, 0), axis=1)
