"""Simplex combinations and prediction-time gating feature policies."""

import numpy as np
import pandas as pd

from src.adaptive_hybrid.experts import CORE, PROBABILITY

ADAPTIVE = "Adaptive full"
RULE = "Rule gate"
STATIC = "Static optimized"
EQUAL = "Static equal"
ABLATIONS = {
    ADAPTIVE: (True, True, True, True),
    "Adaptive without Hurdle": (False, True, True, True),
    "Adaptive without Residual": (True, False, True, True),
    "Adaptive without history": (True, True, False, True),
    "Adaptive without calendar": (True, True, True, False),
}


def inputs(predictions, features, variant):
    hurdle, residual, history, calendar = ABLATIONS[variant]
    names = [CORE[0]] + [name for keep, name in zip((hurdle, residual), CORE[1:]) if keep]
    include = [c for c in features if (history if c.startswith(("lag_", "rolling_")) else calendar)]
    assert features.index.equals(predictions.index)
    meta = pd.concat([predictions[names + ([PROBABILITY] if hurdle else [])], features[include]], axis=1)
    return meta, predictions[names].to_numpy(), names


def combine(x, weights):
    weights = np.broadcast_to(weights, x.shape)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0)
    return (x * weights).sum(axis=1)


def full_weights(weights, names, n):
    out = np.zeros((n, len(CORE)))
    out[:, [CORE.index(name) for name in names]] = weights
    return out


def simplex_grid(step):
    return np.asarray([(a, b, step - a - b) for a in range(step + 1) for b in range(step + 1 - a)]) / step


def fit_static(x, y, grid, regularization):
    prediction = grid @ x.T
    risk = np.abs(prediction - y).mean(axis=1) + regularization * ((grid - 1 / grid.shape[1])**2).sum(axis=1)
    return grid[np.argmin(risk)]
