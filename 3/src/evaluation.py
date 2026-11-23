"""Metrics shared by official forecasts and historical audits."""

import numpy as np


def score(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    residual = np.asarray(prediction, dtype=float) - actual
    if actual.shape != residual.shape or actual.size == 0:
        raise ValueError("Scoring requires aligned, nonempty predictions and observations")
    squared = np.square(residual)
    sst = np.square(actual - actual.mean()).sum()
    r2 = 1 - squared.sum() / sst if actual.size >= 2 and sst > 0 else "NA"
    return {"MAE": np.abs(residual).mean(), "RMSE": np.sqrt(squared.mean()), "R2": r2}
