"""Predict using frozen models and learn the CQR adjustment from calibration labels."""

import numpy as np
import pandas as pd

from src.probabilistic_forecasting.conformal import correction, interval, scores
from src.probabilistic_forecasting.quantile_models import QUANTILES, quantile_rearrangement


def predict(models, x, seasonal):
    matrix = np.column_stack([models[key].predict(x) for key in QUANTILES])
    raw = pd.DataFrame(matrix, columns=[f"{key}_raw" for key in QUANTILES], index=x.index)
    cols = list(QUANTILES)
    raw[cols] = quantile_rearrangement(matrix)
    raw["crossing_raw"] = np.any(np.diff(matrix, axis=1) < 0, axis=1)
    raw["crossing_after"] = np.any(np.diff(raw[cols].to_numpy(), axis=1) < 0, axis=1)
    raw["negative_raw"] = (matrix < 0).any(axis=1)
    raw["Point GB pre-calibration raw"] = models["Point GB pre-calibration"].predict(x)
    raw["Point GB pre-calibration"] = raw["Point GB pre-calibration raw"].clip(lower=0)
    raw["Seasonal Naive"] = seasonal.to_numpy()
    raw["Constant Zero"] = 0
    return raw


def calibrate(p, labels, nominal):
    assert p.index.equals(labels.index)
    raw_scores = scores(labels.to_numpy(), p.P10.to_numpy(), p.P90.to_numpy())
    adj, rank = correction(raw_scores, nominal)
    return raw_scores, adj, rank


def apply(p, adj):
    frame = p.copy()
    frame["CQR_lower"], frame["CQR_upper"], frame["CQR_upper_before_repair"] = interval(frame.P10, frame.P90, adj)
    frame["CQR_empty_repair"] = frame.CQR_upper_before_repair.lt(frame.CQR_lower)
    return frame
