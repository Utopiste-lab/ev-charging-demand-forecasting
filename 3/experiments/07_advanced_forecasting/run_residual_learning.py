"""Fit signed seasonal corrections with source-only released labels."""

import hashlib

import numpy as np

from src.advanced_forecasting.residual_model import refit_residual, select_residual
from validate_cross_site_splits import timestamp_hash


def run_residual(payload, seasonal_source, target, changed, available_at, options, seed):
    assert not hasattr(target, "y")
    chosen, validation, trials = select_residual(payload, seasonal_source, options, seed)
    trained, residuals, elapsed = refit_residual(payload, seasonal_source, chosen["parameters"], seed)
    seasonal = target.seasonal_naive.to_numpy()
    correction = trained.predict(target.X)
    blocked = target.timeline.forecast_issued_at.lt(available_at).to_numpy()
    assert blocked.any()
    np.testing.assert_array_equal(
        (seasonal + correction)[blocked],
        (changed.seasonal_naive.to_numpy() + trained.predict(changed.X))[blocked],
    )
    evidence = {
        "residual_n_initial": len(payload.train.y), "residual_n_final": len(residuals),
        "residual_initial_hash": timestamp_hash(payload.train.y.index),
        "residual_final_hash": timestamp_hash(residuals.index),
        "ordered_residual_values_sha256": hashlib.sha256(residuals.to_numpy(dtype="<f8").tobytes()).hexdigest(),
        "negative_residual_targets": int(residuals.lt(0).sum()),
        "final_fit_seconds": elapsed, "predictions_checked_before_release": int(blocked.sum()),
        "future_perturbation": "PASS",
    }
    return chosen, trials, evidence, {
        "validation": {"seasonal_naive": seasonal_source.loc[payload.validation.y.index].to_numpy(),
                       "predicted_residual": validation},
        "test": {"seasonal_naive": seasonal, "predicted_residual": correction},
    }
