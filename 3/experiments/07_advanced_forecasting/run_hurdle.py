"""Run the Hurdle stages on supervised source rows and inference-only target inputs."""

import numpy as np

from src.advanced_forecasting.hurdle_model import components, refit_hurdle, select_hurdle
from validate_cross_site_splits import timestamp_hash


def run_hurdle(payload, target, changed, available_at, options, seed):
    assert not hasattr(target, "y")
    chosen, validation, trials = select_hurdle(payload, options, seed)
    trained, magnitude_labels, seconds = refit_hurdle(payload, chosen["parameters"], seed)
    prob, magnitude = components(trained, target.X)
    perturbed_prob, perturbed_magnitude = components(trained, changed.X)
    blocked = target.timeline.forecast_issued_at.lt(available_at).to_numpy()
    assert blocked.any()
    np.testing.assert_array_equal(prob[blocked], perturbed_prob[blocked])
    np.testing.assert_array_equal(magnitude[blocked], perturbed_magnitude[blocked])
    initial_positive = payload.train.y.loc[payload.train.y.gt(0)]
    assert magnitude_labels.index.equals(payload.final_refit.y.loc[payload.final_refit.y.gt(0)].index)
    verify = {
        "occurrence_n_initial": len(payload.train.y), "occurrence_n_final": len(payload.final_refit.y),
        "occurrence_initial_hash": timestamp_hash(payload.train.y.index),
        "occurrence_final_hash": timestamp_hash(payload.final_refit.y.index),
        "magnitude_n_initial": len(initial_positive), "magnitude_n_final": len(magnitude_labels),
        "magnitude_initial_hash": timestamp_hash(initial_positive.index),
        "magnitude_final_hash": timestamp_hash(magnitude_labels.index),
    }
    return chosen, trials, verification_rows(verify, seconds, blocked), {
        "validation": {"probability": validation[0], "conditional_magnitude_raw": validation[1]},
        "test": {"probability": prob, "conditional_magnitude_raw": magnitude},
    }


def verification_rows(verify, seconds, blocked):
    return {**verify, "final_fit_seconds": seconds, "predictions_checked_before_release": int(blocked.sum()),
            "future_perturbation": "PASS"}
