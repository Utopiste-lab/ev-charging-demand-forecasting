"""Tune and refit from a source-only supervised payload."""

from dataclasses import dataclass
import hashlib
import json

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_comparison import fit, select
from validate_cross_site_splits import SourceFit, timestamp_hash


@dataclass
class SelectedModel:
    estimator: object
    validation_prediction: object
    chosen: dict
    fit_seconds: float


def train_models(payload, settings, ridge, meta, first_issue, label_release):
    """Fit new model instances without receiving a target-site dataset."""
    assert isinstance(payload, SourceFit) and payload.site == meta["source_site"]
    train, val, final = payload.train, payload.validation, payload.final_refit
    assert final.X.index.equals(train.X.index.append(val.X.index))
    assert final.X.index.equals(final.y.index) and label_release <= first_issue
    options = [
        ("Ridge Regression", lambda **params: make_pipeline(StandardScaler(), Ridge(**params)),
         [{"alpha": alpha, "solver": ridge["solver"]} for alpha in ridge["alpha_candidates"]]),
    ]
    for key, name, constructor in [("random_forest", "Random Forest", RandomForestRegressor),
                                    ("gradient_boosting", "Gradient Boosting", GradientBoostingRegressor)]:
        config = settings[key]
        candidates = [{**config["fixed"], **p, "random_state": settings["random_state"]} for p in config["candidates"]]
        options.append((name, constructor, candidates))
    models, trials, verification = {}, [], []
    final_key = timestamp_hash(final.X.index)
    content_digest = hashlib.sha256(final.y.to_numpy(dtype="<i8").tobytes()).hexdigest()
    for name, make, candidates in options:
        chosen, validation, rows = select(make, candidates, train.X, train.y, val.X, val.y)
        estimator, seconds = fit(make, json.loads(chosen["parameters"]), final.X, final.y)
        models[name] = SelectedModel(estimator, validation, chosen, seconds)
        trials.extend({**meta, "model": name, "n_train": len(train.y), "n_validation": len(val.y), **row} for row in rows)
        verification.append({
            **meta, "model": name, "selection_site": payload.site, "final_fit_site": payload.site,
            "n_train": len(train.y), "n_validation": len(val.y), "n_final_refit": len(final.y),
            "train_timestamps_sha256": timestamp_hash(train.X.index),
            "validation_timestamps_sha256": timestamp_hash(val.X.index),
            "final_training_timestamps_sha256": final_key, "ordered_final_labels_sha256": content_digest,
            "final_training_start": final.X.index[0], "final_training_end": final.X.index[-1],
            "first_test_issue": first_issue, "last_refit_label_available_at": label_release,
            "n_test": meta["n_test"], "selected_candidate": chosen["candidate"],
            "selected_hyperparameters": chosen["parameters"],
            "selection_fit_seconds": chosen["fit_seconds"],
            "validation_prediction_seconds": chosen["validation_prediction_seconds"],
            "final_fit_seconds": seconds,
        })
    return models, trials, verification
