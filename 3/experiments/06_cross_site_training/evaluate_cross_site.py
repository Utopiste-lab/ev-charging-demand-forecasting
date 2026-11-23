"""Forecast on the inference-only payload and score using the saved predictions."""

import json

import numpy as np
import pandas as pd

from run_comparison import timed
from src.evaluation import score


BASELINE = "Seasonal Naive"


def forecasts(models, observations, changed, first_label):
    """Freeze forecasts before test truths are joined. Test actual forecast causality."""
    assert observations.site == changed.site and observations.X.index.equals(changed.X.index)
    unavailable = observations.timeline.forecast_issued_at.lt(first_label)
    available = ~unavailable
    assert unavailable.any() and available.any()
    pd.testing.assert_frame_equal(observations.X.loc[unavailable], changed.X.loc[unavailable], check_exact=True)
    assert not observations.X.loc[available].equals(changed.X.loc[available])
    np.testing.assert_array_equal(observations.seasonal_naive.loc[unavailable], changed.seasonal_naive.loc[unavailable])
    seasonal, seconds = timed(lambda: observations.seasonal_naive.to_numpy(copy=True))
    output = {BASELINE: (seasonal, seconds)}
    checks = []
    for name, trained in models.items():
        pred, elapsed = timed(trained.estimator.predict, observations.X)
        altered = trained.estimator.predict(changed.X)
        np.testing.assert_array_equal(pred[unavailable.to_numpy()], altered[unavailable.to_numpy()])
        checks.append({"model": name, "perturbed_label_available_at": first_label,
                       "predictions_checked_before_release": int(unavailable.sum()),
                       "changed_after_release": int(np.count_nonzero(pred[available.to_numpy()] != altered[available.to_numpy()])),
                       "causality": "PASS"})
        output[name] = (pred, elapsed)
    return output, checks


def make_prediction_records(meta, plan, datasets, ledgers, validation_site, period):
    target, test_values, models, fit_rows = plan
    verified = {r["model"]: r for r in fit_rows}
    source = meta["source_site"]
    val_keys = validation_site.validation.y.index
    assert source == validation_site.site
    val_time = ledgers[source].loc[val_keys]
    test_time = target.timeline
    assert not hasattr(target, "y")
    val_truth = validation_site.validation.y
    test_truth = datasets[target.site].y.loc[target.X.index]
    seasonal_val = datasets[source].seasonal_naive.loc[val_keys].to_numpy()
    frames, computation = [], []
    for name, (test, seconds) in test_values.items():
        reference = name == BASELINE
        validation = seasonal_val if reference else models[name].validation_prediction
        parameters = json.dumps({"period_hours": period}) if reference else verified[name]["selected_hyperparameters"]
        computation.append({
            **meta, "model": name, "selected_hyperparameters": parameters,
            "selected_candidate": None if reference else verified[name]["selected_candidate"],
            "n_train": len(validation_site.train.y), "n_validation": len(val_keys),
            "n_final_refit": 0 if reference else verified[name]["n_final_refit"],
            "n_test": len(test_truth), "test_prediction_seconds": seconds,
            "selection_fit_seconds": 0.0 if reference else verified[name]["selection_fit_seconds"],
            "validation_prediction_seconds": 0.0 if reference else verified[name]["validation_prediction_seconds"],
            "final_fit_seconds": 0.0 if reference else verified[name]["final_fit_seconds"],
        })
        for role, values, truth, timeline, site in [
            ("validation", validation, val_truth, val_time, source),
            ("test", test, test_truth, test_time, target.site),
        ]:
            assert truth.index.equals(timeline.index) and np.isfinite(values).all()
            frames.append(pd.DataFrame({
                **{k: v for k, v in meta.items() if k != "n_test"},
                "model": name, "partition": role, "observation_site": site,
                "target_hour": truth.index, "forecast_issued_at": timeline.forecast_issued_at.to_numpy(),
                "target_end": timeline.target_end.to_numpy(), "actual": truth.to_numpy(),
                "prediction_raw": values, "prediction_nonnegative": np.nan if reference else np.maximum(0, values),
            }))
    return frames, computation


def summarize(saved, computations):
    keys = ["scenario", "direction", "source_site", "target_site", "horizon_hours", "model"]
    stats, slices, bias = [], [], []
    for group, block in saved.groupby(keys, sort=False):
        metadata = dict(zip(keys, group))
        test = block.loc[block.partition.eq("test")]
        val = block.loc[block.partition.eq("validation")]
        assert not test.empty and not val.empty
        assert test.observation_site.eq(metadata["target_site"]).all() and val.observation_site.eq(metadata["source_site"]).all()
        assert test.target_hour.is_unique and val.target_hour.is_unique
        for variant in ("raw", "nonnegative"):
            if variant == "nonnegative" and metadata["model"] == BASELINE:
                continue
            col = f"prediction_{variant}"
            errors = test[col].to_numpy() - test.actual.to_numpy()
            actual_mean, predicted_mean = test.actual.mean(), test[col].mean()
            common = {**metadata, "variant": variant}
            signed = {
                "mean_actual": actual_mean, "mean_prediction": predicted_mean, "mean_prediction_error": errors.mean(),
                "negative_predictions": int(test[col].lt(0).sum()),
                "negative_raw_predictions": int(test.prediction_raw.lt(0).sum()),
            }
            stats.append({
                **common, **{f"validation_{k}": v for k, v in score(val.actual, val[col]).items()},
                **{f"test_{k}": v for k, v in score(test.actual, test[col]).items()},
                "test_start": test.target_hour.iloc[0], "test_end": test.target_hour.iloc[-1],
                "first_test_issue": test.forecast_issued_at.iloc[0], **signed,
            })
            bias.append({**common, **signed, "n_test": len(test)})
            for subset, mask in (("zero", test.actual.eq(0)), ("positive", test.actual.gt(0))):
                part = test.loc[mask]
                measures = score(part.actual, part[col])
                slices.append({**common, "subset": subset, "n": len(part), "n_test": len(test),
                               "percentage": len(part) / len(test) * 100,
                               "MAE": measures["MAE"], "RMSE": measures["RMSE"]})
    return (pd.DataFrame(stats).merge(pd.DataFrame(computations), on=keys, validate="many_to_one"),
            pd.DataFrame(slices), pd.DataFrame(bias))
