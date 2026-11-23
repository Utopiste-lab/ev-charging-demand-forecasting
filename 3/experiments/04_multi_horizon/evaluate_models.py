"""Validate eligible samples and the replay of the official 1h result."""

import numpy as np
import pandas as pd

from src.feature_engineering import HOUR, feature_catalog
from src.forecasting_dataset import build_dataset
from src.temporal_split import PARTITIONS, calendar_windows, split_dataset
from test_feature_validation import (test_a_future_perturbation, test_b_feature_availability,
                                     test_c_temporal_split, test_d_target_leakage)
from tune_models import timestamp_digest


def prepare_horizon(rows, baseline, horizon, policy, published_split):
    settings = baseline["features"]
    period = baseline["models"]["seasonal_naive"]["period_hours"]
    dataset = build_dataset(rows, settings, horizon, period, policy=policy)
    windows = calendar_windows(baseline["split"])
    ledger = split_dataset(dataset, windows)
    fields = [name for name, kind, _, _ in feature_catalog(settings, horizon, period)
              if kind in {"history", "calendar"}]
    assert list(dataset.X) == fields and len(fields) == settings["feature_count"]
    counts = rows.set_index("timestamp").session_start_count.sort_index()
    test_a_future_perturbation(counts, dataset, ledger, settings, period)
    test_b_feature_availability(counts, dataset, ledger, settings, period)
    test_c_temporal_split(dataset, ledger, windows, policy)
    test_d_target_leakage(counts, dataset, ledger, settings, period)
    usable = ledger.loc[ledger.status.eq("eligible")]
    assert np.isfinite(dataset.X.loc[usable.index].to_numpy()).all()
    rows_out, keys = [], {}
    for part in PARTITIONS:
        all_rows = ledger.loc[ledger.partition.eq(part)]
        used = usable.loc[usable.partition.eq(part)]
        keys[part] = index = used.index
        counts_by_reason = all_rows.status.value_counts()
        expected = published_split.loc[published_split.partition.eq(part)].iloc[0]
        assert index.equals(pd.date_range(expected.eligible_target_start, expected.eligible_target_end_inclusive, freq=HOUR))
        assert len(index) == expected.eligible_n
        rows_out.append({
            "site": rows.site.iloc[0], "horizon_hours": horizon, "partition": part,
            "calendar_start": windows[part][0], "calendar_end_exclusive": windows[part][1],
            "candidate_targets": len(all_rows), "eligible_n": len(index),
            "warmup": int(counts_by_reason.get("warmup", 0)),
            "excluded_partial_target": int(counts_by_reason.get("boundary_target", 0)),
            "excluded_partial_history": int(counts_by_reason.get("boundary_history", 0)),
            "excluded_label_release": int(counts_by_reason.get("label_release_purge", 0)),
            "target_start": index[0], "target_end_inclusive": index[-1],
            "first_issue": used.forecast_issued_at.min(), "last_issue": used.forecast_issued_at.max(),
            "last_label_available_at": used.target_end.max(), "target_timestamps_sha256": timestamp_digest(index),
        })
    return dataset, usable, keys, rows_out


def verify_refit(training, timeline, keys):
    refit = keys["train"].append(keys["validation"])
    issued = timeline.loc[keys["test"], "forecast_issued_at"].min()
    release = timeline.loc[refit, "target_end"].max()
    assert refit.is_unique and not len(refit.intersection(keys["test"]))
    assert release <= issued and len(training) == 3 and training.model.nunique() == 3
    assert training.n_final_refit.eq(len(refit)).all()
    assert training.final_training_timestamps_sha256.eq(timestamp_digest(refit)).all()
    assert training.first_test_issue.eq(issued).all() and training.last_refit_label_available_at.eq(release).all()


def one_hour_reference(root, declarations, metrics, predictions, trials, training, tolerance):
    def read(which):
        return pd.read_csv(root / declarations[which], float_precision="round_trip")

    key = ["model", "prediction_variant"]
    original = read("metrics").set_index(key).sort_index()
    actual = metrics.set_index(key).sort_index()
    baseline = read("baseline_metrics").set_index(key).sort_index()
    for frame in (original, actual, baseline):
        for col in ["first_test_issue", "last_fit_label_available_at", "validation_start",
                    "validation_end", "test_start", "test_end"]:
            frame[col] = pd.to_datetime(frame[col], format="mixed")
    stable = [column for column in original if not column.endswith("_seconds")]
    pd.testing.assert_frame_equal(actual[stable], original[stable], check_exact=False, atol=tolerance, rtol=0)
    pd.testing.assert_frame_equal(actual.loc[baseline.index, stable], baseline[stable],
                                  check_exact=False, atol=tolerance, rtol=0)
    old = read("predictions")
    pd.testing.assert_frame_equal(predictions[old.columns], old, check_exact=False, atol=tolerance, rtol=0)
    old_trials = read("trials")
    stable_trials = [col for col in old_trials if not col.endswith("_seconds")]
    pd.testing.assert_frame_equal(trials[stable_trials], old_trials[stable_trials],
                                  check_exact=False, atol=tolerance, rtol=0)
    previous_fit = read("training")
    normalized = training.copy()
    for column in ["first_test_issue", "last_refit_label_available_at",
                   "final_training_start", "final_training_end"]:
        normalized[column] = pd.to_datetime(normalized[column])
        previous_fit[column] = pd.to_datetime(previous_fit[column])
    pd.testing.assert_frame_equal(normalized[previous_fit.columns], previous_fit)
    return float(np.abs(predictions.prediction - old.prediction).max())
