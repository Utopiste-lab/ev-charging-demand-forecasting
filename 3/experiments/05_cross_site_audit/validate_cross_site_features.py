"""Validate causal target input construction and source-only data routing."""

import numpy as np
import pandas as pd

from src.feature_engineering import HOUR, feature_catalog
from test_feature_validation import (test_a_future_perturbation, test_b_feature_availability,
                                     test_d_target_leakage)
from validate_cross_site_splits import site_ledger, source_fit, target_inputs, validate_release


def assert_source_equivalent(original, check):
    assert original.site == check.site
    for partition in ("train", "validation", "final_refit"):
        one, two = getattr(original, partition), getattr(check, partition)
        pd.testing.assert_frame_equal(one.X, two.X, check_exact=True)
        pd.testing.assert_series_equal(one.y, two.y, check_exact=True)


def validate_transfer(common, direction, h, site_data, site_ledgers, settings, period, policy, partial, windows,
                      expected_test):
    source, target = direction["source"], direction["target"]
    assert source != target
    tag = f"{source}->{target}"
    fitted_rows = source_fit(source, site_data[source], site_ledgers[source])
    inputs = target_inputs(target, site_data[target], site_ledgers[target])
    assert fitted_rows.site == source and inputs.site == target
    source_values = common.loc[common.site.eq(source)].set_index("timestamp").session_start_count.sort_index()
    target_values = common.loc[common.site.eq(target)].set_index("timestamp").session_start_count.sort_index()
    test = inputs.X.index

    # Check that changing all target test labels cannot affect the source payload.
    changed = common.copy()
    change = changed.site.eq(target) & changed.timestamp.isin(test)
    assert change.sum() == len(test)
    changed.loc[change, "session_start_count"] += 10_000
    mutated_source = changed.loc[changed.site.eq(source)]
    other_source, other_source_ledger = site_ledger(mutated_source, settings, h, period, policy, partial, windows)
    assert_source_equivalent(fitted_rows, source_fit(source, other_source, other_source_ledger))
    other_target, other_target_ledger = site_ledger(changed.loc[changed.site.eq(target)],
                                                    settings, h, period, policy, partial, windows)
    changed_inputs = target_inputs(target, other_target, other_target_ledger)
    pd.testing.assert_series_equal(inputs.X.iloc[0], changed_inputs.X.iloc[0], check_exact=True)
    # Later rolling forecasts legitimately use some of those now-completed hours.
    assert not inputs.X.equals(changed_inputs.X)
    assert not hasattr(inputs, "y") and "actual" not in inputs.X and "actual" not in inputs.timeline

    # Audit every target test row against original timestamps, and perturb future data at sample origins.
    test_ledger = site_ledgers[target].loc[site_ledgers[target].partition.eq("test")]
    test_a_future_perturbation(target_values, site_data[target], test_ledger, settings, period)
    test_b_feature_availability(target_values, site_data[target], test_ledger, settings, period)
    test_d_target_leakage(target_values, site_data[target], test_ledger, settings, period)
    validate_release(h, fitted_rows, inputs, site_ledgers[source], site_ledgers[target], windows)
    assert test.equals(expected_test)
    assert test.equals(target_inputs(source, site_data[source], site_ledgers[source]).X.index)

    for split in (fitted_rows.train, fitted_rows.validation, fitted_rows.final_refit):
        assert split.X.index.equals(split.y.index)
        np.testing.assert_array_equal(split.y, source_values.loc[split.y.index])
        pd.testing.assert_frame_equal(split.X, site_data[source].X.loc[split.X.index])
    refit_keys = fitted_rows.final_refit.y.index
    different_fit_labels = int(np.count_nonzero(fitted_rows.final_refit.y.to_numpy()
                                                 != target_values.loc[refit_keys].to_numpy()))
    assert different_fit_labels > 0

    reference = test - period * HOUR
    assert inputs.timeline.seasonal_reference_hour.eq(reference).all()
    assert ((reference + HOUR) <= inputs.timeline.forecast_issued_at).all()
    np.testing.assert_array_equal(inputs.seasonal_naive, target_values.loc[reference])
    nonidentical = int(np.count_nonzero(inputs.seasonal_naive.to_numpy() != source_values.loc[reference].to_numpy()))
    assert nonidentical > 0

    definitions = [row for row in feature_catalog(settings, h, period) if row[1] in {"history", "calendar"}]
    refit_X = fitted_rows.final_refit.X
    names = [name for name, *_ in definitions]
    assert list(refit_X) == list(inputs.X) == names and len(names) == 34
    assert refit_X.dtypes.equals(inputs.X.dtypes)
    assert np.isfinite(refit_X.to_numpy()).all() and np.isfinite(inputs.X.to_numpy()).all()
    for name, kind, *_ in definitions:
        if kind == "calendar":
            np.testing.assert_array_equal(inputs.X[name], site_data[source].X.loc[test, name])
    feature_rows = [
        {"direction": tag, "horizon_hours": h, "position": pos, "feature": name, "kind": kind,
         "first_offset_from_t": start, "last_offset_from_t": end,
         "source_refit_dtype": str(refit_X[name].dtype), "target_test_dtype": str(inputs.X[name].dtype),
         "source_refit_missing": int(refit_X[name].isna().sum()), "target_test_missing": int(inputs.X[name].isna().sum()),
         "calendar_clock": "source_clock" if kind == "calendar" else "",
         "source_feature_history_site": source if kind == "history" else "",
         "target_feature_history_site": target if kind == "history" else "",
         "column_count": len(names), "result": "PASS"}
        for pos, (name, kind, start, end) in enumerate(definitions, 1)
    ]
    checks = {
        "A": f"Mutated all {len(test)} target test labels. Source fit/validation/refit payloads stayed identical.",
        "B": f"Independently reconstructed history and known calendar inputs on {len(test)} target rows. Future and current-target perturbations did not change inputs at the checked origin.",
        "C": f"Source training/validation release checked at the next partition's first issue. Both purges removed {h-1} rows.",
        "D": f"Identical {len(test)} target hours in both directions and the published same-site test calendar.",
        "E": f"Source-only fit row labels and features checked. {different_fit_labels} source labels differ from the corresponding target site's counts.",
        "F": f"Target-site count at target−24h checked for all {len(test)} hours. {nonidentical} values differ from the source-site alternative.",
        "G": "Same 34 ordered columns, dtypes, source-clock calendar and demand definitions. No nonfinite eligible values or site identifier in X.",
    }
    check_rows = [{"direction": tag, "horizon_hours": h, "test": letter, "result": "PASS", "evidence": text}
                  for letter, text in checks.items()]
    return fitted_rows, inputs, feature_rows, check_rows
