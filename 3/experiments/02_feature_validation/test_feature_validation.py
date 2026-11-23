"""Automated leakage checks and one historical baseline replay."""

import importlib.util
import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import HOUR, build_features, feature_catalog
from src.evaluation import score


def anchors(ledger):
    selected = ledger.loc[ledger.status.eq("eligible")]
    return selected.groupby("partition", sort=False).head(1)


def test_a_future_perturbation(counts, dataset, ledger, features, period):
    for target, row in anchors(ledger).iterrows():
        changed = counts.copy()
        changed.loc[changed.index > row.forecast_origin] += 10_000
        X, naive = build_features(changed, features, dataset.horizon, period)
        pd.testing.assert_series_equal(X.loc[target], dataset.X.loc[target])
        assert naive.loc[target] == dataset.seasonal_naive.loc[target]


def test_b_feature_availability(counts, dataset, ledger, features, period):
    valid = ledger.loc[ledger.status.eq("eligible")]
    assert len(valid) and valid.forecast_origin.eq(valid.forecast_issued_at - HOUR).all()
    assert ((valid.index - pd.DatetimeIndex(valid.forecast_origin)) == dataset.horizon * HOUR).all()
    assert valid.target_end.eq(valid.index + HOUR).all()
    origins = pd.DatetimeIndex(valid.forecast_origin)
    for name, kind, first, last in feature_catalog(features, dataset.horizon, period):
        if kind in {"history", "baseline"}:
            assert (origins + (last + 1) * HOUR <= valid.forecast_issued_at).all()
            assert origins.min() + first * HOUR >= counts.index.min()
            if first == last:
                expected = counts.loc[origins + first * HOUR].to_numpy()
            else:
                expected = np.mean([counts.loc[origins + offset * HOUR].to_numpy()
                                    for offset in range(first, last + 1)], axis=0)
            series = dataset.seasonal_naive if kind == "baseline" else dataset.X[name]
            np.testing.assert_allclose(series.loc[valid.index], expected, atol=1e-12, rtol=0)
        elif kind == "calendar":
            if name == "weekend":
                expected = valid.index.dayofweek.isin(features["weekend_days"])
            else:
                kind, code = name.split("_")
                expected = (valid.index.hour if kind == "hour" else valid.index.dayofweek) == int(code)
            np.testing.assert_array_equal(dataset.X.loc[valid.index, name], expected)


def test_c_temporal_split(dataset, ledger, windows, policy):
    assert ledger.index.is_unique
    for name, (start, stop) in windows.items():
        np.testing.assert_array_equal(ledger.partition.eq(name),
                                      (ledger.index >= start) & (ledger.index < stop))
    eligible = ledger.loc[ledger.status.eq("eligible")]
    assert eligible.partition.nunique() == 3
    for name, following in [("train", "validation"), ("validation", "test")]:
        current = eligible.loc[eligible.partition.eq(name)]
        later = eligible.loc[eligible.partition.eq(following)]
        assert current.target_end.max() <= later.forecast_issued_at.min()
        purge = ledger.loc[ledger.partition.eq(name) & ledger.status.eq("label_release_purge")]
        threshold = windows[following][0] - (dataset.horizon - 1) * HOUR
        assert purge.target_end.gt(threshold).all()
        assert len(purge) == dataset.horizon - 1
    if policy == "exclude_partial":
        assert not eligible.partial_target.any() and not eligible.partial_history.any()
    else:
        assert not ledger.status.isin(["boundary_target", "boundary_history"]).any()
    assert not dataset.X.loc[eligible.index].isna().any().any()


def test_d_target_leakage(counts, dataset, ledger, features, period):
    expected = [name for name, kind, _, _ in feature_catalog(features, dataset.horizon, period)
                if kind in {"calendar", "history"}]
    assert list(dataset.X) == expected
    np.testing.assert_array_equal(dataset.y, counts.loc[dataset.y.index])
    for target in anchors(ledger).index:
        changed = counts.copy()
        changed.at[target] += 10_000
        X, naive = build_features(changed, features, dataset.horizon, period)
        pd.testing.assert_series_equal(X.loc[target], dataset.X.loc[target])
        assert naive.loc[target] == dataset.seasonal_naive.at[target]
        assert changed.at[target] != dataset.y.at[target]


def test_e_historical_replay(root, reference, counts, dataset, ledger, main_ledger):
    # Stage 2 read-only audit, isolated from the official one-hour runner and its results.
    audit = reference["historical_audit"]
    archive = root.with_name("2")
    code, legacy_path = (archive / audit[key]["path"] for key in ("original_code", "predictions"))
    for path, key in [(code, "original_code"), (legacy_path, "predictions")]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit[key]["sha256"]
    spec = importlib.util.spec_from_file_location("historical_baseline", code)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    stage2_features = legacy.make_features(counts)
    pd.testing.assert_frame_equal(dataset.X, stage2_features, check_dtype=False,
                                  check_names=False, check_exact=True)
    eligible = ledger.loc[ledger.status.eq("eligible")]
    parts = {name: eligible.index[eligible.partition.eq(name)] for name in ["train", "validation", "test"]}
    for name, index in parts.items():
        s = reference["split"][name]
        assert (str(index.min()), str(index.max()), len(index)) == (s["start"], s["end"], s["n"])
    ridge = reference["models"]["ridge"]
    model = make_pipeline(StandardScaler(), Ridge(alpha=audit["ridge_alpha"], solver=ridge["solver"]))
    model.fit(dataset.X.loc[parts["train"]], dataset.y.loc[parts["train"]])
    targets = parts["test"]
    actual = dataset.y.loc[targets]
    raw = model.predict(dataset.X.loc[targets])
    naive = dataset.seasonal_naive.loc[targets].to_numpy()
    frozen = pd.read_csv(legacy_path, float_precision="round_trip")
    atol = reference["evaluation"]["reference_atol"]
    recorded = {}
    np.testing.assert_array_equal(pd.DatetimeIndex(frozen.target_hour), targets)
    np.testing.assert_array_equal(pd.DatetimeIndex(frozen.forecast_issued_at), targets)
    np.testing.assert_array_equal(pd.DatetimeIndex(frozen.forecast_origin_last_observed_hour), targets - HOUR)
    np.testing.assert_array_equal(actual, frozen.actual)
    for name, forecast, archived in [("Seasonal Naive", naive, frozen.seasonal_naive),
                                     ("Ridge Regression", raw, frozen.ridge)]:
        np.testing.assert_allclose(forecast, archived, atol=atol, rtol=0)
        recorded[name] = float(np.abs(forecast - archived.to_numpy()).max())
    accepted = targets.isin(main_ledger.index[main_ledger.status.eq("eligible")])
    metrics = {}
    for key, prediction in [("seasonal_naive", naive), ("ridge_raw", raw),
                            ("ridge_nonnegative", np.maximum(0, raw))]:
        metrics[key] = (score(actual, prediction), score(actual[accepted], prediction[accepted]))
    return recorded, metrics, int(accepted.sum())
