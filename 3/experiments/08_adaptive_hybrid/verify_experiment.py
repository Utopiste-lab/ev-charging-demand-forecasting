"""Independently verify saved predictions, split manifests and protected files."""

import numpy as np
import pandas as pd

from audit_previous_results import HOUR, METRICS, ROOT, TABLE, Project, read, timestamp_hash, verify_protection
from run_rolling_validation import archived_experts, rolling_windows
from src.adaptive_hybrid.experts import CORE
from src.adaptive_hybrid.hybrid_forecaster import ABLATIONS

GROUP = ["window", "case", "horizon_hours"]
PART = [*GROUP, "partition"]
KEY = [*GROUP, "target_hour"]


def independently(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    e = p - y
    sse = e @ e
    centered = y - y.mean()
    sst = centered @ centered
    return np.array([np.abs(e).mean(), np.sqrt(sse / len(e)), 1 - sse / sst if sst else np.nan,
                     e.mean(), (p < 0).sum()])


def verify_samples(project, out, test, validation):
    splits = read(out / TABLE / "adaptive_hybrid_split_summary.csv").set_index(GROUP + ["partition"]).sort_index()
    fits = read(out / TABLE / "adaptive_hybrid_training_verification.csv").set_index(GROUP + ["component"]).sort_index()
    manifests = read(out / TABLE / "oof_prediction_manifest.csv")
    oof = read(out / TABLE / "chronological_oof_predictions.csv", parse_dates=["target_hour", "forecast_issued_at"])
    cache = {}
    all_windows = dict(rolling_windows(project))
    for key, sample in test.groupby(GROUP, sort=False):
        window, case, h = key
        source, target = sample.source_site.iat[0], sample.target_site.iat[0]
        pair = key
        v = validation.get_group(pair)
        specification = {} if window == "official" else {"windows": all_windows[window]}
        fit, data, ledger, future, truth, boundaries = project.payload(case, source, target, h, **specification)
        cache[key] = (fit, data, ledger)
        assert pd.DatetimeIndex(sample.target_hour).equals(truth.index)
        np.testing.assert_array_equal(sample.actual, truth)
        np.testing.assert_array_equal(sample.forecast_issued_at, future.timeline.forecast_issued_at)
        np.testing.assert_array_equal(sample["model:Seasonal Naive"], future.seasonal_naive)
        np.testing.assert_array_equal(v.actual, fit.validation.y)
        np.testing.assert_array_equal(v.target_hour, fit.validation.y.index)
        np.testing.assert_array_equal(v.forecast_issued_at, ledger.loc[fit.validation.y.index, "forecast_issued_at"])
        block = splits.loc[key]
        for row in boundaries:
            listed = block.loc[row["partition"]]
            assert listed["n"] == row["n"] and listed.timestamp_sha256 == row["timestamp_sha256"]
        source_idx = fit.train.y.index
        first = int(len(source_idx) * project.cfg["oof_warmup_fraction"])
        gate_keys = source_idx[first:]
        unique = fits.loc[key]
        for name, row in unique.iterrows():
            if name == "target_pretest_logit_offset":
                assert source != target and row.fit_site == target
                _, given, tledger, _, _, _ = project.payload(f"{target} matched", target, target, h)
                exact = tledger.index[tledger.partition.eq("validation") & tledger.status.eq("eligible")]
                assert exact.intersection(future.X.index).empty
                actual_released = tledger.loc[exact, "target_end"].max()
            elif name.startswith("experts_"):
                role = fit.train if name.endswith("validation") else fit.final_refit
                exact = role.y.index
                pos = role.y.loc[role.y.gt(0)].index
                assert row.positive_n == len(pos) and row.positive_sha256 == timestamp_hash(pos)
                actual_released = ledger.loc[exact, "target_end"].max()
                assert row.fit_site == source
            else:
                exact = gate_keys
                assert row.fit_site == source
                actual_released = ledger.loc[exact, "target_end"].max()
            assert timestamp_hash(exact) == row.fit_sha256 and len(exact) == row.fit_n
            assert pd.Timestamp(row.last_label_available) == actual_released <= future.timeline.forecast_issued_at.min()
            assert row.future_observation_perturbation == "passed"
        if source == target and window == "official":
            saved_oof = oof.loc[oof.case.eq(case) & oof.horizon_hours.eq(h)]
            assert pd.DatetimeIndex(saved_oof.target_hour).equals(gate_keys)
            np.testing.assert_array_equal(saved_oof.actual, fit.train.y.loc[gate_keys])
            np.testing.assert_array_equal(saved_oof.forecast_issued_at, ledger.loc[gate_keys, "forecast_issued_at"])
            np.testing.assert_array_equal(saved_oof[fit.train.X.columns], fit.train.X.loc[gate_keys])
            np.testing.assert_array_equal(saved_oof["Seasonal Naive"], data.seasonal_naive.loc[gate_keys])
            np.testing.assert_allclose(saved_oof["Hurdle"], saved_oof["Hurdle probability"] * saved_oof.conditional_magnitude)
            np.testing.assert_allclose(saved_oof["Residual"], saved_oof["Seasonal Naive"] + saved_oof.predicted_residual)
    for key, rows in manifests.groupby([*GROUP, "oof_blocks"], sort=False):
        window, case, h, block_count = key
        fit, _, ledger = cache[window, case, h]
        keys = fit.train.y.index
        warmup = int(len(keys) * project.cfg["oof_warmup_fraction"])
        initial_cut = project.cfg["oof_inner_validation_hours"]
        prefix = keys[warmup - initial_cut:warmup]
        prefix = prefix[ledger.loc[prefix, "target_end"].le(ledger.at[keys[warmup], "forecast_issued_at"])]
        prefix_train = keys[:warmup - initial_cut]
        prefix_train = prefix_train[ledger.loc[prefix_train, "target_end"].le(ledger.at[prefix[0], "forecast_issued_at"])]
        assert len(rows) == block_count
        for row, positions in zip(rows.sort_values("block").itertuples(index=False),
                                  np.array_split(np.arange(warmup, len(keys)), block_count), strict=True):
            block = keys[positions]
            issue = block[0] - (h - 1) * HOUR
            prior = keys[:positions[0]]
            precise = prior[prior + HOUR <= issue]
            assert precise.intersection(block).empty and prefix.difference(precise).empty
            assert row.fit_n == len(precise) and row.purged == len(prior) - len(precise) == h - 1
            assert row.fit_sha256 == timestamp_hash(precise)
            assert row.prediction_n == len(block) and row.prediction_sha256 == timestamp_hash(block)
            assert row.prefix_validation_sha256 == timestamp_hash(prefix)
            assert row.prefix_train_sha256 == timestamp_hash(prefix_train)
            positive = precise[fit.train.y.loc[precise].gt(0)]
            assert row.positive_fit_n == len(positive) and row.positive_fit_sha256 == timestamp_hash(positive)
            assert pd.Timestamp(row.last_fit_label_available) <= pd.Timestamp(row.first_issue) == issue
    trials = read(out / TABLE / "adaptive_hybrid_hyperparameters.csv")
    for key, group in trials.groupby([*GROUP, "family", "selection_partition"], sort=False):
        expected = group.sort_values(["validation_MAE", "candidate"]).iloc[0]
        selected = group.loc[group.selected]
        assert len(selected) == 1 and selected.candidate.iat[0] == expected.candidate
        assert group.selection_site.eq(group.source_site).all()
        if key[-1] == "validation" and key[-2] not in {"hurdle", "residual"}:
            val = validation.get_group(key[:3])
            value = independently(val.actual, val["model:" + key[-2]])[0]
            np.testing.assert_allclose(selected.validation_MAE.iat[0], value, rtol=0, atol=5e-13)
    return cache


def verify_metrics(project, out, supplement, predictions):
    metric_cols = ["MAE", "RMSE", "R2", "bias", "negative_predictions"]
    roles = predictions.groupby(PART, sort=False)
    rolling = read(supplement / TABLE / "rolling_experiment07_predictions.csv").assign(partition="test").groupby(PART)
    for name in ("adaptive_hybrid_comparison", "static_hybrid_comparison", "gating_comparison", "rolling_window_comparison",
                 "ablation_comparison", "zero_positive_performance", "gating_sensitivity", "median_hurdle_comparison",
                 "cross_site_adaptive_comparison"):
        frame = read(out / METRICS / (name + ".csv"))
        for row in frame.itertuples(index=False):
            key = (row.window, row.case, row.horizon_hours, row.partition)
            rows = roles.get_group(key)
            column = "model:" + row.model
            if column not in rows:
                rows = rolling.get_group(key)
            if hasattr(row, "regime"):
                rows = rows.loc[rows.actual.eq(0) if row.regime == "zero" else rows.actual.gt(0)]
            assert len(rows) == row.n
            given = [getattr(row, col) for col in metric_cols]
            np.testing.assert_allclose(independently(rows.actual, rows[column]), given, atol=1e-11, rtol=1e-11)
            if hasattr(row, "matched_case"):
                target = roles.get_group(("official", row.matched_case, row.horizon_hours, "test"))
                np.testing.assert_array_equal(rows.target_hour, target.target_hour)
                same = independently(target.actual, target["model:" + row.matched_model])
                np.testing.assert_allclose([row.MAE_gap, row.RMSE_gap], np.asarray(given[:2]) - same[:2], atol=1e-11)
    intervals = read(out / METRICS / "bootstrap_confidence_intervals.csv")
    previous = read(supplement / METRICS / "experiment07_bootstrap.csv")
    pd.testing.assert_frame_equal(intervals.loc[intervals.suite.eq("Experiment 07")].reset_index(drop=True), previous)
    assert (intervals.replicates >= 2000).all() and intervals.block_hours.isin([24, 168]).all()
    assert intervals.improvement_fraction.between(0, 1).all() and intervals.ci_high.ge(intervals.ci_low).all()
    old = None
    for key, block in intervals.groupby(["suite", "case", "horizon_hours", "baseline", "model"], sort=False):
        suite, case, h, base, model = key
        if suite == "Experiment 07":
            if old is None:
                old = archived_experts(project).groupby(["case", "horizon_hours"], sort=False)
            rows = old.get_group((case, h))
        else:
            rows = roles.get_group(("official", case, h, "test"))
        difference = independently(rows.actual, rows["model:" + model])[:2] - independently(rows.actual, rows["model:" + base])[:2]
        for length, v in block.groupby("block_hours"):
            assert len(v) == 2 and length in [24, 168]
            results = v.set_index("metric").loc[["MAE", "RMSE"]]
            np.testing.assert_allclose(results.difference, difference, atol=1e-11)
            error = rows[["model:" + model, "model:" + base]].to_numpy() - rows.actual.to_numpy()[:, None]
            losses = np.column_stack([abs(error), error**2])
            n, b = len(rows), results.replicates.iloc[0]
            full, tail = divmod(n, length)
            starts = np.random.default_rng(project.cfg["seed"] + length).integers(
                0, n - length + 1, size=(b, full + bool(tail)))
            total = np.lib.stride_tricks.sliding_window_view(losses, length, axis=0).sum(axis=-1)[starts[:, :full]].sum(axis=1)
            if tail:
                total += np.lib.stride_tricks.sliding_window_view(losses, tail, axis=0).sum(axis=-1)[starts[:, -1]]
            differences = np.column_stack([(total[:, 0] - total[:, 1]) / n,
                                           np.sqrt(total[:, 2] / n) - np.sqrt(total[:, 3] / n)])
            np.testing.assert_allclose(results.bootstrap_mean_difference, differences.mean(axis=0), atol=1e-11)
            np.testing.assert_allclose(results[["ci_low", "ci_high"]], np.quantile(differences, [0.025, 0.975], axis=0).T,
                                       atol=1e-11)
            # Exact ties can sit within the floating-point error from block sums.
            fraction_one = (differences < -1e-12).mean(axis=0)
            fraction_two = (differences < +1e-12).mean(axis=0)
            assert (results.improvement_fraction >= fraction_one - 1e-12).all()
            assert (results.improvement_fraction <= fraction_two + 1e-12).all()
    relative = read(out / METRICS / "relative_improvements.csv")
    for key, rows in relative.groupby(PART, sort=False):
        actual = roles.get_group(key)
        observed = {model: independently(actual.actual, actual["model:" + model])[:2]
                    for model in set(rows.model).union(rows.baseline)}
        for row in rows.itertuples(index=False):
            p, b = observed[row.model], observed[row.baseline]
            expected = np.divide(100 * (b - p), b, out=np.full(2, np.nan), where=b != 0)
            np.testing.assert_allclose([row.MAE_improvement_percent, row.RMSE_improvement_percent], expected, atol=1e-11)


def verify_saved(project, out, supplement):
    predictions = read(out / TABLE / "adaptive_hybrid_predictions.csv", parse_dates=["target_hour", "forecast_issued_at"])
    weights = read(out / TABLE / "gating_weight_predictions.csv", parse_dates=["target_hour"])
    assert not predictions.duplicated(PART + ["target_hour"]).any()
    assert not weights.duplicated(KEY + ["model"]).any()
    test = predictions.loc[predictions.partition.eq("test")]
    validation = predictions.loc[predictions.partition.eq("validation")].groupby(GROUP)
    assert len(test.groupby(GROUP)) == 44
    assert len(test.loc[test.window.eq("official")]) == 20 * 3383
    assert len(test.loc[test.window.ne("official")]) == 24 * 1008
    verify_samples(project, out, test, validation)
    for name in ("Hurdle", "Residual"):
        raw = predictions[f"model:{name}"]
        np.testing.assert_array_equal(predictions[f"model:{name} clipped"], np.maximum(raw, 0))
    probability = predictions["Hurdle probability"]
    np.testing.assert_allclose(predictions["model:Hurdle"], probability * predictions.conditional_magnitude, atol=5e-13)
    np.testing.assert_allclose(predictions["model:Residual"], predictions["model:Seasonal Naive"] + predictions.predicted_residual, atol=5e-13)
    assert probability.between(0, 1).all()
    assert predictions["model:Hurdle median"].eq(0).equals(probability.le(0.5))
    assert predictions.median_cdf_below.max() < 0.5 + 1e-12 and predictions.median_cdf_at.min() >= 0.5 - 1e-12
    fields = ["seasonal_weight", "hurdle_weight", "residual_weight"]
    assert weights[fields].ge(0).all().all() and weights[fields].le(1).all().all()
    np.testing.assert_allclose(weights[fields].sum(axis=1), 1, rtol=0, atol=1e-12)
    for label, rows in weights.groupby("model", sort=False):
        columns = ["model:" + c for c in CORE]
        forecast = rows.merge(test[KEY + [*columns, "model:" + label]], on=KEY, validate="one_to_one")
        estimated = (forecast[fields].to_numpy() * forecast[columns].to_numpy()).sum(axis=1)
        np.testing.assert_allclose(estimated, forecast["model:" + label], rtol=0, atol=1e-12)
        if label in ABLATIONS:
            for present, name in zip(ABLATIONS[label][:2], fields[1:]):
                if not present:
                    assert rows[name].eq(0).all()
    verify_metrics(project, out, supplement, predictions)
    verify_protection(read(out / TABLE / "protected_file_hashes.csv"))
    print(f"Independent readback verified {len(predictions):,} prediction rows, {len(weights):,} weight rows, "
          "44 split sets, expert/gating hashes, all score and transfer tables, and prior SHA-256 values.", flush=True)


if __name__ == "__main__":
    from argparse import ArgumentParser
    import json
    from joblib import hash as estimator_hash
    from src.adaptive_hybrid import experts

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check-expert-state", action="store_true",
                        help="Also refit the chosen experts and compare serialized state with poisoned scoring labels")
    args = parser.parse_args()
    project = Project()
    verify_saved(project, ROOT, ROOT)
    if args.check_expert_state:
        trials = read(ROOT / TABLE / "adaptive_hybrid_hyperparameters.csv")
        trials = trials.loc[trials.selected & trials.window.eq("official") & trials.selection_partition.eq("validation")]
        checked = 0
        for case, site in (("Caltech full", "Caltech"), ("JPL matched", "JPL"), ("Caltech matched", "Caltech")):
            for h in project.horizons:
                rows = trials.loc[trials.case.eq(case) & trials.horizon_hours.eq(h)].set_index("family")
                chosen = {family: json.loads(rows.at[family, "parameters"]) for family in ("hurdle", "residual")}
                clean, data, _, target, _, _ = project.payload(case, site, site, h)
                poisoned, changed, _, _, _, _ = project.payload(case, site, site, h, poison=True)
                a, _ = experts.fit(clean, data.seasonal_naive, chosen, project.cfg["seed"], project.cfg["median_positive_forest"])
                b, _ = experts.fit(poisoned, changed.seasonal_naive, chosen, project.cfg["seed"], project.cfg["median_positive_forest"])
                before = estimator_hash(a, hash_name="sha1")
                assert before == estimator_hash(b, hash_name="sha1"), (case, h)
                experts.predict(a, target.X, target.seasonal_naive)
                assert before == estimator_hash(a, hash_name="sha1"), (case, h)
                checked += 1
        print(f"Verified {checked} source-by-horizon expert-state hashes across evaluation-label perturbation and inference.")
