"""Stage-gated Experiment 08, with a clean-process poisoned-label replay."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from tempfile import TemporaryDirectory

for setting in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[setting] = "1"
sys.dont_write_bytecode = True

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from audit_previous_results import CASES, FIGURES, METRICS, ROOT, TABLE, Project, protect, publish, read, timestamp_hash, verify_protection
from generate_oof_predictions import generate
from run_rolling_validation import bootstrap_table, execute, rolling_windows
from train_adaptive_gate import frozen_sensitivity, learn, parameter_digest, predictions, train
from train_static_hybrid import mae
from src.adaptive_hybrid import experts, gating_network
from src.adaptive_hybrid.hybrid_forecaster import ABLATIONS, ADAPTIVE, combine, full_weights, inputs
from src.adaptive_hybrid.temporal_oof import rows_at
from src.adaptive_hybrid.evaluation import score_groups

CORE = experts.CORE
ADAPTED = "Adaptive target-calibrated"
EXPERT_VARIANTS = (*CORE, "Hurdle", "Residual", "Hurdle median")
KEY = ["window", "case", "horizon_hours"]
TRAINING = TABLE / "adaptive_hybrid_training_verification.csv"
OFFICIAL = TABLE / "adaptive_hybrid_predictions.csv"
REPORT = Path("reports/08_adaptive_hybrid.md")
SUPPLEMENT = Path("reports/07_robustness_supplement.md")
LOG = Path("logs/phase9_adaptive_hybrid.log")


def raw_attestation(fit, positive, partition, trial):
    return {"component": f"experts_{partition}", "fit_site": fit.site,
            "fit_n": len(trial), "fit_sha256": timestamp_hash(trial),
            "positive_n": len(positive), "positive_sha256": timestamp_hash(positive)}


def assert_expert_reference(project, key, actual, p):
    case, h, part = key
    prior = project.previous_forecasts
    prior = prior.loc[prior.case.eq(case) & prior.horizon_hours.eq(h) & prior.partition.eq(part)]
    for old, name in (("Hurdle", "Hurdle"), ("Seasonal Residual", "Residual")):
        ref = prior.loc[prior.model.eq(old)].set_index("target_hour")
        assert p.index.equals(ref.index)
        np.testing.assert_array_equal(actual, ref.actual)
        np.testing.assert_allclose(p[name], ref.prediction_raw, atol=1e-11, rtol=0)
        np.testing.assert_allclose(p[name + " clipped"], ref.prediction_nonnegative, atol=1e-11, rtol=0)


def summarize_weight(group, matrix, meta):
    from src.adaptive_hybrid.experts import PROBABILITY
    groups = {"overall": np.full(len(group), "all"),
              "observed_regime_posthoc": np.where(group.actual.eq(0), "zero", "positive"),
              "probability_regime": np.where(group[PROBABILITY].le(0.5), "p<=0.5", "p>0.5"),
              "source_clock_hour": group.index.hour,
              "source_clock_weekday": group.index.dayofweek}
    rows = []
    for group_type, levels in groups.items():
        for level in np.unique(levels):
            sample = matrix[levels == level]
            for i, expert in enumerate(CORE):
                col = sample[:, i]
                rows.append({**meta, "group_type": group_type, "group": str(level), "expert": expert, "n": len(col),
                             "mean_weight": col.mean(), "std_weight": col.std(),
                             "p10": np.quantile(col, 0.10), "p50": np.quantile(col, 0.50),
                             "p90": np.quantile(col, 0.90), "weight_over_95_fraction": (col >= .95).mean(),
                             "avg_adjacent_total_variation": np.abs(np.diff(matrix, axis=0)).sum(axis=1).mean() / 2
                             if group_type == "overall" else np.nan})
    return rows


def select_source(project, case, site, h, windows, poison):
    fit, data, ledger, target, truth, split = project.payload(case, site, site, h, windows=windows, poison=poison)
    config, advanced, seed = project.cfg, project.advanced, project.cfg["seed"]
    chosen, expert_trials = experts.select(fit, data.seasonal_naive, advanced, seed)
    preliminary, positive_train = experts.fit(replace(fit, final_refit=fit.train), data.seasonal_naive,
                                             chosen, seed, config["median_positive_forest"])
    vx = fit.validation.X
    val = experts.predict(preliminary, vx, data.seasonal_naive.loc[vx.index])
    if windows is None:
        assert_expert_reference(project, (case, h, "validation"), fit.validation.y, val)
    count = config["oof_blocks"]
    oof, ox, manifest, prefix = generate(project, fit, data, ledger, h, count)
    static, rule, gates, best_name, gate_trials = train(project, oof, ox, val, vx, fit.validation.y, ablate=windows is None)
    _, params = gates[ADAPTIVE]
    prespecified = {}
    if windows is None:
        for new_seed in config["gating_seeds"]:
            prespecified["seed", new_seed] = gates[ADAPTIVE][0] if new_seed == seed else frozen_sensitivity(
                project, oof, ox, params, new_seed)
        for alternate in config["oof_sensitivity"]:
            mx, xx, manifest_part, _ = generate(project, fit, data, ledger, h, alternate, prefix)
            prespecified["oof_blocks", alternate] = frozen_sensitivity(project, mx, xx, params, seed)
            manifest.extend(manifest_part)
        prespecified["oof_blocks", count] = gates[ADAPTIVE][0]
    final, positive = experts.fit(fit, data.seasonal_naive, chosen, seed, config["median_positive_forest"])
    validates, val_weights = predictions(val, vx, static, rule, gates, best_name)
    settings = {"fit": fit, "preliminary": preliminary, "final": final, "oof": oof, "ox": ox,
                "static": static, "rule": rule, "gates": gates, "best": best_name, "sensitive": prespecified,
                "val": val, "val_forecasts": validates, "val_weights": val_weights,
                "manifest": manifest, "ledger": ledger, "review": expert_trials + gate_trials,
                "prefix_trials": prefix[1], "selected_experts": chosen}
    settings["attestations"] = [raw_attestation(fit, positive_train, "validation", fit.train.y.index),
                                raw_attestation(fit, positive, "test", fit.final_refit.y.index)]
    return settings


def target_calibrate(project, chosen, case, target, h, target_inputs, weights, p):
    s = chosen
    data, ledger = project.dataset(case, target, h, project.windows)
    keys = ledger.index[ledger.partition.eq("validation") & ledger.status.eq("eligible")]
    assert keys.intersection(target_inputs.X.index).empty and len(keys)
    assert ledger.loc[keys, "target_end"].max() <= target_inputs.timeline.forecast_issued_at.min()
    src_train = s["fit"].train.X.index
    assert s["ledger"].loc[src_train, "target_end"].max() <= ledger.loc[keys, "forecast_issued_at"].min()
    vx = data.X.loc[keys]
    vp = experts.predict(s["preliminary"], vx, data.seasonal_naive.loc[keys])
    gate_inputs, expert_matrix, _ = inputs(vp, vx, ADAPTIVE)
    gate = s["gates"][ADAPTIVE][0]
    offset = gating_network.calibrate(gate.weights(gate_inputs), expert_matrix, data.y.loc[keys].to_numpy(),
                                      project.cfg["adaptation"]["offset_l2"], project.cfg["gate"]["smoothing"])
    return gating_network.apply_offset(weights[ADAPTIVE], offset), {
        "component": "target_pretest_logit_offset", "fit_site": target, "fit_n": len(keys),
        "fit_sha256": timestamp_hash(keys), "last_label_available": ledger.loc[keys, "target_end"].max(),
        "parameters_sha256": hashlib.sha256(np.asarray(offset, dtype="<f8").tobytes()).hexdigest(),
        "configuration": json.dumps({"offset": offset.tolist(), "l2": project.cfg["adaptation"]["offset_l2"]}),
    }


def output_rows(meta, p, truth, issue, variants):
    out = pd.DataFrame({**meta, "target_hour": p.index, "forecast_issued_at": issue, "actual": truth})
    for name in EXPERT_VARIANTS:
        out["model:" + name] = p[name]
    for name, forecast in variants.items():
        out["model:" + name] = forecast
    for field in (experts.PROBABILITY, "conditional_magnitude", "predicted_residual", "median_cdf_below", "median_cdf_at"):
        out[field] = p[field]
    return out


def run_case(project, case, source, target, h, window, windows, cached, poison):
    meta = dict(window=window, case=case, horizon_hours=h, source_site=source, target_site=target)
    source_case = case if source == target else f"{source} matched"
    key = (window, source_case, h)
    if key not in cached:
        cached[key] = select_source(project, source_case, source, h, windows, poison)
    s = cached[key]
    fit, data, ledger, target_inputs, truth, split = project.payload(case, source, target, h, windows=windows, poison=poison)
    assert fit.final_refit.y.equals(s["fit"].final_refit.y) and fit.train.y.equals(s["fit"].train.y)
    p = experts.predict(s["final"], target_inputs.X, target_inputs.seasonal_naive)
    if windows is None:
        assert_expert_reference(project, (case, h, "test"), truth, p)
    forecasts, weights = predictions(p, target_inputs.X, s["static"], s["rule"], s["gates"], s["best"])
    adapt = None
    if source != target:
        weights[ADAPTED], adapt = target_calibrate(project, s, case, target, h, target_inputs, weights, p)
        forecasts[ADAPTED] = combine(p[CORE].to_numpy(), weights[ADAPTED])
    recent, blocked = project.historical_perturbation(case, target, h, project.windows if windows is None else windows,
                                                     target_inputs)
    altered = experts.predict(s["final"], recent.X, recent.seasonal_naive)
    pd.testing.assert_frame_equal(altered.loc[blocked], p.loc[blocked], check_exact=True)
    changed, changed_weights = predictions(altered, recent.X, s["static"], s["rule"], s["gates"], s["best"])
    for name, variant in changed.items():
        np.testing.assert_allclose(forecasts[name][blocked], variant[blocked], atol=1e-12, rtol=0)
    if adapt is not None:
        off = json.loads(adapt["configuration"])["offset"]
        recalculated = combine(altered[CORE].to_numpy(), gating_network.apply_offset(changed_weights[ADAPTIVE], off))
        np.testing.assert_allclose(forecasts[ADAPTED][blocked], recalculated[blocked], atol=1e-12, rtol=0)
    test = output_rows(meta, p, truth, target_inputs.timeline.forecast_issued_at, forecasts)
    val = output_rows(meta, s["val"], fit.validation.y, ledger.loc[fit.validation.y.index, "forecast_issued_at"],
                      s["val_forecasts"])
    val["forecast_site"] = source
    val["partition"] = "validation"
    test["forecast_site"] = target
    test["partition"] = "test"
    frames = {"prediction": [test, val], "weights": [], "weight_stats": [], "manifest": [], "fit": [],
              "trials": [], "split": [dict(r, window=window) for r in split], "sensitivity": [], "shift": [],
              "oof": [], "concentration": []}
    for label, matrix, frame in [(k, v, test) for k, v in weights.items()]:
        assert np.isfinite(frame[f"model:{label}"]).all()
        row = pd.DataFrame({**meta, "model": label, "target_hour": frame.index,
                            "seasonal_weight": matrix[:, 0], "hurdle_weight": matrix[:, 1], "residual_weight": matrix[:, 2]})
        frames["weights"].append(row)
        frames["weight_stats"].extend(summarize_weight(frame, matrix, {**meta, "model": label}))
    gate_checks = {"fit_n": len(s["oof"]), "fit_sha256": timestamp_hash(s["oof"].index), "fit_site": source,
                   "last_label_available": ledger.loc[s["oof"].index, "target_end"].max()}
    first_test = target_inputs.timeline.forecast_issued_at.min()
    for r in s["attestations"]:
        keys = fit.train.y.index if r["component"].endswith("validation") else fit.final_refit.y.index
        assert r["fit_sha256"] == timestamp_hash(keys)
        frames["fit"].append(dict(r, **meta, last_label_available=ledger.loc[keys, "target_end"].max()))
    for gate, (model, parameters) in s["gates"].items():
        frames["fit"].append({**meta, **gate_checks, "component": gate, "parameters_sha256": parameter_digest(model),
                              "configuration": json.dumps(parameters, sort_keys=True)})
    for name in ("Static optimized", "Rule gate"):
        parameters = s["static"] if name == "Static optimized" else s["rule"]
        frames["fit"].append({**meta, **gate_checks, "component": name,
                              "configuration": json.dumps(parameters, sort_keys=True)})
    if adapt is not None:
        frames["fit"].append({**meta, **adapt})
    for r in frames["fit"]:
        assert r["last_label_available"] <= first_test
        r["first_test_issue"] = first_test
        r["future_observation_perturbation"] = "passed"
    for r in s["review"]:
        frames["trials"].append({**meta, **r, "selection_site": source, "selection_partition": "validation",
                                 "parameters": json.dumps(r["parameters"], sort_keys=True)})
    if source == target:
        for r in s["prefix_trials"]:
            frames["trials"].append({**meta, **r, "selection_site": source, "selection_partition": "OOF prefix",
                                     "parameters": json.dumps(r["parameters"], sort_keys=True)})
        frames["manifest"].extend(dict(r, **meta) for r in s["manifest"])
        if windows is None:
            mx = pd.concat([s["oof"], s["ox"]], axis=1).rename_axis("target_hour").reset_index()
            frames["oof"].append(mx.assign(**meta))
    for (dimension, setting), gate in s["sensitive"].items():
        meta_x, matrix, _ = inputs(p, target_inputs.X, ADAPTIVE)
        name = f"Sensitivity {dimension}={setting}"
        value = combine(matrix, gate.weights(meta_x))
        test["model:" + name] = value
        frames["sensitivity"].append({**meta, "dimension": dimension, "setting": setting, "model": name,
                                      "parameters_sha256": parameter_digest(gate)})
    for field in (*CORE, experts.PROBABILITY):
        ref = s["oof"][field]
        for part, v in (("OOF", ref), ("validation", s["val"][field]), ("test_refit", p[field])):
            scale = ref.std(ddof=0)
            frames["shift"].append({**meta, "feature": field, "partition": part,
                                    "mean": v.mean(), "std": v.std(ddof=0), "p05": v.quantile(.05), "p95": v.quantile(.95),
                                    "standardized_mean_change_from_oof": (v.mean() - ref.mean()) / scale if scale else np.nan})
    for name in (ADAPTIVE, "Static optimized", "Rule gate"):
        diff = (test[f"model:{name}"] - truth).abs() - (test["model:Seasonal Naive"] - truth).abs()
        for week, points in diff.groupby(np.arange(len(diff)) // 168):
            frames["concentration"].append({**meta, "model": name, "week": week, "start": points.index[0],
                                            "n": len(points), "loss_difference_sum": points.sum(),
                                            "win_hours": points.lt(0).sum()})
    assert np.isfinite(test["actual"]).all() and len(test) == len(truth)
    print(f"{window} {case} {h}h verified, selected expert={s['best']}; adaptive test MAE={mae(truth, forecasts[ADAPTIVE]):.6f}",
          flush=True)
    return frames


def source_trace(frames):
    # Convert once so numerical replay and the independent verifier share the published schema.
    return {field: pd.concat(values, ignore_index=True) if isinstance(values[0], pd.DataFrame) else pd.DataFrame(values)
            for field, values in frames.items() if values}


def suite(project, *, poison=False):
    frames, cached = {}, {}
    for case, source, target in CASES:
        for h in project.horizons:
            chunk = run_case(project, case, source, target, h, "official", None, cached, poison)
            for field, records in chunk.items():
                frames.setdefault(field, []).extend(records)
    # Rolling cases run only after the two same-site plus both transfer phases pass.
    for window, windows in rolling_windows(project):
        for case, source in [("Caltech full", "Caltech"), ("JPL matched", "JPL")]:
            for h in project.horizons:
                chunk = run_case(project, case, source, source, h, window, windows, cached, poison)
                for field, records in chunk.items():
                    frames.setdefault(field, []).extend(records)
    return source_trace(frames)


def compare_with_reference(metrics):
    reference = metrics.loc[metrics.model.isin(["Seasonal Naive", "Best validation expert"])]
    data = metrics.merge(reference, on=KEY + ["partition"], suffixes=("", "_reference"))
    for metric in ("MAE", "RMSE"):
        data[f"{metric}_improvement_percent"] = (100 * (data[f"{metric}_reference"] - data[metric]) /
                                                data[f"{metric}_reference"].replace(0, np.nan))
    data["baseline"] = data.model_reference
    return data[KEY + ["partition", "model", "baseline", "MAE_improvement_percent", "RMSE_improvement_percent"]]


def save(project, output, supplemental, protected, frames):
    test = frames["prediction"].loc[frames["prediction"].partition.eq("test")]
    all_scores, subsets = score_groups(frames["prediction"], KEY + ["partition"])
    official = all_scores.loc[all_scores.window.eq("official")]
    rolling = all_scores.loc[all_scores.window.ne("official") & all_scores.partition.eq("test")]
    reference_rolling = read(supplemental / METRICS / "rolling_experiment07_comparison.csv")
    test_rolling = rolling.merge(reference_rolling, on=KEY + ["model"], suffixes=("", "_supplement"))
    for metric in ("MAE", "RMSE"):
        np.testing.assert_allclose(test_rolling[metric], test_rolling[f"{metric}_supplement"], rtol=0, atol=1e-11)
    rolling = pd.concat([rolling, reference_rolling.loc[reference_rolling.model.isin(
        ["Ridge Regression", "Random Forest", "Gradient Boosting"])].assign(partition="test")], ignore_index=True)
    chief = ["Static optimized", "Rule gate", ADAPTIVE]
    intervals = bootstrap_table(test.loc[test.window.eq("official")], project, "Experiment 08", chief)
    intervals_best = bootstrap_table(test.loc[test.window.eq("official")], project, "Experiment 08", chief,
                                     baseline="Best validation expert")
    original_intervals = read(supplemental / METRICS / "experiment07_bootstrap.csv")
    intervals = pd.concat([original_intervals, intervals, intervals_best], ignore_index=True)
    cross = official.loc[official.case.str.contains("->", regex=False) & official.partition.eq("test")].copy()
    cross["matched_case"] = np.where(cross.case.eq("Caltech->JPL"), "JPL matched", "Caltech matched")
    cross["evaluation_mode"] = np.where(cross.model.eq(ADAPTED), "target pretest calibration", "source-only transfer")
    cross["matched_model"] = cross.model.replace({ADAPTED: ADAPTIVE})
    refs = official.loc[official.partition.eq("test"), ["case", "horizon_hours", "model", "MAE", "RMSE"]]
    cross = cross.merge(refs, left_on=["matched_case", "horizon_hours", "matched_model"],
                        right_on=["case", "horizon_hours", "model"], suffixes=("", "_matched"), validate="many_to_one")
    for metric in ("MAE", "RMSE"):
        cross[f"{metric}_gap"] = cross[metric] - cross[f"{metric}_matched"]
        cross[f"{metric}_gap_percent"] = 100 * cross[f"{metric}_gap"] / cross[f"{metric}_matched"].replace(0, np.nan)
    comparison = official.loc[~official.model.str.startswith("Sensitivity")]
    ablated = [*CORE, "Static equal", "Static optimized", *ABLATIONS]
    result = {
        OFFICIAL: frames["prediction"],
        TRAINING: frames["fit"],
        TABLE / "adaptive_hybrid_split_summary.csv": frames["split"],
        TABLE / "adaptive_hybrid_hyperparameters.csv": frames["trials"].drop(columns="candidate_fit_seconds"),
        TABLE / "oof_prediction_manifest.csv": frames["manifest"],
        TABLE / "chronological_oof_predictions.csv": frames["oof"],
        TABLE / "gating_weight_predictions.csv": frames["weights"],
        TABLE / "sensitivity_configurations.csv": frames["sensitivity"],
        TABLE / "expert_distribution_shift.csv": frames["shift"],
        TABLE / "weekly_error_contributions.csv": frames["concentration"],
        TABLE / "protected_file_hashes.csv": protected,
        METRICS / "adaptive_hybrid_comparison.csv": comparison,
        METRICS / "static_hybrid_comparison.csv": comparison.loc[comparison.model.isin(["Static equal", "Static optimized", *CORE])],
        METRICS / "gating_comparison.csv": comparison.loc[comparison.model.isin([*chief, ADAPTED, "Best validation expert"])],
        METRICS / "rolling_window_comparison.csv": rolling,
        METRICS / "bootstrap_confidence_intervals.csv": intervals,
        METRICS / "expert_weight_summary.csv": frames["weight_stats"],
        METRICS / "zero_positive_performance.csv": subsets,
        METRICS / "cross_site_adaptive_comparison.csv": cross,
        METRICS / "ablation_comparison.csv": comparison.loc[comparison.model.isin(ablated)],
        METRICS / "median_hurdle_comparison.csv": all_scores.loc[all_scores.model.isin(["Seasonal Naive", "Hurdle clipped", "Hurdle median"])],
        METRICS / "relative_improvements.csv": compare_with_reference(all_scores),
        METRICS / "gating_sensitivity.csv": official.loc[official.model.str.startswith("Sensitivity")],
    }
    for key, frame in result.items():
        publish(frame, output / key)
    return list(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Replay with poisoned scoring-only labels, without publication")
    options = parser.parse_args()
    from analyze_errors import artifacts
    from verify_experiment import verify_saved
    if not options.verify and (ROOT / OFFICIAL).exists():
        raise FileExistsError("Experiment 08 outputs exist. Use --verify.")
    start = datetime.now(timezone.utc)
    project = Project()
    protected = read(ROOT / TABLE / "protected_file_hashes.csv") if options.verify else protect()
    verify_protection(protected)
    with TemporaryDirectory(prefix=".exp08-", dir=ROOT) as path:
        output = Path(path)
        supplemental = output if options.verify else ROOT
        if options.verify or not (ROOT / METRICS / "experiment07_bootstrap.csv").exists():
            execute(project, supplemental, poison=options.verify)
        frames = suite(project, poison=options.verify)
        products = save(project, output, supplemental, protected, frames)
        del frames
        verify_saved(project, output, supplemental)
        versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                    "PyYAML": yaml.__version__}
        products.extend(artifacts(output, supplemental, project, versions))
        verify_protection(protected)
        if options.verify:
            supplements = [p.relative_to(output) for folder in (TABLE, METRICS) for p in (output / folder).glob("*experiment07*")]
            for rel in [*products, *supplements]:
                if rel.suffix == ".csv":
                    pd.testing.assert_frame_equal(read(output / rel), read(ROOT / rel), check_exact=False, atol=1e-11, rtol=1e-11)
                elif rel.suffix == ".md":
                    assert (output / rel).read_text("utf-8") == (ROOT / rel).read_text("utf-8"), rel
                else:
                    assert (output / rel).read_bytes() == (ROOT / rel).read_bytes(), rel
            print("Clean replay with poisoned evaluation labels agrees on all tables, plots and reports.", flush=True)
        else:
            for rel in products:
                assert not (ROOT / rel).exists() and (ROOT / rel).resolve().is_relative_to(ROOT)
            for rel in products:
                destination = ROOT / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                with (output / rel).open("rb") as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            (ROOT / LOG).write_text(f"Experiment 08 started {start.isoformat()}\nCompleted {datetime.now(timezone.utc).isoformat()}\n"
                                    f"Versions: {json.dumps(versions, sort_keys=True)}\nSeeds: {project.cfg['gating_seeds']}\n"
                                    f"Verified {len(products)} new artifacts and {len(protected)} prior Stage 2/3 SHA-256 values.\n"
                                    "Supplement: three equal-length windows at both sites and 2500 paired block replicates.\n"
                                    "No raw archive reads.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
