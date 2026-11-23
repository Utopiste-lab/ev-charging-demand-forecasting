"""Stage-gated Experiment 07. Publish after all checks, or replay without writing."""

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), *[str(ROOT / "experiments" / name) for name in
                            ("02_feature_validation", "05_cross_site_audit")]]

from analyze_errors import FIGURE_NAMES, analyze, figures, report
from audit_common_window import independent_window
from run_hurdle import run_hurdle
from run_residual_learning import run_residual
from src.advanced_forecasting.evaluation import demand_metrics, occurrence_metrics
from src.feature_engineering import HOUR
from src.forecasting_dataset import load_hourly
from src.temporal_split import calendar_windows
from test_feature_validation import test_b_feature_availability, test_c_temporal_split, test_d_target_leakage
from validate_cross_site_splits import site_ledger, source_fit, target_inputs, timestamp_hash, validate_release
from verify_experiment import verify_case, verify_protection, verify_saved


TABLE = "results/tables/advanced_forecasting"
METRICS = "results/metrics/advanced_forecasting"
FIGURE = "results/figures/advanced_forecasting"
REPORT = "reports/07_advanced_forecasting.md"
LOG = "logs/phase8_advanced_forecasting.log"
PRED = f"{TABLE}/advanced_forecasting_predictions.csv"
HURDLE = f"{TABLE}/hurdle_predictions.csv"
RESIDUAL = f"{TABLE}/residual_learning_predictions.csv"
TRIAL = f"{TABLE}/advanced_forecasting_hyperparameters.csv"
SPLIT = f"{TABLE}/advanced_forecasting_split_summary.csv"
PROTECT = f"{TABLE}/protected_file_hashes.csv"
CAL = f"{TABLE}/hurdle_calibration.csv"
COMPARE = f"{METRICS}/advanced_forecasting_comparison.csv"
CLASS = f"{METRICS}/hurdle_classification_metrics.csv"
MAG = f"{METRICS}/hurdle_positive_regression_metrics.csv"
RESCOMPARE = f"{METRICS}/residual_learning_comparison.csv"
SUB = f"{METRICS}/zero_positive_performance.csv"
GAP = f"{METRICS}/cross_site_advanced_forecasting.csv"
FIT = f"{METRICS}/advanced_forecasting_training_verification.csv"
CSV = (PRED, HURDLE, RESIDUAL, TRIAL, SPLIT, PROTECT, CAL, COMPARE, CLASS, MAG, RESCOMPARE, SUB, GAP, FIT)
IMAGES = tuple(f"{FIGURE}/{name}" for name in FIGURE_NAMES)
OUTPUT = (*CSV, *IMAGES, REPORT, LOG)
CASES = (
    ("07A", "Caltech full", "Caltech", "Caltech"),
    ("07B", "JPL matched", "JPL", "JPL"),
    ("07C", "Caltech matched", "Caltech", "Caltech"),
    ("07C", "Caltech->JPL", "Caltech", "JPL"),
    ("07C", "JPL->Caltech", "JPL", "Caltech"),
)
MODELS = ("Hurdle", "Seasonal Residual")
KEY = ["case", "horizon_hours", "partition", "target_hour"]


def read(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def yaml_file(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def protection_snapshot():
    entries = []
    new_paths = {str(ROOT / name) for name in OUTPUT}
    for base in (ROOT, ROOT.with_name("2")):
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(base).as_posix()
            current_implementation = base == ROOT and (
                relative == "config/advanced_forecasting.yaml"
                or relative.startswith(("src/advanced_forecasting/", "experiments/07_advanced_forecasting/")))
            if path.is_file() and not current_implementation and str(path) not in new_paths and not any(
                    part.startswith(".advanced_forecasting_") or part == "__pycache__" for part in path.relative_to(base).parts):
                with path.open("rb") as stream:
                    entries.append({"project_relative_path": path.relative_to(ROOT.parent).as_posix(),
                                    "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
    return pd.DataFrame(entries)


def archived_refs(full, cross, case, h):
    if case == "Caltech full":
        records = full.loc[full.horizon_hours.eq(h)].copy()
        records = records.rename(columns={"prediction_variant": "variant"})
        reference_file = "results/tables/multi_horizon_predictions.csv"
    else:
        direction = {"Caltech matched": "Caltech->Caltech", "JPL matched": "JPL->JPL"}.get(case, case)
        wide = cross.loc[cross.direction.eq(direction) & cross.horizon_hours.eq(h)]
        records = pd.concat([
            wide.drop(columns="prediction_nonnegative").rename(columns={"prediction_raw": "prediction"}).assign(variant="raw"),
            wide.loc[wide.model.isin(["Ridge Regression", "Gradient Boosting"])]
                .drop(columns="prediction_raw").rename(columns={"prediction_nonnegative": "prediction"})
                .assign(variant="nonnegative"),
        ], ignore_index=True)
        reference_file = "results/tables/cross_site_predictions.csv"
    return records.assign(case=case, reference_file=reference_file)


def block_data(rows, settings, h, period, windows, policy, partial):
    data, ledger = site_ledger(rows, settings, h, period, policy, partial, windows)
    counts = rows.set_index("timestamp").session_start_count.sort_index()
    test_b_feature_availability(counts, data, ledger, settings, period)
    test_c_temporal_split(data, ledger, windows, policy)
    test_d_target_leakage(counts, data, ledger, settings, period)
    return data, ledger, counts


def verify_reference_split(case, h, src, fit, target, ledger, tables):
    full_split, full_fit, shared = tables
    if case == "Caltech full":
        expected = full_split.loc[full_split.horizon_hours.eq(h)].set_index("partition")
        hash_row = full_fit.loc[full_fit.horizon_hours.eq(h)]
        assert hash_row.final_training_timestamps_sha256.nunique() == 1
        expected_final = hash_row.final_training_timestamps_sha256.iat[0]
    else:
        expected = shared.loc[shared.horizon_hours.eq(h) & shared.direction.str.startswith(f"{src}->")].set_index("partition")
        expected_final = expected.at["final_refit", "target_timestamps_sha256"]
    assert timestamp_hash(fit.final_refit.y.index) == expected_final
    out = []
    for part, index in [
        ("train", fit.train.y.index), ("validation", fit.validation.y.index),
        ("test", target.X.index), ("final_refit", fit.final_refit.y.index),
    ]:
        expected_hash = expected_final if part == "final_refit" else expected.at[part, "target_timestamps_sha256"]
        assert timestamp_hash(index) == expected_hash
        if part != "final_refit":
            assert len(index) == expected.at[part, "eligible_n"]
        timing = target.timeline if part == "test" else ledger.loc[index]
        out.append({"case": case, "horizon_hours": h, "partition": part, "site": target.site if part == "test" else src,
                    "n": len(index), "target_start": index[0], "target_end_inclusive": index[-1],
                    "first_issue": timing.forecast_issued_at.min(), "last_label_available_at": timing.target_end.max(),
                    "timestamp_sha256": expected_hash})
    return out


def compact_components(meta, partition, truth, timeline, outputs, common):
    p, m = outputs[0]["probability"], outputs[0]["conditional_magnitude_raw"]
    sn, correction = outputs[1]["seasonal_naive"], outputs[1]["predicted_residual"]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    frames = []
    for name, values in zip(MODELS, (p * m, sn + correction)):
        assert np.isfinite(values).all()
        frames.append(pd.DataFrame({
            **meta, "partition": partition, "model": name, "target_hour": truth.index,
            "forecast_issued_at": timeline.forecast_issued_at.to_numpy(),
            "actual": truth.to_numpy(), "prediction_raw": values, "prediction_nonnegative": np.maximum(0, values),
        }))
    keys = {k: frames[0][k] for k in KEY}
    common[0].extend(frames)
    common[1].append(pd.DataFrame({**keys, "probability": p, "conditional_magnitude_raw": m}))
    common[2].append(pd.DataFrame({**keys, "seasonal_naive": sn, "predicted_residual": correction}))


def audit_and_train(cfg, view, overlap, partial, keys, baseline, cross_cfg, ref_tables, archive, replay):
    period = baseline["models"]["seasonal_naive"]["period_hours"]
    settings = baseline["features"]
    policy = cross_cfg["boundary_policy"]
    windows = calendar_windows(baseline["split"])
    values = {site: rows.set_index("timestamp").session_start_count.sort_index() for site, rows in view.groupby("site")}
    complete = [[], [], []]
    all_trials, all_fit, all_splits, staged_notes = [], [], [], []
    for phase in ("07A", "07B", "07C"):
        holder = [[], [], []]
        for _, case, src, tgt in [c for c in CASES if c[0] == phase]:
            for h in cross_cfg["horizons_hours"]:
                meta = {"phase": phase, "case": case, "source_site": src, "target_site": tgt, "horizon_hours": h}
                input_frame = view if case == "Caltech full" else overlap
                boundaries = None if case == "Caltech full" else partial
                source_rows, target_rows = (input_frame.loc[input_frame.site.eq(site)] for site in (src, tgt))
                data_s, ledger_s, _ = block_data(source_rows, settings, h, period, windows, policy, boundaries)
                if src == tgt:
                    data_t, ledger_t = data_s, ledger_s
                else:
                    data_t, ledger_t, _ = block_data(target_rows, settings, h, period, windows, policy, boundaries)
                if replay:
                    # Poison evaluation labels without rewriting causal history. Fitting,
                    # selection, prediction and all authoritative results must agree.
                    poisoned = data_s.y.copy()
                    poisoned.loc[keys] = -90_001
                    data_s = replace(data_s, y=poisoned)
                    poisoned = data_t.y.copy()
                    poisoned.loc[keys] = -90_002
                    data_t = replace(data_t, y=poisoned)
                payload = source_fit(src, data_s, ledger_s)
                target = target_inputs(tgt, data_t, ledger_t)
                assert not hasattr(target, "y") and target.X.index.equals(keys)
                validate_release(h, payload, target, ledger_s, ledger_t, windows)
                assert list(payload.final_refit.X) == list(target.X) and len(target.X.columns) == 34
                all_splits.extend(verify_reference_split(case, h, src, payload, target, ledger_s, ref_tables))
                assert payload.final_refit.y.eq(values[src].loc[payload.final_refit.y.index]).all()
                changed_rows = target_rows.copy()
                changed_rows.loc[changed_rows.timestamp.eq(keys[0]), "session_start_count"] += 10_000
                changed, changed_ledger = site_ledger(changed_rows, settings, h, period, policy, boundaries, windows)
                perturbed = target_inputs(tgt, changed, changed_ledger)
                cutoff = keys[0] + HOUR
                assert not target.X.loc[target.timeline.forecast_issued_at.ge(cutoff)].equals(
                    perturbed.X.loc[perturbed.timeline.forecast_issued_at.ge(cutoff)])
                sn = data_s.seasonal_naive.loc[payload.final_refit.X.index]
                hurdle = run_hurdle(payload, target, perturbed, cutoff, cfg["hurdle"], cfg["random_state"])
                residual = run_residual(payload, sn, target, perturbed, cutoff, cfg["residual"]["candidates"], cfg["random_state"])
                # True held-out labels enter only after both forecasts are fixed.
                truth = values[tgt].loc[keys]
                ref = archived_refs(*archive, case, h)
                archived_sn = ref.loc[ref.model.eq("Seasonal Naive") & ref.partition.eq("test")]
                np.testing.assert_array_equal(pd.DatetimeIndex(archived_sn.target_hour), keys)
                np.testing.assert_array_equal(archived_sn.actual, truth)
                np.testing.assert_array_equal(archived_sn.prediction, target.seasonal_naive)
                for partition, labels, timeline in [
                    ("validation", payload.validation.y, ledger_s.loc[payload.validation.y.index]),
                    ("test", truth, target.timeline),
                ]:
                    compact_components(meta, partition, labels, timeline, (hurdle[3][partition], residual[3][partition]), holder)
                first_issue = target.timeline.forecast_issued_at.min()
                latest = ledger_s.loc[payload.final_refit.y.index, "target_end"].max()
                for name, result in zip(MODELS, (hurdle, residual)):
                    best, trials, extra, _ = result
                    all_trials.extend({**meta, "model": name, **r, "parameters": json.dumps(r["parameters"], sort_keys=True),
                                       "selected": r is best} for r in trials)
                    all_fit.append({
                        **meta, "model": name, "selection_site": src, "fit_site": src,
                        "n_initial": len(payload.train.y), "n_validation": len(payload.validation.y),
                        "n_final": len(payload.final_refit.y), "n_test": len(keys),
                        "train_sha256": timestamp_hash(payload.train.y.index),
                        "validation_sha256": timestamp_hash(payload.validation.y.index),
                        "final_sha256": timestamp_hash(payload.final_refit.y.index),
                        "test_sha256": timestamp_hash(keys),
                        "first_test_issue": first_issue, "last_fit_label_available_at": latest,
                        "chosen_candidate": best["candidate"], "chosen_parameters": json.dumps(best["parameters"], sort_keys=True),
                        "validation_MAE": best["validation_MAE"], **extra,
                    })
                verify_case(*(pd.concat(group[-n:], ignore_index=True) for group, n in zip(holder, (4, 2, 2))),
                            pd.DataFrame(all_trials[-17:]), pd.DataFrame(all_fit[-2:]), values, keys, h, cfg["probability_threshold"])
                print(f"{phase} {case} {h}h passed. Hurdle={hurdle[0]['parameters']}; residual={residual[0]['parameters']}",
                      flush=True)
        staged_notes.append(f"{phase} passed all per-case formula, regression/classification/magnitude score, "
                            "fitting-hash and availability checks before proceeding.")
        for output, chunk in zip(complete, holder):
            output.extend(chunk)
    return [pd.concat(x, ignore_index=True) for x in complete], pd.DataFrame(all_trials), pd.DataFrame(all_fit), pd.DataFrame(all_splits), staged_notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Refit with poisoned evaluation-only labels and compare all archived outputs")
    args = parser.parse_args()
    for name in OUTPUT:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or (path.exists() and not args.verify):
            raise FileExistsError(f"Output not writable or already exists: {path}. Use --verify for replay.")
    cfg = yaml_file("config/advanced_forecasting.yaml")
    training = yaml_file(cfg["cross_site_training"])
    cross_cfg = yaml_file(training["audit_config"])
    framework = yaml_file(cross_cfg["framework"])
    baseline = yaml_file(framework["legacy_config"])
    assert cfg["selection_metric"] == "raw_validation_MAE" and cfg["random_state"] == 42
    protected = read(ROOT / PROTECT) if args.verify else protection_snapshot()
    verify_protection(ROOT, protected)
    view, provenance = load_hourly(ROOT / framework["data_view"], ROOT / framework["provenance"])
    contract = yaml_file(f"config/{baseline['data_contract']}")
    assert all(provenance[name] == contract[name] for name in ("target", "coverage", "clock"))
    overlap, _, _, partial, test_keys, _ = independent_window(view, cross_cfg["expected_common"], cross_cfg["expected_test"])
    tables = [read(ROOT / path) for path in [
        cross_cfg["references"]["same_site_split"], cross_cfg["references"]["same_site_refit"],
        training["audit_outputs"]["split"]]]
    archive = [read(ROOT / name) for name in ["results/tables/multi_horizon_predictions.csv",
                                             "results/tables/cross_site_predictions.csv"]]
    outputs, trials, fit, splits, notes = audit_and_train(cfg, view, overlap, partial, test_keys, baseline,
                                                          cross_cfg, tables, archive, args.verify)
    env = {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
           "numpy": np.__version__, "matplotlib": matplotlib.__version__, "PyYAML": yaml.__version__,
           "scipy": scipy.__version__}
    with TemporaryDirectory(prefix=".advanced_forecasting_", dir=ROOT) as directory:
        temp = Path(directory)
        for name in OUTPUT:
            (temp / name).parent.mkdir(parents=True, exist_ok=True)
        for name, frame in zip((PRED, HURDLE, RESIDUAL, TRIAL, FIT, SPLIT, PROTECT),
                               (*outputs, trials, fit, splits, protected), strict=True):
            frame.to_csv(temp / name, index=False, na_rep="NA")
        saved, hurdle, residual, trials, fit, splits = [read(temp / f) for f in (PRED, HURDLE, RESIDUAL, TRIAL, FIT, SPLIT)]
        comparison, classification, positive, residual_score, subs, gaps, calibrate = analyze(
            saved, hurdle, residual, cfg, archive, archived_refs)
        for name, frame in zip((COMPARE, CLASS, MAG, RESCOMPARE, SUB, GAP, CAL),
                               (comparison, classification, positive, residual_score, subs, gaps, calibrate), strict=True):
            frame.to_csv(temp / name, index=False, na_rep="NA")
        verify_saved(ROOT, temp)
        figures(temp / FIGURE, *[read(temp / f) for f in (COMPARE, SUB, CLASS, CAL, RESCOMPARE)],
                read(temp / PRED, parse_dates=["target_hour"]), read(temp / RESIDUAL, parse_dates=["target_hour"]))
        report_tables = (COMPARE, CLASS, MAG, RESCOMPARE, SUB, GAP, FIT)
        (temp / REPORT).write_text(report(*[read(temp / f) for f in report_tables], env), encoding="utf-8")
        verify_protection(ROOT, protected)
        log = [
            datetime.now(timezone(timedelta(hours=8))).isoformat() + " Experiment 07 advanced forecasting.",
            f"Dependency versions: {json.dumps(env, sort_keys=True)}. Random state: {cfg['random_state']}",
            *notes,
            f"20 settings, {len(fit)} refitted workflows, {len(trials)} candidate validation scores; {len(saved):,} saved new forecasts.",
            "Independent saved checks passed: MAE/RMSE/R2/bias, classification/magnitude scores, component formulas, clipping and alignment.",
            f"Protected {len(protected)} previous Stage 2/3 files. No raw archive reads and no official output writes.",
        ]
        (temp / LOG).write_text("\n".join(log) + "\n", encoding="utf-8")
        if args.verify:
            for name in CSV:
                old, new = read(ROOT / name), read(temp / name)
                columns = [c for c in old if not c.endswith("_seconds")]
                pd.testing.assert_frame_equal(old[columns], new[columns], check_exact=True)
            for name in IMAGES:
                assert (ROOT / name).read_bytes() == (temp / name).read_bytes(), name
            assert (ROOT / REPORT).read_text(encoding="utf-8") == (temp / REPORT).read_text(encoding="utf-8"), REPORT
            print("Replay with tampered evaluation-only labels reproduced all tables, charts and report.", flush=True)
        else:
            for name in OUTPUT:
                dest = ROOT / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as out, (temp / name).open("rb") as source:
                    shutil.copyfileobj(source, out)
    print("\n".join(log), flush=True)


if __name__ == "__main__":
    main()
