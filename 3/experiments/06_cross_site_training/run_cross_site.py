"""Train and compare two transfers and two matched controls on the audited calendar."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), *[str(ROOT / "experiments" / exp) for exp in
                            ("03_tree_models_1h", "05_cross_site_audit", "02_feature_validation")]]

from analyze_transfer_gap import analyze, render_figures, report
from audit_common_window import independent_window
from evaluate_cross_site import forecasts, make_prediction_records, summarize
from run_same_site_reference import train_reference
from src.feature_engineering import HOUR
from src.forecasting_dataset import load_hourly
from src.temporal_split import calendar_windows
from train_transfer_models import train_models
from validate_cross_site_features import assert_source_equivalent, validate_transfer
from validate_cross_site_splits import site_ledger, source_fit, summarize_direction, target_inputs, validate_release


METRICS = "results/metrics"
TABLE = "results/tables"
FIGURE = "results/figures/cross_site"
PRIMARY = f"{METRICS}/cross_site_comparison.csv"
CLIPPED = f"{METRICS}/cross_site_clipped_comparison.csv"
CONTROL = f"{METRICS}/same_site_reference_comparison.csv"
GAP = f"{METRICS}/transfer_performance_gap.csv"
ERROR = f"{METRICS}/cross_site_error_breakdown.csv"
PRED = f"{TABLE}/cross_site_predictions.csv"
GRID = f"{TABLE}/cross_site_hyperparameters.csv"
TRAIN = f"{TABLE}/cross_site_training_verification.csv"
SPLIT = f"{TABLE}/cross_site_training_split_summary.csv"
BIAS = f"{TABLE}/cross_site_prediction_bias.csv"
GROUPS = f"{TABLE}/cross_site_zero_positive_summary.csv"
ROLLING = f"{TABLE}/cross_site_rolling_origin_checks.csv"
FIGURES = [f"{FIGURE}/{name}.png" for name in [
    "cross_site_mae_comparison", "cross_site_rmse_comparison", "cross_site_r2_comparison", "transfer_vs_same_site",
    "caltech_to_jpl", "jpl_to_caltech", "transfer_mae_gap", "transfer_rmse_gap", "actual_vs_predicted_cross_site",
    "prediction_bias", "zero_vs_positive_performance",
]]
REPORT = "reports/06_cross_site_analysis.md"
LOG = "logs/phase7_cross_site_training.log"
TABLES = [PRIMARY, CLIPPED, CONTROL, GAP, ERROR, PRED, GRID, TRAIN, SPLIT, BIAS, GROUPS, ROLLING]
OUTPUT = [*TABLES, *FIGURES, REPORT, LOG]


def yaml_file(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def read(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_audited_splits(new_rows, old_rows, source, target, h):
    source_ref = old_rows.loc[old_rows.direction.str.startswith(f"{source}->") & old_rows.horizon_hours.eq(h)]
    target_ref = old_rows.loc[old_rows.direction.str.endswith(f"->{target}") & old_rows.horizon_hours.eq(h)]
    for row in new_rows:
        prior = target_ref if row["partition"] == "test" else source_ref
        previous = prior.loc[prior.partition.eq(row["partition"])]
        assert len(previous) == 1
        previous = previous.iloc[0]
        for key in ("site", "eligible_n", "target_timestamps_sha256", "excluded_label_release",
                    "warmup", "excluded_partial_target", "excluded_partial_history"):
            assert row[key] == previous[key], (source, target, h, row["partition"], key)
        for key in ("target_start", "target_end_inclusive", "first_issue", "last_label_available_at"):
            assert row[key] == pd.Timestamp(previous[key]), (source, target, h, row["partition"], key)


def verify_replay(staged):
    for name in TABLES:
        one = read(ROOT / name)
        two = read(staged / name)
        stable = [column for column in one if not column.endswith("_seconds")]
        pd.testing.assert_frame_equal(one[stable], two[stable], check_exact=True)
    assert all(digest(ROOT / f) == digest(staged / f) for f in FIGURES)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify", action="store_true", help="Rebuild the experiments and compare saved outputs")
    args = cli.parse_args()
    for name in OUTPUT:
        dest = (ROOT / name).resolve()
        if not dest.is_relative_to(ROOT) or (dest.exists() and not args.verify):
            raise FileExistsError(f"Outside Stage 3 or already exists: {dest}. For replay use --verify.")
    config_path = "config/cross_site_training.yaml"
    cfg = yaml_file(config_path)
    audit = yaml_file(cfg["audit_config"])
    unified = yaml_file(cfg["unified_config"])
    framework = yaml_file(audit["framework"])
    multi = yaml_file(audit["multi_horizon_reference"])
    baseline = yaml_file(framework["legacy_config"])
    contract_path = f"config/{baseline['data_contract']}"
    contract = yaml_file(contract_path)
    horizons, policy = audit["horizons_hours"], audit["boundary_policy"]
    assert horizons == framework["horizons_hours"] == multi["horizons_hours"] == [1, 3, 6, 24]
    assert policy == unified["boundary_policy"] == "exclude_partial" and unified["selection_metric"] == "MAE"
    declarations = cfg["audit_outputs"]
    inputs = [config_path, cfg["audit_config"], cfg["unified_config"], audit["framework"], audit["multi_horizon_reference"],
              framework["legacy_config"], contract_path, framework["data_view"], framework["provenance"],
              *declarations.values(), "src/feature_engineering.py", "src/forecasting_dataset.py", "src/temporal_split.py",
              "src/evaluation.py", "experiments/03_tree_models_1h/run_comparison.py"]
    before = {name: digest(ROOT / name) for name in inputs}
    common_record, split_record, schema_record, distribution, checks_record = (
        read(ROOT / declarations[name]) for name in ("window", "split", "features", "distributions", "checks"))
    assert len(checks_record) == 56 and checks_record.result.eq("PASS").all()
    assert len(schema_record) == 272 and schema_record.result.eq("PASS").all()
    hourly, provenance = load_hourly(ROOT / framework["data_view"], ROOT / framework["provenance"])
    assert all(provenance[key] == contract[key] for key in ("target", "clock", "coverage"))
    overlap, start, end, partial, test_keys, window = independent_window(hourly, audit["expected_common"], audit["expected_test"])
    assert window.common_timestamps_sha256.equals(common_record.common_timestamps_sha256)
    assert window.test_timestamps_sha256.equals(common_record.test_timestamps_sha256)
    settings = baseline["features"]
    period = baseline["models"]["seasonal_naive"]["period_hours"]
    ridge = baseline["models"]["ridge"]
    windows = calendar_windows(baseline["split"])
    directions = [(item["source"], item["target"]) for item in audit["directions"]]
    references = [(site, site) for site in cfg["matched_references"]]
    assert set(directions) == {("Caltech", "JPL"), ("JPL", "Caltech")}
    assert set(references) == {("Caltech", "Caltech"), ("JPL", "JPL")}
    split_rows, grids, verified, causal, frozen = [], [], [], [], []
    for h in horizons:
        data, ledger, altered, target_naive = {}, {}, {}, {}
        for site, rows in overlap.groupby("site"):
            data[site], ledger[site] = site_ledger(rows, settings, h, period, policy, partial, windows)
            # Alter the first test observation in memory. It becomes usable at its hour end.
            changed = rows.copy()
            mask = changed.timestamp.eq(test_keys[0])
            assert mask.sum() == 1
            changed.loc[mask, "session_start_count"] += 10_000
            altered[site] = site_ledger(changed, settings, h, period, policy, partial, windows)
        for src, tgt in directions + references:
            direction = f"{src}->{tgt}"
            transfer = src != tgt
            meta = {"scenario": "transfer" if transfer else "same_site", "direction": direction, "source_site": src,
                    "target_site": tgt, "horizon_hours": h, "n_test": len(test_keys)}
            if transfer:
                payload, target, schema, tests = validate_transfer(
                    overlap, {"source": src, "target": tgt}, h, data, ledger, settings, period, policy, partial, windows, test_keys)
                assert all(row["result"] == "PASS" for row in tests)
                published = schema_record.loc[schema_record.direction.eq(direction) & schema_record.horizon_hours.eq(h)]
                assert list(published.sort_values("position").feature) == [row["feature"] for row in schema]
            else:
                payload = source_fit(src, data[src], ledger[src])
                target = target_inputs(tgt, data[tgt], ledger[tgt])
                validate_release(h, payload, target, ledger[src], ledger[tgt], windows)
                assert_source_equivalent(payload, source_fit(src, *altered[src]))
            assert target.X.index.equals(test_keys) and not hasattr(target, "y")
            records = list(summarize_direction(h, direction, payload, ledger[src], ledger[tgt], windows, start, end))
            verify_audited_splits(records, split_record, src, tgt, h)
            split_rows.extend({**row, "scenario": meta["scenario"]} for row in records)
            assert list(payload.train.X) == list(payload.validation.X) == list(payload.final_refit.X) == list(target.X)
            release = ledger[src].loc[payload.final_refit.y.index, "target_end"].max()
            first_issue = target.timeline.forecast_issued_at.min()
            trainer = train_models if transfer else train_reference
            learned, trials, verification = trainer(payload, unified, ridge, meta, first_issue, release)
            expected_fit = next(row for row in records if row["partition"] == "final_refit")
            assert len(verification) == 3 and all(
                r["final_training_timestamps_sha256"] == expected_fit["target_timestamps_sha256"]
                and r["final_fit_site"] == src and r["selection_site"] == src for r in verification)
            altered_target = target_inputs(tgt, *altered[tgt])
            forecast, proof = forecasts(learned, target, altered_target, test_keys[0] + HOUR)
            if tgt in target_naive:
                np.testing.assert_array_equal(forecast["Seasonal Naive"][0], target_naive[tgt])
                forecast["Seasonal Naive"] = (target_naive[tgt], forecast["Seasonal Naive"][1])
            else:
                target_naive[tgt] = forecast["Seasonal Naive"][0]
            grids.extend(trials)
            verified.extend(verification)
            causal.extend({**meta, **record} for record in proof)
            frozen.append((meta, (target, forecast, learned, verification), data, ledger, payload))
            params = {r["model"]: json.loads(r["selected_hyperparameters"]) for r in verification}
            print(f"{direction} {h}h complete, selection/refit site {src}: {params}", flush=True)
    assert len(frozen) == 16 and len(grids) == 384 and len(verified) == len(causal) == 48
    all_models = [model.estimator for _, plan, *_ in frozen for model in plan[2].values()]
    assert len({id(model) for model in all_models}) == 48, "Each matched control must fit a distinct instance"
    # All forecasts are fixed before attaching test truths and calculating outcomes.
    predictions, computations = [], []
    for meta, plan, data, ledger, payload in frozen:
        frame, timings = make_prediction_records(meta, plan, data, ledger, payload, period)
        predictions.extend(frame)
        computations.extend(timings)
    with TemporaryDirectory(prefix=".cross_site_fit_", dir=ROOT) as directory:
        temp = Path(directory)
        for name in OUTPUT:
            (temp / name).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(predictions, ignore_index=True).to_csv(temp / PRED, index=False)
        saved = read(temp / PRED)
        metrics, errors, bias = summarize(saved, computations)
        transferred = metrics.loc[metrics.scenario.eq("transfer")]
        raw, clipped = (transferred.loc[transferred.variant.eq(v)] for v in ("raw", "nonnegative"))
        controls = metrics.loc[metrics.scenario.eq("same_site")]
        gap, groups = analyze(transferred, controls, errors)
        assert len(raw) == 32 and len(clipped) == 24 and len(controls) == 56 and len(gap) == 56
        written = {PRIMARY: raw, CLIPPED: clipped, CONTROL: controls, GAP: gap, ERROR: errors,
                   GRID: pd.DataFrame(grids), TRAIN: pd.DataFrame(verified), SPLIT: pd.DataFrame(split_rows),
                   BIAS: bias, GROUPS: groups, ROLLING: pd.DataFrame(causal)}
        for name, frame in written.items():
            frame.to_csv(temp / name, index=False, na_rep="NA")
        primary, clip, same, paired, err, signed, zero_pos, fits = (
            read(temp / name) for name in (PRIMARY, CLIPPED, CONTROL, GAP, ERROR, BIAS, GROUPS, TRAIN))
        test = read(temp / PRED, parse_dates=["target_hour"]).query("partition == 'test'")
        render_figures(temp / FIGURE, primary, same, paired, err, signed, test)
        environment = {"python": platform.python_version(), "pandas": pd.__version__,
                       "numpy": np.__version__, "sklearn": sklearn.__version__}
        (temp / REPORT).write_text(report(primary, clip, same, paired, err, signed, zero_pos, fits,
                                          distribution, environment), encoding="utf-8")
        assert {name: digest(ROOT / name) for name in inputs} == before
        now = datetime.now(timezone(timedelta(hours=8))).isoformat()
        (temp / LOG).write_text("\n".join([
            f"{now} Phase 7 cross-site transfers and matched same-site controls.",
            "Unified source-only validation ranking and source-only refits; 384 candidate fits plus 48 fresh final fits.",
            "All audited target, source fit and purge keys matched. Replayed A–G and 48 actual frozen-model perturbation checks.",
            "All forecasts were frozen before joining test truth. 3,383 common test keys per setting, target Seasonal Naive reused.",
            f"Saved {len(saved):,} prediction rows and all 32+32 primary transfer/control scores; controls include 8 reused baselines.",
            "All recorded forecasts retain original count units. Zero raw ZIP access.",
            f"Runtime: {json.dumps(environment, sort_keys=True)}",
            f"Protected Stage 3 input SHA-256 before/after: {json.dumps(before, sort_keys=True)}", "",
        ]), encoding="utf-8")
        if args.verify:
            verify_replay(temp)
            print("Reproduced all 12 CSVs apart from timings and all 11 figure hashes.", flush=True)
        else:
            for name in OUTPUT:
                dest = ROOT / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("xb") as out, (temp / name).open("rb") as stream:
                    shutil.copyfileobj(stream, out)
    print(raw[["direction", "horizon_hours", "model", "test_MAE", "test_RMSE", "test_R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
