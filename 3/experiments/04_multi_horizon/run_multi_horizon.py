"""Caltech direct multi-horizon forecasts under the official unified protocol."""

import argparse
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
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/02_feature_validation")]

from analyze_errors import FIGURE_NAMES, FIGURE_ROOT, create_figures, error_breakdown, research_report
from evaluate_models import one_hour_reference, prepare_horizon, verify_refit
from src.forecasting_dataset import load_hourly
from tune_models import summarize_horizon, tune_horizon


RAW = "results/metrics/multi_horizon_comparison.csv"
CLIPPED = "results/metrics/multi_horizon_clipped_comparison.csv"
ERRORS = "results/metrics/horizon_error_breakdown.csv"
FIT = "results/metrics/multi_horizon_training_verification.csv"
PREDICTION = "results/tables/multi_horizon_predictions.csv"
TRIALS = "results/tables/multi_horizon_hyperparameters.csv"
SPLIT = "results/tables/multi_horizon_split_summary.csv"
GROUPS = "results/tables/multi_horizon_zero_positive_summary.csv"
FIGURES = [f"{FIGURE_ROOT}/{name}" for name in FIGURE_NAMES]
REPORT = "reports/04_multi_horizon_analysis.md"
LOG = "logs/phase5_multi_horizon.log"
TABLES = (RAW, CLIPPED, ERRORS, FIT, PREDICTION, TRIALS, SPLIT, GROUPS)
OUTPUT = (*TABLES, *FIGURES, REPORT, LOG)


def yaml_file(relative):
    return yaml.safe_load((ROOT / relative).read_text("utf-8"))


def read(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def results_variants(table):
    keys = ["horizon_hours", "model"]
    raw = table.loc[table.prediction_variant.eq("raw")]
    partner = raw.set_index(keys)
    clipped = table.loc[table.prediction_variant.eq("nonnegative")].copy()
    for metric in ["MAE", "RMSE", "R2", "negative_predictions"]:
        original = partner.loc[pd.MultiIndex.from_frame(clipped[keys]), f"test_{metric}"].to_numpy()
        clipped[f"raw_test_{metric}"] = original
        if metric != "negative_predictions":
            clipped[f"delta_test_{metric}"] = clipped[f"test_{metric}"] - original
    return raw, clipped


def compare_rerun(staging):
    for filename in TABLES:
        original, recreated = read(ROOT / filename), read(staging / filename)
        keep = [name for name in original.columns if not name.endswith("_seconds")]
        pd.testing.assert_frame_equal(original[keep], recreated[keep], check_exact=True)
    assert all(digest(ROOT / relative) == digest(staging / relative) for relative in FIGURES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Recompute and compare tables and figures without writing")
    args = parser.parse_args()
    for filename in OUTPUT:
        path = (ROOT / filename).resolve()
        if not path.is_relative_to(ROOT) or (path.exists() and not args.verify):
            raise FileExistsError(f"Output is outside Stage 3 or already exists: {path}. Use --verify to replay.")
    config_path = "config/multi_horizon.yaml"
    config = yaml_file(config_path)
    official = yaml_file(config["official_one_hour"])
    framework = yaml_file(official["framework_config"])
    baseline = yaml_file(framework["legacy_config"])
    horizons = config["horizons_hours"]
    site, policy = config["site"], config["boundary_policy"]
    assert horizons == framework["horizons_hours"] == [1, 3, 6, 24]
    assert site == official["site"] == baseline["site"] == "Caltech"
    assert policy == official["boundary_policy"] == "exclude_partial" and official["selection_metric"] == "MAE"
    contract_path = f"config/{baseline['data_contract']}"
    inputs = [config_path, config["official_one_hour"], official["framework_config"], framework["legacy_config"],
              contract_path, framework["data_view"], framework["provenance"], official["split_summary"],
              *config["reference"].values(),
              "experiments/03_tree_models_1h/run_comparison.py", "src/feature_engineering.py",
              "src/forecasting_dataset.py", "src/temporal_split.py", "src/evaluation.py"]
    before = {key: digest(ROOT / key) for key in inputs}
    hourly, record = load_hourly(ROOT / framework["data_view"], ROOT / framework["provenance"])
    contract = yaml_file(contract_path)
    assert all(record[key] == contract[key] for key in ("target", "clock", "coverage"))
    caltech = hourly.loc[hourly.site.eq(site)]
    published = read(ROOT / official["split_summary"]).query(
        "scope == 'same_site' and direction == @site and boundary_policy == @policy")
    env = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
           "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__}
    predicted, grids, fits, rows, checks = [], [], [], [], []
    plans = {}
    for h in horizons:
        print(f"Caltech {h}h", flush=True)
        dataset, timeline, keys, summary = prepare_horizon(caltech, baseline, h, policy,
                                                           published.loc[published.horizon_hours.eq(h)])
        preds, trials, fit, trained = tune_horizon(dataset, timeline, keys, official, baseline)
        verify_refit(fit, timeline, keys)
        predicted.append(preds)
        grids.append(trials)
        fits.append(fit)
        rows.extend(summary)
        plans[h] = (keys, trained, fit)
        checks.append(f"{h}h A–D leakage tests passed; train/validation/test="
                      f"{len(keys['train'])}/{len(keys['validation'])}/{len(keys['test'])}")
    test_key = plans[1][0]["test"]
    assert all(keys["test"].equals(test_key) for keys, _, _ in plans.values()), "Test calendars differ by horizon"

    with TemporaryDirectory(prefix=".multi_horizon_", dir=ROOT) as directory:
        temp = Path(directory)
        for link in OUTPUT:
            (temp / link).parent.mkdir(parents=True, exist_ok=True)
        pd.concat(predicted, ignore_index=True).to_csv(temp / PREDICTION, index=False)
        pd.concat(grids, ignore_index=True).to_csv(temp / TRIALS, index=False)
        pd.concat(fits, ignore_index=True).to_csv(temp / FIT, index=False)
        pd.DataFrame(rows).to_csv(temp / SPLIT, index=False)
        saved = read(temp / PREDICTION)
        all_metrics, selected_predictions = [], []
        for h, (keys, trained, fit) in plans.items():
            pred = saved.loc[saved.horizon_hours.eq(h)]
            metrics = summarize_horizon(pred, trained, fit, keys, official, h)
            neg = pred.loc[pred.prediction_variant.eq("raw") & pred.prediction.lt(0), "model"].unique()
            selected_predictions.append(pred.loc[pred.prediction_variant.eq("raw") | pred.model.isin(neg)])
            all_metrics.append(metrics.loc[metrics.prediction_variant.eq("raw") | metrics.model.isin(neg)])
        pd.concat(selected_predictions, ignore_index=True).to_csv(temp / PREDICTION, index=False)
        primary, clipped = results_variants(pd.concat(all_metrics, ignore_index=True))
        primary.to_csv(temp / RAW, index=False, na_rep="NA")
        clipped.to_csv(temp / CLIPPED, index=False, na_rep="NA")
        trials, fit, splits, saved = (read(temp / file) for file in (TRIALS, FIT, SPLIT, PREDICTION))
        saved_raw, saved_clipped = read(temp / RAW), read(temp / CLIPPED)
        combined = pd.concat([saved_raw, saved_clipped], ignore_index=True, sort=False)
        max_delta = one_hour_reference(
            ROOT, config["reference"], combined.loc[combined.horizon_hours.eq(1)],
            saved.loc[saved.horizon_hours.eq(1)].reset_index(drop=True),
            trials.loc[trials.horizon_hours.eq(1)].reset_index(drop=True),
            fit.loc[fit.horizon_hours.eq(1)].reset_index(drop=True), baseline["evaluation"]["reference_atol"])
        err, groups = error_breakdown(saved)
        err.to_csv(temp / ERRORS, index=False)
        groups.to_csv(temp / GROUPS, index=False)
        errors, groups = read(temp / ERRORS), read(temp / GROUPS)
        chart_data = read(temp / PREDICTION, parse_dates=["target_hour"])
        create_figures(temp / FIGURE_ROOT, saved_raw, saved_clipped, chart_data, errors, horizons)
        (temp / REPORT).write_text(research_report(saved_raw, saved_clipped, splits, fit, trials, errors,
                                                  groups, max_delta, env), encoding="utf-8")
        assert before == {key: digest(ROOT / key) for key in inputs}
        now = datetime.now(timezone(timedelta(hours=8))).isoformat()
        log = [
            f"{now} Phase 5 Caltech unified direct multi-horizon forecasts",
            f"Runtime: {json.dumps(env, sort_keys=True)}",
            f"Protected input SHA-256 before/after identical: {json.dumps(before, sort_keys=True)}",
            *checks,
            f"Official 1h replay passed. Max row-level absolute difference: {max_delta:.17g}",
            "Four horizons share identical test keys. Each horizon's 3 learned models share its refit key hash.",
            f"Completed {len(trials)} candidate fits plus 12 final refits, {len(saved)} saved forecasts and {len(FIGURES)} figures.",
            "Caltech only. No Stage 2 or raw archive reads/writes and no transfer fit.",
        ]
        (temp / LOG).write_text("\n".join(log) + "\n", encoding="utf-8")
        if args.verify:
            compare_rerun(temp)
            print("Replayed eight tables (excluding timings) and ten figure hashes.", flush=True)
        else:
            for name in OUTPUT:
                path = ROOT / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as dest, (temp / name).open("rb") as source:
                    shutil.copyfileobj(source, dest)
    print("\n".join(log[3:]), flush=True)
    print(saved_raw[["horizon_hours", "model", "test_MAE", "test_RMSE", "test_R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
