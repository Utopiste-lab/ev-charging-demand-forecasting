"""Run the sole official Caltech one-hour model comparison."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from presentation import FIGURES, create_figures, render_reports
from src.evaluation import score
from src.feature_engineering import feature_catalog
from src.forecasting_dataset import build_dataset, load_hourly
from src.temporal_split import PARTITIONS, calendar_windows, split_dataset


METRICS = "results/metrics/tree_models_1h_comparison.csv"
PREDICTIONS = "results/tables/tree_models_1h_predictions.csv"
TRIALS = "results/tables/tree_models_1h_hyperparameters.csv"
SAMPLES = "results/tables/training_sample_verification_1h.csv"
BASELINE_METRICS = "results/metrics/baseline_1h_unified.csv"
BASELINE_PREDICTIONS = "results/tables/baseline_predictions.csv"
REPORT = "reports/03_tree_models_1h.md"
BASELINE_REPORT = "reports/01_data_contract_and_baseline.md"
LOG = "logs/phase4_tree_models.log"
TABLES = (METRICS, PREDICTIONS, TRIALS, SAMPLES, BASELINE_METRICS, BASELINE_PREDICTIONS)
DELIVERABLES = (*TABLES, *FIGURES, REPORT, BASELINE_REPORT, LOG)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def timed(operation, *args):
    started = perf_counter()
    result = operation(*args)
    return result, perf_counter() - started


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp_digest(index):
    return hashlib.sha256("".join(f"{key.isoformat()}\n" for key in index).encode("ascii")).hexdigest()


def fit(factory, parameters, X, y):
    model, elapsed = timed(factory(**parameters).fit, X, y)
    if isinstance(model, RandomForestRegressor):
        # Stable tree summation order for predictions while keeping parallel fitting.
        model.set_params(n_jobs=1)
    return model, elapsed


def select(factory, candidates, X_train, y_train, X_val, y_val):
    """Candidates can see only training and chronological validation arrays."""
    trials = []
    val_predictions = []
    for moment, params in enumerate(candidates, 1):
        model, fit_time = fit(factory, params, X_train, y_train)
        pred, prediction_time = timed(model.predict, X_val)
        val_predictions.append(pred)
        trials.append({
            "candidate": moment, "parameters": json.dumps(params, sort_keys=True),
            "fit_seconds": fit_time, "validation_prediction_seconds": prediction_time,
            **{f"train_{k}": v for k, v in score(y_train, model.predict(X_train)).items()},
            **{f"validation_{k}": v for k, v in score(y_val, pred).items()},
            "validation_negative_predictions": int(np.count_nonzero(pred < 0)),
        })
    chosen = min(trials, key=lambda trial: (trial["validation_MAE"], trial["candidate"]))
    for trial in trials:
        trial["selected"] = trial is chosen
    return chosen, val_predictions[chosen["candidate"] - 1], trials


def planned_dataset(caltech, baseline, cfg, phase3):
    features = baseline["features"]
    period = baseline["models"]["seasonal_naive"]["period_hours"]
    dataset = build_dataset(caltech, features, cfg["horizon_hours"], period, policy=cfg["boundary_policy"])
    timeline = split_dataset(dataset, calendar_windows(baseline["split"]))
    eligible = timeline.loc[timeline.status.eq("eligible")]
    keys = {part: eligible.index[eligible.partition.eq(part)] for part in PARTITIONS}
    check(len(keys["test"]) == cfg["expected_test_samples"] == 3383, "Expected 3,383 official test targets")
    for part, index in keys.items():
        recorded = phase3.loc[phase3.partition.eq(part)].iloc[0]
        expected = pd.date_range(recorded.eligible_target_start, recorded.eligible_target_end_inclusive, freq="h")
        check(index.equals(expected) and len(index) == recorded.eligible_n, f"{part} differs from the Phase 3 split")
    fields = [name for name, kind, _, _ in feature_catalog(features, dataset.horizon, period)
              if kind in {"history", "calendar"}]
    check(list(dataset.X) == fields and len(fields) == 34, "Feature schema changed")
    check(not eligible.partial_target.any() and not eligible.partial_history.any(), "Partial hour selected")
    check(np.isfinite(dataset.X.loc[eligible.index].to_numpy()).all(), "Nonfinite eligible input")
    check(eligible.forecast_issued_at.eq(eligible.index).all(), "Issue/target mismatch")
    for earlier, later in zip(PARTITIONS, PARTITIONS[1:]):
        check(eligible.loc[keys[earlier], "target_end"].max() <= eligible.loc[keys[later], "forecast_issued_at"].min(),
              f"{earlier} labels are not available for the first {later} forecast")
    return dataset, eligible, keys


def run_models(dataset, timeline, keys, cfg, ridge, period):
    train, val, test = (keys[part] for part in PARTITIONS)
    X, y = dataset.X, dataset.y
    refit_index = train.append(val)
    X_refit, y_refit = X.loc[refit_index], y.loc[refit_index]
    first_issue = timeline.at[test[0], "forecast_issued_at"]
    last_label = timeline.loc[refit_index, "target_end"].max()
    check(refit_index.is_unique and not len(refit_index.intersection(test))
          and X_refit.index.equals(y_refit.index) and last_label <= first_issue,
          "Refit labels must be disjoint from test and available by the first test issue")
    init = [("Ridge Regression", lambda **kw: make_pipeline(StandardScaler(), Ridge(**kw)),
             [{"alpha": alpha, "solver": ridge["solver"]} for alpha in ridge["alpha_candidates"]])]
    for key, model, factory in [("random_forest", "Random Forest", RandomForestRegressor),
                                ("gradient_boosting", "Gradient Boosting", GradientBoostingRegressor)]:
        setting = cfg[key]
        init.append((model, factory, [{**setting["fixed"], **params, "random_state": cfg["random_state"]}
                                    for params in setting["candidates"]]))
    trials, training, trained = [], [], {}
    for name, factory, candidates in init:
        selected, prediction, all_trials = select(factory, candidates, X.loc[train], y.loc[train], X.loc[val], y.loc[val])
        trials.extend({"model": name, "n_train": len(train), "n_validation": len(val), **row} for row in all_trials)
        model, seconds = fit(factory, json.loads(selected["parameters"]), X_refit, y_refit)
        fit_index = X_refit.index
        training.append({
            "model": name, "n_train": len(train), "n_validation": len(val), "n_final_refit": len(fit_index),
            "final_training_start": fit_index[0], "final_training_end": fit_index[-1], "n_test": len(test),
            "selected_hyperparameters": selected["parameters"], "final_training_timestamps_sha256": timestamp_digest(fit_index),
            "last_refit_label_available_at": last_label, "first_test_issue": first_issue,
        })
        trained[name] = (model, prediction, selected, seconds)
        print(f"{name}: selected {selected['parameters']} at validation MAE {selected['validation_MAE']:.6f}", flush=True)
    check(len({row["final_training_timestamps_sha256"] for row in training}) == 1,
          "Final fit timestamps differ across models")
    # Every choice and refit is frozen before any test prediction or scoring.
    naive = dataset.seasonal_naive
    val_naive, val_time = timed(lambda: naive.loc[val].to_numpy())
    test_naive, test_time = timed(lambda: naive.loc[test].to_numpy())
    results = {"Seasonal Naive": (val_naive, test_naive,
        {"parameters": json.dumps({"period_hours": period}), "fit_seconds": 0.0,
         "validation_prediction_seconds": val_time, "candidate": None,
         "train_MAE": score(y.loc[train], naive.loc[train])["MAE"]}, 0.0, test_time)}
    for name, (model, prediction, selected, seconds) in trained.items():
        test_pred, pred_seconds = timed(model.predict, X.loc[test])
        results[name] = (prediction, test_pred, selected, seconds, pred_seconds)

    frames = []
    for name, (vp, tp, *_) in results.items():
        variants = ["raw", "nonnegative"] if name in {"Ridge Regression", "Gradient Boosting"} else ["raw"]
        if name == "Random Forest":
            check(np.array_equal(vp, vp.clip(min=0)) and np.array_equal(tp, tp.clip(min=0)),
                  "The Random Forest forecast should have no negative predictions for nonnegative training targets")
        for part, values in [("validation", vp), ("test", tp)]:
            index = keys[part]
            for variant in variants:
                prediction = values if variant == "raw" else values.clip(min=0)
                check(np.isfinite(prediction).all(), "Nonfinite model prediction")
                actual = y.loc[index].to_numpy()
                frames.append(pd.DataFrame({
                    "protocol": cfg["protocol"], "partition": part, "site": cfg["site"],
                    "horizon_hours": dataset.horizon, "model": name, "prediction_variant": variant,
                    "forecast_origin_last_observed_hour": timeline.loc[index, "forecast_origin"].to_numpy(),
                    "forecast_issued_at": timeline.loc[index, "forecast_issued_at"].to_numpy(),
                    "target_hour": index, "target_end": timeline.loc[index, "target_end"].to_numpy(),
                    "actual": actual, "prediction": prediction, "residual_prediction_minus_actual": prediction - actual,
                }))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(trials), pd.DataFrame(training), results


def summarize(saved, results, training, keys, cfg):
    fit_table = training.set_index("model")
    rows = []
    train, val, test = (keys[part] for part in PARTITIONS)
    for (name, variant), block in saved.groupby(["model", "prediction_variant"], sort=False):
        _, _, selected, final_seconds, prediction_seconds = results[name]
        fitted = name in fit_table.index
        row = {
            "protocol": cfg["protocol"], "site": cfg["site"], "horizon_hours": cfg["horizon_hours"],
            "model": name, "prediction_variant": variant, "fit_policy": cfg["final_refit"] if fitted else "no_fit",
            "selected_candidate": selected["candidate"], "parameters": selected["parameters"],
            "n_train": len(train), "n_validation": len(val), "n_test": len(test),
            "n_final_fit": fit_table.at[name, "n_final_refit"] if fitted else 0,
            "first_test_issue": test[0],
            "last_fit_label_available_at": fit_table.at[name, "last_refit_label_available_at"] if fitted else pd.NaT,
            "candidate_train_MAE_raw": selected["train_MAE"], "validation_fit_seconds": selected["fit_seconds"],
            "validation_prediction_seconds": selected["validation_prediction_seconds"],
            "final_fit_seconds": final_seconds, "test_prediction_seconds": prediction_seconds,
        }
        for part in ("validation", "test"):
            split = block.loc[block.partition.eq(part)]
            check(pd.DatetimeIndex(split.target_hour).equals(keys[part]), f"Timestamp mismatch for {name}, {variant}")
            row.update({f"{part}_{metric}": v for metric, v in score(split.actual, split.prediction).items()})
            row[f"{part}_negative_predictions"] = int(split.prediction.lt(0).sum())
            row[f"{part}_start"], row[f"{part}_end"] = keys[part][[0, -1]]
        rows.append(row)
    check(len(rows) == 6, "Expected six official model variants")
    return pd.DataFrame(rows)


def verify_rerun(staged, official):
    for name in TABLES:
        before = pd.read_csv(official / name, float_precision="round_trip")
        after = pd.read_csv(staged / name, float_precision="round_trip")
        stable = [column for column in before if not column.endswith("_seconds")]
        pd.testing.assert_frame_equal(after[stable], before[stable], check_exact=True)
    for name in FIGURES:
        check(digest(staged / name) == digest(official / name), f"Regenerated plot differs: {name}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_mutually_exclusive_group()
    modes.add_argument("--replace", action="store_true", help="Replace the official outputs after staged generation")
    modes.add_argument("--verify", action="store_true", help="Rerun, compare results and leave official files untouched")
    args = cli.parse_args()
    for name in DELIVERABLES:
        path = (ROOT / name).resolve()
        check(path.is_relative_to(ROOT), f"Output outside Stage 3: {name}")
        if path.exists() and not (args.replace or args.verify):
            raise FileExistsError(f"{path} already exists. Use --verify or --replace.")
    config_path = ROOT / "config/tree_models_1h.yaml"
    cfg = read_yaml(config_path)
    framework_path = ROOT / cfg["framework_config"]
    framework = read_yaml(framework_path)
    baseline_path = ROOT / framework["legacy_config"]
    baseline = read_yaml(baseline_path)
    contract_path = baseline_path.parent / baseline["data_contract"]
    contract = read_yaml(contract_path)
    check(cfg["site"] == baseline["site"] == "Caltech" and cfg["horizon_hours"] == baseline["horizon_hours"] == 1
          and cfg["protocol"] == "boundary_aware" and cfg["boundary_policy"] == "exclude_partial"
          and cfg["selection_metric"] == "MAE"
          and cfg["final_refit"] == "train_and_validation_labels_available_by_first_test_issue",
          "Official runner requires boundary-aware Caltech 1h, validation MAE and a shared final refit")
    view, provenance_file = (ROOT / framework[key] for key in ("data_view", "provenance"))
    split_path = ROOT / cfg["split_summary"]
    watched = [config_path, framework_path, baseline_path, contract_path, view, provenance_file, split_path,
               *[ROOT / "src" / name for name in ("feature_engineering.py", "forecasting_dataset.py", "temporal_split.py")]]
    before = {str(p.relative_to(ROOT)): digest(p) for p in watched}
    hourly, provenance = load_hourly(view, provenance_file)
    check(all(provenance[key] == contract[key] for key in ("target", "clock", "coverage")), "Data contract mismatch")
    site, policy = cfg["site"], cfg["boundary_policy"]
    phase3 = pd.read_csv(split_path).query("scope == 'same_site' and direction == @site"
                                         " and horizon_hours == 1 and boundary_policy == @policy")
    check(len(phase3) == 3, "Missing boundary-aware Phase 3 split references")
    dataset, timeline, keys = planned_dataset(hourly.loc[hourly.site.eq(cfg["site"])], baseline, cfg, phase3)
    prediction, trials, training, details = run_models(
        dataset, timeline, keys, cfg, baseline["models"]["ridge"], baseline["models"]["seasonal_naive"]["period_hours"])
    trials.insert(0, "protocol", cfg["protocol"])
    env = {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
           "numpy": np.__version__, "matplotlib": matplotlib.__version__}
    with TemporaryDirectory(prefix=".unified_1h_", dir=ROOT) as directory:
        staged = Path(directory)
        for name in DELIVERABLES:
            (staged / name).parent.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(staged / PREDICTIONS, index=False)
        saved = pd.read_csv(staged / PREDICTIONS, float_precision="round_trip", parse_dates=["target_hour"])
        metrics = summarize(saved, details, training, keys, cfg)
        metrics.to_csv(staged / METRICS, index=False, na_rep="NA")
        trials.to_csv(staged / TRIALS, index=False)
        training.to_csv(staged / SAMPLES, index=False)
        baselines = ["Seasonal Naive", "Ridge Regression"]
        metrics.loc[metrics.model.isin(baselines)].to_csv(staged / BASELINE_METRICS, index=False, na_rep="NA")
        prediction.loc[prediction.model.isin(baselines)].to_csv(staged / BASELINE_PREDICTIONS, index=False)
        create_figures(metrics, saved, {name: staged / name for name in FIGURES})
        for name, contents in zip((REPORT, BASELINE_REPORT), render_reports(metrics, saved, training, provenance, env)):
            (staged / name).write_text(contents, encoding="utf-8")
        now = datetime.now(timezone(timedelta(hours=8))).isoformat()
        after = {str(p.relative_to(ROOT)): digest(p) for p in watched}
        check(after == before, "Protected experiment input changed")
        (staged / LOG).write_text("\n".join([
            f"{now} Official unified Caltech 1h boundary-aware revision",
            f"Runtime: {json.dumps(env, sort_keys=True)}",
            f"Protected input SHA-256 before: {json.dumps(before, sort_keys=True)}",
            f"Protected input SHA-256 after: {json.dumps(after, sort_keys=True)}",
            "Validation MAE alone selected one of eight candidates per learned model.",
            "Initial replay exposed floating-point reduction order in parallel RF prediction; serial prediction fixes it while keeping parallel fit.",
            "All final refits used the same ordered rows and ended before or at first test label cutoff.",
            training.to_csv(index=False).rstrip(),
            "Recomputed metrics from saved predictions and generated five figures from the new official tables.",
            "Stage 2 and raw data not accessed by the official runner. No other horizon or transfer fitted.", "",
        ]), encoding="utf-8")
        if args.verify:
            verify_rerun(staged, ROOT)
            print("Reproduced all six tables (excluding timings) and five figure hashes.")
        else:
            for name in DELIVERABLES:
                (ROOT / name).parent.mkdir(parents=True, exist_ok=True)
                (staged / name).replace(ROOT / name)
    print(metrics[["model", "prediction_variant", "validation_MAE", "test_MAE", "test_RMSE", "test_R2",
                   "test_negative_predictions"]].to_string(index=False))


if __name__ == "__main__":
    main()
