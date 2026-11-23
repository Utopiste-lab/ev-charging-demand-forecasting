"""Train and evaluate Experiment 09 or replay it without changing official results."""

import argparse
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from stage3_context import CASE_SITES, HOUR, PROB_METRICS, PROB_TABLES, ROOT, Project, config, publish, read, timestamp_hash
from calibrate_intervals import apply, calibrate, predict
from evaluate_probabilistic import evaluate
from run_rolling_validation import windows as rolling_windows
from train_quantile_models import train
from src.probabilistic_forecasting.calibration import release_split


PREDICTIONS = PROB_TABLES / "probabilistic_predictions.csv"


def settings(project, cfg):
    for case, (source, target) in CASE_SITES.items():
        if cfg["cross_site"] or (source == target and case != "Caltech matched"):
            yield "official", case, source, target, project.windows
    for window, calendar in rolling_windows():
        for case in ("Caltech full", "JPL matched"):
            site = CASE_SITES[case][0]
            yield window, case, site, site, calendar


def source_model(project, source_case, site, h, calendar, cfg, days, poison):
    data, _ = project.dataset(source_case, site, h, calendar)
    keys, start = release_split(data, calendar, days)
    if poison:
        # Perturb evaluation-only labels. Roll-forward history remains unchanged.
        data = replace(data, y=data.y.mask(data.y.index.isin(keys["test"]), -999997))
    models, trials, tuning = train(data, keys, cfg)
    cal = keys["calibration"]
    block = predict(models, data.X.loc[cal], data.seasonal_naive.loc[cal])
    scores, adj, rank = calibrate(block, data.y.loc[cal], cfg["coverage_nominal"])
    merged = apply(block, adj)
    merged["actual"] = data.y.loc[cal]
    merged["nonconformity"] = scores
    merged["forecast_issued_at"] = data.timeline.loc[cal, "forecast_issued_at"]
    return models, trials, tuning, keys, start, data.timeline, merged, adj, rank


def run(project, cfg, output, poison=False):
    packets, forecasts, tuning_rows, hyperparameters, attestations, adjustments = {}, [], [], [], [], []
    for window, case, src, tgt, calendar in settings(project, cfg):
        for h in project.horizons:
            official = window == "official"
            extra = {} if official else {"windows": calendar}
            _, _, _, target, truth, _ = project.payload(case, src, tgt, h, poison=poison, **extra)
            days = cfg["calibration_days"]["official" if official else "rolling"]
            fit_case = case if src == tgt else src + " matched"
            cache_key = (window, fit_case, h)
            if cache_key not in packets:
                packets[cache_key] = source_model(project, fit_case, src, h, calendar, cfg, days, poison)
            models, trials, tuning, keys, start, timeline, cal, adj, rank = packets[cache_key]
            tag = dict(window=window, case=case, horizon_hours=h, source_site=src, target_site=tgt)
            cal_record = cal.assign(**tag, partition="calibration", observation_site=src)
            test = apply(predict(models, target.X, target.seasonal_naive), adj)
            test["actual"] = truth
            test["forecast_issued_at"] = target.timeline.forecast_issued_at
            test = test.assign(**tag, partition="test", observation_site=tgt)
            assert target.X.index.equals(keys["test"])
            assert len(test) == (len(project.keys) if official else 1008)
            assert not cal.index.intersection(test.index).size
            assert timeline.loc[keys["calibration"], "target_end"].max() <= target.timeline.forecast_issued_at.min()
            altered, before = project.historical_perturbation(case, tgt, h, calendar, target)
            unaffected = predict(models, altered.X.loc[before], altered.seasonal_naive.loc[before])
            compare = [c for c in unaffected if c not in {"actual", "forecast_issued_at"}]
            pd.testing.assert_frame_equal(test.loc[before, compare], unaffected, check_exact=True)
            for part in ["train", "validation", "final_fit", "calibration", "test"]:
                index = keys[part]
                times = timeline.loc[index] if part != "test" else target.timeline
                attestations.append({**tag, "partition": part, "n": len(index), "start": index[0], "end": index[-1],
                                     "fit_or_observation_site": tgt if part == "test" else src,
                                     "timestamp_sha256": timestamp_hash(index), "first_issue": times.forecast_issued_at.min(),
                                     "last_label_available": times.target_end.max(), "calendar_calibration_start": start,
                                     "future_perturbation": "passed"})
            adjustments.append({**tag, "n_calibration": len(cal), "rank_1_based": rank, "adjustment": adj,
                                "nominal": cfg["coverage_nominal"], "calibration_sha256": timestamp_hash(cal.index),
                                "frozen_before_first_test_issue": True})
            hyperparameters.extend({**tag, **r} for r in trials)
            tuning_rows.append(tuning.assign(**tag))
            forecasts.extend([f.rename_axis("target_hour").reset_index() for f in (cal_record, test)])
            print(f"{window} {case} {h}h n_cal={len(cal)} adj={adj:.6g} n_test={len(test)} verified", flush=True)
    pred = pd.concat(forecasts, ignore_index=True)
    tables = {PREDICTIONS: pred, PROB_TABLES / "tuning_predictions.csv": pd.concat(tuning_rows, ignore_index=True),
              PROB_TABLES / "selected_hyperparameters.csv": pd.DataFrame(hyperparameters),
              PROB_TABLES / "calibration_adjustments.csv": pd.DataFrame(adjustments),
              PROB_TABLES / "temporal_verification.csv": pd.DataFrame(attestations)}
    tables.update({PROB_METRICS / (k + ".csv"): v for k, v in evaluate(pred, cfg).items()})
    versions = {"seed": cfg["seed"], "python": __import__("platform").python_version(),
                **{dep: version(dep) for dep in ["numpy","pandas","scikit-learn","scipy","PyYAML","matplotlib"]}}
    for path, table in tables.items():
        publish(table, output / path)
    environment = PROB_TABLES / "execution_environment.json"
    (output / environment).write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    return [*tables, environment]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if (ROOT / PREDICTIONS).exists() and not args.verify:
        raise FileExistsError("Experiment 09 results exist; use --verify")
    project, cfg = Project(), config("config/probabilistic_forecasting.yaml")
    with TemporaryDirectory(dir=ROOT, prefix=".prob09-") as scratch:
        temp = Path(scratch)
        files = run(project, cfg, temp, poison=args.verify)
        from verify_experiment import verify_saved
        verify_saved(project, cfg, temp)
        for path in files:
            if args.verify:
                if path.suffix == ".csv":
                    pd.testing.assert_frame_equal(read(temp / path), read(ROOT / path), check_exact=True)
                else:
                    assert (temp / path).read_bytes() == (ROOT / path).read_bytes()
            else:
                p = ROOT / path
                p.parent.mkdir(parents=True, exist_ok=True)
                with (temp / path).open("rb") as src, p.open("xb") as out:
                    shutil.copyfileobj(src, out)
        print(("Clean poisoned-label replay reproduced" if args.verify else "Published") + f" {len(files)} Experiment 09 artifacts.",
              flush=True)


if __name__ == "__main__":
    main()
