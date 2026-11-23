"""Supplement the fixed-window experiment with predeclared equal-length rolling windows."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from audit_previous_results import METRICS, ROOT, TABLE, Project, protect, publish, read, verify_protection
from src.adaptive_hybrid import experts
from src.adaptive_hybrid.evaluation import bootstrap, score_groups


OTHER = ("Ridge Regression", "Random Forest", "Gradient Boosting")


def rolling_windows(project):
    cfg = project.cfg["rolling"]
    duration = {name: pd.Timedelta(days=cfg[f"{name}_days"]) for name in ("train", "validation", "test")}
    for number, value in enumerate(cfg["first_test_starts"], 1):
        test = pd.Timestamp(value)
        val = test - duration["validation"]
        windows = {"train": (val - duration["train"], val), "validation": (val, test),
                   "test": (test, test + duration["test"])}
        assert windows["test"][1] <= project.windows["test"][0]
        yield f"R{number}", windows


def ordinary_references(project, dataset, ledger, keys, horizon):
    sys.path.insert(0, str(ROOT / "experiments/04_multi_horizon"))
    from tune_models import tune_horizon
    cfg = project.cfg["rolling"]
    baseline = project.baseline
    base_models = baseline["models"]
    baseline = {**baseline, "models": {**base_models, "ridge": {
        **base_models["ridge"], "alpha_candidates": cfg["ridge_candidates"]}}}
    unified = project.one_hour
    changed = {**unified}
    for key, choice in (("random_forest", "forest_candidate_indices"), ("gradient_boosting", "boosting_candidate_indices")):
        params = unified[key]
        changed[key] = {**params, "candidates": [params["candidates"][i] for i in cfg[choice]]}
    splits = {name: ledger.index[ledger.partition.eq(name) & ledger.status.eq("eligible")]
              for name in ("train", "validation", "test")}
    assert splits["test"].equals(keys)
    pred, _, _, _ = tune_horizon(dataset, ledger, splits, changed, baseline)
    pred = pred.loc[pred.partition.eq("test") & pred.prediction_variant.eq("raw") & pred.model.isin(OTHER)]
    result = pred.pivot(index="target_hour", columns="model", values="prediction")
    assert result.index.equals(keys)
    return result


def archived_experts(project):
    rows = project.previous_forecasts
    rows = rows.loc[rows.partition.eq("test")]
    entries = []
    for (case, h), frame in rows.groupby(["case", "horizon_hours"], sort=False):
        one = frame.loc[frame.model.eq("Hurdle")].set_index("target_hour").sort_index()
        two = frame.loc[frame.model.eq("Seasonal Residual")].set_index("target_hour").sort_index()
        assert one.index.equals(two.index) and one.index.equals(project.keys) and one.actual.equals(two.actual)
        target = one.target_site.iat[0]
        counts = project.full.loc[project.full.site.eq(target)].set_index("timestamp").session_start_count
        previous = counts.shift(project.period).loc[one.index]
        subset = project.previous_scores
        scores = subset.loc[subset.case.eq(case) & subset.horizon_hours.eq(h)].set_index(["model", "variant"])
        from src.adaptive_hybrid.evaluation import metrics
        refs = {"Seasonal Naive": previous, "Hurdle": one.prediction_raw, "Hurdle clipped": one.prediction_nonnegative,
                "Residual": two.prediction_raw, "Residual clipped": two.prediction_nonnegative}
        for label, pred in refs.items():
            name = label.removesuffix(" clipped").replace("Residual", "Seasonal Residual")
            variant = "nonnegative" if label.endswith("clipped") else "raw"
            assert np.isclose(metrics(one.actual, pred)["MAE"], scores.at[(name, variant), "test_MAE"], atol=1e-12)
        entries.append(pd.DataFrame({"case": case, "horizon_hours": h, "target_hour": one.index, "actual": one.actual,
                                     **{f"model:{name}": v for name, v in refs.items()}}))
    return pd.concat(entries, ignore_index=True)


def bootstrap_table(frame, project, suite, models, baseline="Seasonal Naive"):
    opts = project.cfg["bootstrap"]
    out = []
    for (case, h), entry in frame.groupby(["case", "horizon_hours"], sort=False):
        rows = entry.sort_values("target_hour")
        keys = pd.DatetimeIndex(rows.target_hour)
        assert keys.is_unique and keys.equals(project.keys)
        y = rows.actual.to_numpy()
        b = rows[f"model:{baseline}"].to_numpy()
        for name in models:
            p = rows[f"model:{name}"].to_numpy()
            out.extend(dict(r, suite=suite, case=case, horizon_hours=h, model=name, baseline=baseline)
                       for r in bootstrap(y, p, b, opts["blocks_hours"], opts["replicates"], project.cfg["seed"]))
    return pd.DataFrame(out)


def execute(project, out, *, poison=False):
    forecasts, trials, splits = [], [], []
    for label, windows in rolling_windows(project):
        for case, site in (("Caltech full", "Caltech"), ("JPL matched", "JPL")):
            for h in project.horizons:
                fit, data, ledger, target, actual, keys = project.payload(case, site, site, h, windows=windows, poison=poison)
                opts, options = experts.select(fit, data.seasonal_naive, project.advanced, project.cfg["seed"])
                model, _ = experts.fit(fit, data.seasonal_naive, opts, project.cfg["seed"],
                                       project.cfg["median_positive_forest"])
                p = experts.predict(model, target.X, target.seasonal_naive)
                perturbed, blocked = project.historical_perturbation(case, site, h, windows, target)
                changed = experts.predict(model, perturbed.X, perturbed.seasonal_naive)
                pd.testing.assert_frame_equal(p.loc[blocked], changed.loc[blocked], check_exact=True)
                # The legacy comparison API materializes scoring labels for its return
                # table. Forecasting with the poisoned label buffer must still agree.
                ref = ordinary_references(project, data, ledger, target.X.index, h)
                meta = {"window": label, "case": case, "horizon_hours": h}
                saved = pd.DataFrame({**meta, "target_hour": p.index, "actual": actual,
                                      "forecast_issued_at": target.timeline.forecast_issued_at})
                for name in (*experts.CORE, "Hurdle", "Residual", "Hurdle median", *OTHER):
                    saved[f"model:{name}"] = (p if name in p else ref)[name]
                for name in (experts.PROBABILITY, "median_cdf_below", "median_cdf_at"):
                    saved[name] = p[name]
                assert len(saved) == project.cfg["rolling"]["test_days"] * 24 and saved.index.is_unique
                forecasts.append(saved)
                trials.extend({**meta, **r, "parameters": json.dumps(r["parameters"], sort_keys=True)}
                              for r in options)
                splits.extend(dict(r, window=label) for r in keys)
                print(f"Part I rolling {label} {case} {h}h verified", flush=True)
    pred = pd.concat(forecasts, ignore_index=True)
    summary, subsets = score_groups(pred, ["window", "case", "horizon_hours"])
    past = archived_experts(project)
    intervals = bootstrap_table(past, project, "Experiment 07", ["Hurdle", "Hurdle clipped", "Residual", "Residual clipped"])
    for name, frame in {
        TABLE / "rolling_experiment07_predictions.csv": pred,
        TABLE / "rolling_experiment07_selections.csv": pd.DataFrame(trials).drop(columns="candidate_fit_seconds"),
        TABLE / "rolling_experiment07_splits.csv": pd.DataFrame(splits),
        METRICS / "rolling_experiment07_comparison.csv": summary,
        METRICS / "rolling_experiment07_subgroups.csv": subsets,
        METRICS / "experiment07_bootstrap.csv": intervals,
    }.items():
        publish(frame, out / name)
    return pred


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(ROOT)
    original = protect()
    execute(Project(), args.out)
    verify_protection(original)
