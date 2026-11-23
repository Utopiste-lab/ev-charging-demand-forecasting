"""Recompute the archived scoreboards and formally add a constant-zero forecast."""

import numpy as np
import pandas as pd

from stage3_context import CASE_SITES, FINAL_METRICS, FINAL_TABLES, HOUR, MATCHED, ROOT, Project, publish, read, timestamp_hash
from src.advanced_forecasting.evaluation import demand_metrics

GROUP = ["case", "horizon_hours", "model", "variant"]


def occurrence(y, pred):
    actual, forecast = np.asarray(y) > 0, np.asarray(pred) > 0
    tp = (actual & forecast).sum()
    positive, guessed = actual.sum(), forecast.sum()
    return {"occurrence_true_positive": tp, "occurrence_predicted_positive": guessed,
            "occurrence_precision": tp / guessed if guessed else 0.0,
            "occurrence_precision_defined": guessed > 0,
            "occurrence_recall": tp / positive if positive else np.nan,
            "occurrence_F1": 2 * tp / (positive + guessed) if positive + guessed else np.nan}


def point_summary(y, p):
    y, p = np.asarray(y), np.asarray(p)
    result = demand_metrics(y, p)
    for name, mask in (("zero", y == 0), ("positive", y > 0)):
        err = p[mask] - y[mask]
        result.update({f"{name}_n": len(err), f"{name}_proportion": mask.mean(),
                       f"{name}_MAE": np.abs(err).mean() if len(err) else np.nan,
                       f"{name}_RMSE": np.sqrt(np.square(err).mean()) if len(err) else np.nan})
    return {**result, **occurrence(y, p)}


def audit_metric(row, saved, point_names, count, part):
    out = demand_metrics(row.actual, row.prediction)
    assert len(row) == int(count), count
    for key in point_names:
        observed = np.nan if out[key] == "NA" else float(out[key])
        assert np.isclose(observed, float(saved[key]), atol=2e-11, rtol=2e-11, equal_nan=True), (part, key, out, saved)


def normalized_tables():
    base = ROOT / "results"
    t, m = base / "tables", base / "metrics"
    h1 = read(t / "tree_models_1h_predictions.csv")
    fallback = read(t / "baseline_predictions.csv")
    compare1 = read(m / "tree_models_1h_comparison.csv")
    subset = compare1.loc[compare1.model.isin(["Ridge Regression", "Seasonal Naive"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(read(m / "baseline_1h_unified.csv"), subset)
    pd.testing.assert_frame_equal(fallback, h1.loc[h1.model.isin(["Ridge Regression", "Seasonal Naive"])].reset_index(drop=True))
    rename = {"prediction_variant": "variant"}
    h4 = read(t / "multi_horizon_predictions.csv").rename(columns=rename).assign(case="Caltech full")
    p4 = pd.concat([read(m / (name + ".csv")) for name in ("multi_horizon_comparison","multi_horizon_clipped_comparison")],
                   ignore_index=True).rename(columns=rename).assign(case="Caltech full")
    cross = read(t / "cross_site_predictions.csv")
    cross["case"] = cross.direction.replace(MATCHED)
    cross = pd.concat([cross.rename(columns={"prediction_raw": "prediction"}).assign(variant="raw"),
                       cross.loc[cross.prediction_nonnegative.notna()].rename(columns={"prediction_nonnegative": "prediction"})
                       .assign(variant="nonnegative")], ignore_index=True)
    cscore = pd.concat([read(m / (name + ".csv")) for name in
                        ("cross_site_comparison","same_site_reference_comparison","cross_site_clipped_comparison")],
                       ignore_index=True)
    cscore["case"] = cscore.direction.replace(MATCHED)
    both7 = read(t / "advanced_forecasting/advanced_forecasting_predictions.csv")
    p7 = pd.concat([both7.rename(columns={"prediction_raw": "prediction"}).assign(variant="raw"),
                    both7.rename(columns={"prediction_nonnegative": "prediction"}).assign(variant="nonnegative")],
                   ignore_index=True)
    s7 = read(m / "advanced_forecasting/advanced_forecasting_comparison.csv")
    p8 = read(t / "adaptive_hybrid/adaptive_hybrid_predictions.csv")
    p8 = p8.loc[p8.window.eq("official")]
    s8 = read(m / "adaptive_hybrid/adaptive_hybrid_comparison.csv")
    p8 = p8.melt(id_vars=["case", "horizon_hours", "partition", "forecast_issued_at", "target_hour", "actual"],
                 value_vars=["model:" + name for name in s8.model.unique()], var_name="model", value_name="prediction")
    p8.model = p8.model.str.removeprefix("model:")
    p8 = p8.loc[p8.prediction.notna()].assign(variant="published")
    s8 = s8.assign(variant="published")
    return {"03": (h1.rename(columns=rename).assign(case="Caltech full"), compare1.rename(columns=rename).assign(case="Caltech full")),
            "04": (h4, p4), "06": (cross, cscore), "07": (p7, s7), "08": (p8, s8)}


def audit_and_baseline(project, out):
    frames = normalized_tables()
    counts = {site: frame.set_index("timestamp").session_start_count for site, frame in project.full.groupby("site")}
    summary, checks = [], []
    test_archive = {}
    for experiment, (pred, old_score) in frames.items():
        pred = pred.copy()
        pred["target_hour"] = pd.to_datetime(pred.target_hour)
        pred["forecast_issued_at"] = pd.to_datetime(pred.forecast_issued_at, format="mixed")
        groups = pred.groupby(GROUP + ["partition"], sort=False)
        if experiment == "07":
            old_score = old_score.loc[old_score.prediction_reference.eq("Experiment 07")]
        for ref in old_score.itertuples(index=False):
            fields = ref._asdict()
            key = tuple(fields[k] for k in GROUP)
            case, horizon, name, variant = key
            source, target = CASE_SITES[case]
            partitions = [fields["partition"]] if experiment == "08" else ["validation", "test"]
            for part in partitions:
                data = groups.get_group((*key, part)).sort_values("target_hour")
                keys = pd.DatetimeIndex(data.target_hour)
                site = target if part == "test" else source
                assert keys.is_unique and np.isfinite(data.prediction).all()
                if part == "test":
                    assert keys.equals(project.keys)
                np.testing.assert_array_equal(data.actual, counts[site].loc[keys])
                np.testing.assert_array_equal(data.forecast_issued_at, keys - (horizon - 1) * HOUR)
                names = ["MAE", "RMSE", "R2"]
                prefix = "" if experiment == "08" else part + "_"
                audit_metric(data, {k: fields[prefix + k] for k in names}, names,
                             fields["n" if experiment == "08" else ("n_test" if part == "test" else "n_validation")], (*key, part))
                checks.append({"experiment": experiment, **dict(zip(GROUP, key)), "partition": part, "n": len(keys),
                               "target_sha256": timestamp_hash(keys), "status": "verified"})
                if part != "test" or experiment == "03":
                    continue
                if name == "Seasonal Naive":
                    np.testing.assert_array_equal(data.prediction, counts[site].loc[keys - project.period * HOUR])
                row = {"experiment": experiment, "case": case, "source_site": source, "target_site": target,
                       "horizon_hours": horizon, "model": name, "variant": variant, "n": len(keys),
                       "target_sha256": timestamp_hash(keys), "fit_protocol": "historical_experiment_protocol",
                       **point_summary(data.actual, data.prediction)}
                summary.append(row)
                test_archive[(experiment, case, horizon, name, variant)] = data.prediction.to_numpy()
    for h in project.horizons:
        for c, (_, target) in CASE_SITES.items():
            for old, name in [("Hurdle", "Hurdle"), ("Seasonal Residual", "Residual")]:
                for v7, v8 in [("raw", name), ("nonnegative", name+" clipped")]:
                    np.testing.assert_allclose(test_archive["07", c, h, old, v7], test_archive["08", c, h, v8, "published"],
                                               atol=1e-11, rtol=0)
    zero = []
    for case, (source, target) in CASE_SITES.items():
        y = counts[target].loc[project.keys]
        for h in project.horizons:
            pred = np.zeros(len(y))
            zero.append(pd.DataFrame({"case": case, "source_site": source, "target_site": target, "horizon_hours": h,
                                      "target_hour": y.index, "forecast_issued_at": y.index - (h-1)*HOUR,
                                      "actual": y.to_numpy(), "prediction": pred}))
            summary.append({"experiment": "final", "case": case, "source_site": source, "target_site": target,
                            "horizon_hours": h, "model": "Constant Zero", "variant": "raw", "n": len(y),
                            "target_sha256": timestamp_hash(y.index), "fit_protocol": "no_fit",
                            **point_summary(y, pred)})
            assert np.isclose(point_summary(y, pred)["MAE"], y.mean()) and occurrence(y,pred)["occurrence_recall"] == 0
    metrics = pd.DataFrame(summary)
    baseline = metrics.loc[metrics.model.isin(["Constant Zero", "Seasonal Naive"])]
    baseline = baseline.drop_duplicates(["case", "horizon_hours", "model"])
    outputs = {FINAL_TABLES / "constant_zero_predictions.csv": pd.concat(zero, ignore_index=True),
               FINAL_TABLES / "historical_prediction_audit.csv": pd.DataFrame(checks),
               FINAL_METRICS / "archived_point_comparison.csv": metrics,
               FINAL_METRICS / "constant_zero_vs_seasonal_naive.csv": baseline}
    for k, frame in outputs.items():
        publish(frame, out / k)
    print(f"Historical readback verified {len(checks)} test/validation score rows and saved {len(zero)} constant-zero test settings.", flush=True)
    return list(outputs)


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from stage3_context import digest

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    if parser.parse_args().verify:
        with TemporaryDirectory(dir=ROOT, prefix=".stage3-baseline-") as d:
            stage = Path(d)
            for file in audit_and_baseline(Project(), stage):
                assert digest(stage/file) == digest(ROOT/file), file
        print("Reproduced all four archived-audit and constant-zero files.")
    else:
        audit_and_baseline(Project(), ROOT)
