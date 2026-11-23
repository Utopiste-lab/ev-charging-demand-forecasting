"""Independently recompute saved Experiment 07 evidence without fitting estimators."""

import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.advanced_forecasting.evaluation import demand_metrics, occurrence_metrics

TABLE = Path("results/tables/advanced_forecasting")
METRIC = Path("results/metrics/advanced_forecasting")
KEY = ["case", "horizon_hours", "partition", "target_hour"]


def sha(index):
    return hashlib.sha256("".join(f"{hour.isoformat()}\n" for hour in index).encode("ascii")).hexdigest()


def get_csv(root, path):
    return pd.read_csv(root / path, float_precision="round_trip")


def verify_protection(root, protected):
    assert protected.project_relative_path.is_unique
    for row in protected.itertuples():
        path = (root.parent / row.project_relative_path).resolve()
        assert path.is_relative_to(root) or path.is_relative_to(root.with_name("2"))
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == row.sha256, path


def independent_score(actual, forecast):
    y, p = np.asarray(actual), np.asarray(forecast)
    n = len(y)
    errors = p - y
    squares = math.fsum(errors * errors)
    sst = math.fsum((y - math.fsum(y) / n) ** 2)
    return {"MAE": math.fsum(abs(errors)) / n, "RMSE": math.sqrt(squares / n),
            "R2": 1 - squares / sst if sst else np.nan, "bias": math.fsum(errors) / n,
            "negative_predictions": int((p < 0).sum())}


def verify_case(prediction, hurdle, residual, trials, fitting, values, keys, h, threshold):
    assert len(fitting) == 2 and len(trials) == 17
    assert not prediction.duplicated(KEY + ["model"]).any()
    for model, component, cols in [
        ("Hurdle", hurdle, ["probability", "conditional_magnitude_raw"]),
        ("Seasonal Residual", residual, ["seasonal_naive", "predicted_residual"]),
    ]:
        base = prediction.loc[prediction.model.eq(model)]
        component = component.set_index(KEY).loc[pd.MultiIndex.from_frame(base[KEY])].reset_index()
        first, second = (component[c].to_numpy() for c in cols)
        raw = first * second if model == "Hurdle" else first + second
        np.testing.assert_allclose(raw, base.prediction_raw, atol=1e-13, rtol=0)
        np.testing.assert_array_equal(base.prediction_nonnegative, base.prediction_raw.clip(lower=0))
        val = base.loc[base.partition.eq("validation")]
        test = base.loc[base.partition.eq("test")]
        chosen = trials.loc[trials.model.eq(model)].sort_values(["validation_MAE", "candidate"]).iloc[0]
        r = fitting.loc[fitting.model.eq(model)].iloc[0]
        assert chosen.selected and chosen.parameters == r.chosen_parameters and chosen.candidate == r.chosen_candidate
        assert abs(independent_score(val.actual, val.prediction_raw)["MAE"] - chosen.validation_MAE) <= 1e-12
        assert r.selection_site == r.fit_site == r.source_site and r.future_perturbation == "PASS"
        assert r.predictions_checked_before_release == h
        assert pd.Timestamp(r.last_fit_label_available_at) <= pd.Timestamp(r.first_test_issue)
        np.testing.assert_array_equal(pd.DatetimeIndex(test.target_hour), keys)
        assert len(test) == r.n_test == 3383
        for name, site in [("validation", r.source_site), ("test", r.target_site)]:
            frame = base.loc[base.partition.eq(name)]
            idx = pd.DatetimeIndex(frame.target_hour)
            np.testing.assert_array_equal(frame.actual, values[site].loc[idx])
            np.testing.assert_array_equal(pd.DatetimeIndex(frame.forecast_issued_at), idx - pd.Timedelta(hours=h - 1))
            for variant in ("raw", "nonnegative"):
                y, p = frame.actual, frame[f"prediction_{variant}"]
                close(independent_score(y, p), demand_metrics(y, p))
                for positive in (False, True):
                    mask = y.gt(0).eq(positive)
                    close(independent_score(y.loc[mask], p.loc[mask]), demand_metrics(y.loc[mask], p.loc[mask]))
            if model == "Seasonal Residual":
                reference = component.loc[component.partition.eq(name), "seasonal_naive"]
                np.testing.assert_array_equal(reference, values[site].loc[idx - pd.Timedelta(hours=24)])
            else:
                p = component.loc[component.partition.eq(name)]
                y = frame.actual.gt(0).to_numpy()
                close(independent_score(frame.actual[y], p.conditional_magnitude_raw.to_numpy()[y]),
                      demand_metrics(frame.actual[y], p.conditional_magnitude_raw.to_numpy()[y]))
                close(independent_occurrence(y, p.probability, threshold),
                      occurrence_metrics(frame.actual, p.probability, threshold))


def independent_occurrence(y, probability, threshold):
    prob = np.asarray(probability)
    decision = prob >= threshold
    tp = int(np.count_nonzero(y & decision))
    positives, negatives = int(y.sum()), int((~y).sum())
    precision, recall, _ = precision_recall_curve(y, prob)
    return {
        "precision": tp / max(1, decision.sum()), "recall": tp / positives,
        "F1": 2 * tp / (positives + decision.sum()),
        "ROC_AUC": (rankdata(prob)[y].sum() - positives * (positives + 1) / 2) / (positives * negatives),
        "PR_AUC": auc(recall, precision), "average_precision": average_precision_score(y, prob),
        "Brier": np.square(prob - y).mean(),
    }


def close(result, saved):
    for metric, value in result.items():
        target = saved[metric]
        if pd.isna(value):
            assert pd.isna(target) or target == "NA", metric
        else:
            assert math.isclose(value, target, abs_tol=1e-12, rel_tol=0), (metric, value, target)


def verify_saved(root, results_root):
    table = lambda name: get_csv(results_root, TABLE / f"{name}.csv")
    metrics = lambda name: get_csv(results_root, METRIC / f"{name}.csv")
    prediction, hurdle, residual, trials, splits = map(table, [
        "advanced_forecasting_predictions", "hurdle_predictions", "residual_learning_predictions",
        "advanced_forecasting_hyperparameters", "advanced_forecasting_split_summary"])
    comparison, classification, magnitude, reconstructed, subset, gap, fitting = map(metrics, [
        "advanced_forecasting_comparison", "hurdle_classification_metrics", "hurdle_positive_regression_metrics",
        "residual_learning_comparison", "zero_positive_performance", "cross_site_advanced_forecasting",
        "advanced_forecasting_training_verification"])
    calibration = table("hurdle_calibration")
    config = yaml.safe_load((root / "config/advanced_forecasting.yaml").read_text("utf-8"))
    hourly = get_csv(root, "data/processed/hourly_counts_v1.csv")
    hourly["timestamp"] = pd.to_datetime(hourly.timestamp)
    values = {site: rows.set_index("timestamp").session_start_count for site, rows in hourly.groupby("site")}
    keys = pd.date_range("2020-07-15 00:00", "2020-12-02 22:00", freq="h")
    assert len(fitting) == 40 and len(trials) == 340 and len(splits) == 80 and len(classification) == 40
    assert len(hurdle) == len(residual) and len(prediction) == 2 * len(hurdle)
    assert not any(name.startswith("test") for name in trials.columns)
    for (case, h), s in splits.groupby(["case", "horizon_hours"]):
        family = lambda frame: frame.loc[frame.case.eq(case) & frame.horizon_hours.eq(h)]
        rows, hrows, rrows, selected, verf = map(family, (prediction, hurdle, residual, trials, fitting))
        verify_case(rows, hrows, rrows, selected, verf, values, keys, h, config["probability_threshold"])
        parts = s.set_index("partition")
        train, val = (pd.date_range(parts.at[p, "target_start"], parts.at[p, "target_end_inclusive"], freq="h")
                      for p in ("train", "validation"))
        final = train.append(val)
        assert len(train) == parts.at["train", "n"] and len(val) == parts.at["validation", "n"]
        assert verf.train_sha256.eq(sha(train)).all() and verf.validation_sha256.eq(sha(val)).all()
        assert verf.final_sha256.eq(sha(final)).all() and verf.test_sha256.eq(sha(keys)).all()
        assert verf.n_final.eq(len(final)).all() and final.intersection(keys).empty
        assert train.max() + pd.Timedelta(hours=1) <= pd.Timestamp(parts.at["validation", "first_issue"])
        assert final.max() + pd.Timedelta(hours=1) <= pd.Timestamp(parts.at["test", "first_issue"])
        site = verf.source_site.iat[0]
        training, all_fit = values[site].loc[train], values[site].loc[final]
        hp = verf.loc[verf.model.eq("Hurdle")].iloc[0]
        for term, counts in [("initial", training), ("final", all_fit)]:
            positive = counts.loc[counts.gt(0)]
            assert hp[f"occurrence_{term}_hash"] == sha(counts.index)
            assert hp[f"occurrence_n_{term}"] == len(counts)
            assert hp[f"magnitude_{term}_hash"] == sha(positive.index)
            assert hp[f"magnitude_n_{term}"] == len(positive)
        rr = verf.loc[verf.model.eq("Seasonal Residual")].iloc[0]
        signed = all_fit.to_numpy() - values[site].loc[final - pd.Timedelta(hours=24)].to_numpy()
        assert rr.residual_initial_hash == sha(train) and rr.residual_final_hash == sha(final)
        assert rr.residual_n_initial == len(train) and rr.residual_n_final == len(final)
        assert rr.ordered_residual_values_sha256 == hashlib.sha256(signed.astype("<f8").tobytes()).hexdigest()
        assert int((signed < 0).sum()) == rr.negative_residual_targets

    full = get_csv(root, "results/tables/multi_horizon_predictions.csv")
    cross = get_csv(root, "results/tables/cross_site_predictions.csv")
    for entry in comparison.itertuples(index=False):
        row = entry._asdict()
        if entry.model in {"Hurdle", "Seasonal Residual"}:
            block = prediction.loc[prediction.case.eq(entry.case) & prediction.horizon_hours.eq(entry.horizon_hours)
                                   & prediction.model.eq(entry.model)]
            field = f"prediction_{entry.variant}"
        elif entry.case == "Caltech full":
            block = full.loc[full.horizon_hours.eq(entry.horizon_hours) & full.model.eq(entry.model)
                             & full.prediction_variant.eq(entry.variant)]
            field = "prediction"
        else:
            direction = {"Caltech matched": "Caltech->Caltech", "JPL matched": "JPL->JPL"}.get(entry.case, entry.case)
            block = cross.loc[cross.direction.eq(direction) & cross.horizon_hours.eq(entry.horizon_hours)
                              & cross.model.eq(entry.model)]
            field = f"prediction_{entry.variant}"
        for partition, part in block.groupby("partition"):
            result = independent_score(part.actual, part[field])
            close({f"{partition}_{k}": v for k, v in result.items()}, row)
            if partition == "test":
                np.testing.assert_array_equal(pd.DatetimeIndex(part.target_hour), keys)
                np.testing.assert_array_equal(part.actual, values[entry.target_site].loc[keys])
                for name, mask in [("zero", part.actual.eq(0)), ("positive", part.actual.gt(0))]:
                    found = subset.loc[subset.case.eq(entry.case) & subset.horizon_hours.eq(entry.horizon_hours)
                                       & subset.model.eq(entry.model) & subset.variant.eq(entry.variant) & subset.group.eq(name)]
                    assert len(found) == 1 and found.n.iat[0] == mask.sum()
                    close(independent_score(part.loc[mask, "actual"], part.loc[mask, field]), found.iloc[0])
    for _, r in classification.iterrows():
        chosen = (hurdle.case.eq(r["case"]) & hurdle.horizon_hours.eq(r.horizon_hours) & hurdle.partition.eq(r.partition))
        p = hurdle.loc[chosen].merge(prediction.loc[prediction.model.eq("Hurdle"), KEY + ["actual"]],
                                     on=KEY, validate="one_to_one")
        y = p.actual.gt(0).to_numpy()
        prob = p.probability.to_numpy()
        close(independent_occurrence(y, prob, config["probability_threshold"]), r)
        m = magnitude.loc[magnitude.case.eq(r["case"]) & magnitude.horizon_hours.eq(r.horizon_hours) & magnitude.partition.eq(r.partition)]
        close(independent_score(p.loc[y, "actual"], p.loc[y, "conditional_magnitude_raw"]), m.iloc[0])
        if r.partition == "test":
            bins = np.minimum((prob * config["calibration_bins"]).astype(int), config["calibration_bins"] - 1)
            for _, b in calibration.loc[calibration.case.eq(r["case"]) & calibration.horizon_hours.eq(r.horizon_hours)].iterrows():
                mask = bins == b["bin"]
                assert mask.sum() == b.n
                close({"mean_probability": prob[mask].mean(), "observed_positive_rate": y[mask].mean()}, b)
    for r in gap.itertuples():
        assert math.isclose(r.transfer_MAE - r.matched_MAE, r.MAE_gap, abs_tol=1e-12)
        assert math.isclose(r.transfer_RMSE - r.matched_RMSE, r.RMSE_gap, abs_tol=1e-12)
    for r in reconstructed.itertuples():
        assert math.isclose(r.seasonal_MAE - r.residual_MAE, r.MAE_improvement, abs_tol=1e-12)
        assert math.isclose(r.seasonal_RMSE - r.residual_RMSE, r.RMSE_improvement, abs_tol=1e-12)
    verify_protection(root, table("protected_file_hashes"))
    print(f"Verified {len(prediction):,} new predictions, {len(comparison)} comparison scores, "
          f"{len(classification)} classification/magnitude rows, and {len(fitting)} fit attestations.", flush=True)


if __name__ == "__main__":
    stage3 = Path(__file__).resolve().parents[2]
    verify_saved(stage3, stage3)
