"""Use the official one-hour selection and refit implementation at each horizon."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_tree_models_1h"))
from run_comparison import run_models, summarize, timestamp_digest


def tune_horizon(dataset, timeline, keys, one_hour, baseline):
    settings = {**one_hour, "horizon_hours": dataset.horizon}
    ridge = baseline["models"]["ridge"]
    period = baseline["models"]["seasonal_naive"]["period_hours"]
    prediction, trials, training, fitted = run_models(dataset, timeline, keys, settings, ridge, period)
    for table in (trials, training):
        table.insert(0, "horizon_hours", dataset.horizon)
        table.insert(0, "protocol", settings["protocol"])
    return prediction, trials, training, fitted


def summarize_horizon(saved, fitted, training, keys, one_hour, horizon):
    settings = {**one_hour, "horizon_hours": horizon}
    metrics = summarize(saved, fitted, training, keys, settings)
    # The 1h writer records its target timestamp as its issue timestamp. For
    # longer horizons the issued_at field in the saved records precedes target.
    metrics["first_test_issue"] = saved.loc[saved.partition.eq("test"), "forecast_issued_at"].min()
    return metrics
