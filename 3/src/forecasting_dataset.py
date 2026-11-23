"""Read the processed hourly view and assemble direct forecasting datasets."""

from dataclasses import dataclass
import hashlib
import json

import pandas as pd

from src.feature_engineering import HOUR, build_features, feature_catalog


@dataclass
class ForecastingDataset:
    X: pd.DataFrame
    y: pd.Series
    seasonal_naive: pd.Series
    timeline: pd.DataFrame
    horizon: int


def load_hourly(view_path, provenance_path):
    source = json.loads(provenance_path.read_text(encoding="utf-8"))
    with view_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source["view"]["sha256"]:
            raise ValueError("Processed view SHA-256 does not match provenance")
        stream.seek(0)
        view = pd.read_csv(stream, dtype={"session_start_count": "int64",
                                        "is_first_hour": bool, "is_last_hour": bool})
    representation = view.timestamp
    view["timestamp"] = pd.to_datetime(representation, format=source["clock"]["timestamp_format"])
    if (view.isna().any().any() or view.timestamp.dt.tz is not None
            or not view.timestamp.dt.strftime(source["clock"]["timestamp_format"]).eq(representation).all()
            or not view.timestamp.dt.floor("h").eq(view.timestamp).all()
            or not view.session_start_count.ge(0).all()
            or len(view) != source["view"]["rows"]
            or set(view.site) != set(source["sites"])):
        raise ValueError("Hourly view violates the data contract")
    for column, value in {"clock_basis": source["clock"]["basis"],
                          "timezone_status": source["clock"]["timezone_status"],
                          "coverage_status": source["coverage"]["status"]}.items():
        if not view[column].eq(value).all():
            raise ValueError(f"Unexpected value in {column}")
    for site, rows in view.groupby("site"):
        stats = source["sites"][site]
        ts = rows.timestamp
        expected = (stats["hours_in_series"], pd.Timestamp(stats["first_hour"]),
                    pd.Timestamp(stats["last_hour"]), stats["zero_hours"],
                    stats["max_starts_in_one_hour"], stats["total_session_starts"])
        counts = rows.session_start_count
        if ((len(rows), ts.min(), ts.max(), counts.eq(0).sum(), counts.max(), counts.sum()) != expected
                or not rows.is_first_hour.eq(ts.eq(ts.min())).all()
                or not rows.is_last_hour.eq(ts.eq(ts.max())).all()):
            raise ValueError(f"{site}: counts, range or boundary flags disagree with provenance")
    return view, source


def build_dataset(site_rows, settings, horizon, period, *, policy, shared_partial_hours=None):
    """Targets index predictors, labels and metadata separately; the input is read-only."""
    if policy not in {"exclude_partial", "retain_sensitivity"}:
        raise ValueError(f"Unknown boundary policy: {policy}")
    if site_rows.site.nunique() != 1:
        raise ValueError("A dataset requires exactly one site")
    rows = site_rows.set_index("timestamp").sort_index()
    index = rows.index
    if not index.equals(pd.date_range(index[0], index[-1], freq=HOUR)):
        raise ValueError("Hourly index must be unique and continuous")
    X, seasonal = build_features(rows.session_start_count, settings, horizon, period)
    origin = index - horizon * HOUR
    timeline = pd.DataFrame({
        "forecast_origin": origin,
        "forecast_issued_at": origin + HOUR,
        "target_end": index + HOUR,
        "seasonal_reference_hour": index - period * HOUR,
    }, index=index)
    partial = rows.is_first_hour | rows.is_last_hour
    if shared_partial_hours is not None:
        partial |= index.isin(shared_partial_hours)
    blocked_history = pd.Series(False, index=index)
    offsets = {
        offset
        for _, kind, first, last in feature_catalog(settings, horizon, period)
        if kind in {"history", "baseline"}
        for offset in range(first, last + 1)
    }
    for offset in offsets:
        blocked_history |= partial.shift(horizon - offset, fill_value=False)
    timeline["partial_target"] = partial
    timeline["partial_history"] = blocked_history
    timeline["status"] = "eligible"
    timeline.loc[X.isna().any(axis=1) | seasonal.isna(), "status"] = "warmup"
    if policy == "exclude_partial":
        timeline.loc[blocked_history & timeline.status.eq("eligible"), "status"] = "boundary_history"
        timeline.loc[partial, "status"] = "boundary_target"
    return ForecastingDataset(X, rows.session_start_count.rename("actual"), seasonal, timeline, horizon)
