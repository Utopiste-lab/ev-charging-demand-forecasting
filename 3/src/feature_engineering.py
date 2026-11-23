"""Causal feature definitions on a regular hourly source-clock series."""

import pandas as pd


HOUR = pd.Timedelta(hours=1)


def feature_catalog(settings, horizon, period):
    """Offsets index interval starts relative to the last completed hour t."""
    hours = settings["hours"]
    days = settings["weekdays"]
    records = [
        (f"hour_{i}", "calendar", horizon, horizon)
        for i in range(hours["first_indicator"], hours["last_indicator"] + 1)
    ] + [
        (f"weekday_{i}", "calendar", horizon, horizon)
        for i in range(days["first_indicator"], days["last_indicator"] + 1)
    ]
    records.append(("weekend", "calendar", horizon, horizon))
    records.extend((f"lag_{k}", "history", 1 - k, 1 - k) for k in settings["lag_hours"])
    rolling = settings["rolling_mean"]
    last = 1 - rolling["shift_hours"]
    records.append((f"rolling_mean_{rolling['window_hours']}", "history",
                    last - rolling["window_hours"] + 1, last))
    if len(records) != settings["feature_count"]:
        raise ValueError("Feature count differs from the historical feature contract")
    return records + [
        ("seasonal_naive", "baseline", horizon - period, horizon - period),
        ("actual", "label", horizon, horizon),
    ]


def build_features(counts, settings, horizon, period):
    """Return candidate predictors and the independent seasonal reference."""
    if not 1 <= horizon <= period:
        raise ValueError("Seasonal reference must be available at the forecast origin")
    index = counts.index
    features = pd.DataFrame(index=index)
    for name, kind, first, last in feature_catalog(settings, horizon, period):
        if kind in {"baseline", "label"}:
            continue
        if kind == "calendar":
            if name == "weekend":
                values = index.dayofweek.isin(settings["weekend_days"])
            elif name.startswith("weekday_"):
                values = index.dayofweek == int(name.split("_")[1])
            else:
                values = index.hour == int(name.split("_")[1])
            features[name] = values.astype(float)
            continue
        if last > 0:
            raise ValueError(f"Feature accesses an unfinished hour: {name}")
        values = counts.shift(horizon - last)
        if first < last:
            values = values.rolling(last - first + 1,
                                    min_periods=settings["rolling_mean"]["min_periods"]).mean()
        features[name] = values
    return features, counts.shift(period).rename("seasonal_naive")
