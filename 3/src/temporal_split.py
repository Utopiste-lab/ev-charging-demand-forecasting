"""Chronological target windows and label-release guards for rolling forecasts."""

import pandas as pd

from src.feature_engineering import HOUR


PARTITIONS = ("train", "validation", "test")


def calendar_windows(legacy_split):
    windows = {
        name: (pd.Timestamp(legacy_split[name]["start"]), pd.Timestamp(legacy_split[name]["end"]) + HOUR)
        for name in PARTITIONS
    }
    if any(start >= stop for start, stop in windows.values()):
        raise ValueError("Empty or reversed calendar window")
    if any(windows[one][1] != windows[two][0] for one, two in zip(PARTITIONS, PARTITIONS[1:])):
        raise ValueError("Target windows must be adjacent and chronological")
    return windows


def split_dataset(dataset, windows):
    """Allocate by target. Purge unreleased fit/selection labels at each next issue."""
    ledger = dataset.timeline.copy()
    ledger["partition"] = "outside_calendar"
    index = ledger.index
    for name, (start, stop) in windows.items():
        ledger.loc[(index >= start) & (index < stop), "partition"] = name
    ledger.loc[ledger.partition.eq("outside_calendar"), "status"] = "outside_calendar"
    for name, following in zip(PARTITIONS, PARTITIONS[1:]):
        first_scheduled_issue = windows[following][0] - (dataset.horizon - 1) * HOUR
        purge = ledger.partition.eq(name) & ledger.target_end.gt(first_scheduled_issue)
        ledger.loc[purge & ledger.status.eq("eligible"), "status"] = "label_release_purge"
    return ledger


def common_view(hourly):
    """Restrict both sites' histories and targets to their observed calendar overlap."""
    ends = hourly.groupby("site").timestamp.agg(["min", "max"])
    first, last = ends["min"].max(), ends["max"].min()
    if first > last:
        raise ValueError("Sites have no common hourly range")
    common = hourly.loc[hourly.timestamp.between(first, last)]
    shared_partial = pd.DatetimeIndex(common.loc[common.is_first_hour | common.is_last_hour,
                                                 "timestamp"].unique())
    return common, first, last, shared_partial
