"""Separate chronological selection, calibration and evaluation label sets."""

import pandas as pd

from src.feature_engineering import HOUR
from src.temporal_split import split_dataset


def release_split(dataset, windows, calibration_days):
    val_start, val_stop = windows["validation"]
    start = val_stop - pd.Timedelta(days=calibration_days)
    if not val_start < start < val_stop:
        raise ValueError("Earlier validation and calibration must both be nonempty")
    early_calendar = {"train": windows["train"], "validation": (val_start, start), "test": (start, val_stop)}
    earlier, full = split_dataset(dataset, early_calendar), split_dataset(dataset, windows)
    selected = {}
    for role in ("train", "validation"):
        selected[role] = earlier.index[earlier.partition.eq(role) & earlier.status.eq("eligible")]
    selected["calibration"] = full.index[full.partition.eq("validation") & full.index.to_series().ge(start)
                                         & full.status.eq("eligible")]
    selected["test"] = full.index[full.partition.eq("test") & full.status.eq("eligible")]
    selected["final_fit"] = selected["train"].append(selected["validation"])
    timeline = dataset.timeline
    roles = ("train", "validation", "calibration", "test")
    assert all(len(selected[role]) and selected[role].is_unique for role in roles)
    for one, two in zip(roles, roles[1:]):
        left, right = selected[one], selected[two]
        assert left[-1] < right[0]
        assert timeline.loc[left, "target_end"].max() <= timeline.loc[right, "forecast_issued_at"].min()
    assert timeline.loc[selected["final_fit"], "target_end"].max() <= start - (dataset.horizon - 1) * HOUR
    assert not timeline.loc[selected["final_fit"].append(selected["calibration"]).append(selected["test"]),
                            ["partial_target", "partial_history"]].any().any()
    return selected, start
