"""Expose the three existing Experiment 07/08 rolling calendars for Experiment 09."""

from stage3_context import ROOT, config

import pandas as pd


def windows():
    rolling = config("config/adaptive_hybrid.yaml")["rolling"]
    days = {k: pd.Timedelta(days=rolling[k + "_days"]) for k in ("train", "validation", "test")}
    for index, date in enumerate(rolling["first_test_starts"], 1):
        start = pd.Timestamp(date)
        pre = start - days["validation"]
        yield f"R{index}", {"train": (pre - days["train"], pre), "validation": (pre, start),
                            "test": (start, start + days["test"])}
