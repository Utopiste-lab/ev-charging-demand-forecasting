"""Separate source fit payloads from target inference inputs without fitting models."""

import hashlib
from dataclasses import dataclass

import pandas as pd

from src.feature_engineering import HOUR
from src.forecasting_dataset import build_dataset
from src.temporal_split import PARTITIONS, split_dataset


def timestamp_hash(index):
    return hashlib.sha256("".join(f"{hour.isoformat()}\n" for hour in index).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SupervisedRows:
    X: pd.DataFrame
    y: pd.Series


@dataclass(frozen=True)
class SourceFit:
    site: str
    train: SupervisedRows
    validation: SupervisedRows
    final_refit: SupervisedRows


@dataclass(frozen=True)
class TargetInputs:
    site: str
    X: pd.DataFrame
    seasonal_naive: pd.Series
    timeline: pd.DataFrame


def site_ledger(rows, settings, horizon, period, policy, partial, windows):
    dataset = build_dataset(rows, settings, horizon, period, policy=policy, shared_partial_hours=partial)
    return dataset, split_dataset(dataset, windows)


def eligible_rows(ledger, partition):
    return ledger.loc[ledger.partition.eq(partition) & ledger.status.eq("eligible")]


def source_fit(site, dataset, ledger):
    """Supply only source-site supervised rows to a later model runner."""
    train, val = (eligible_rows(ledger, role).index for role in ("train", "validation"))

    def select(index):
        return SupervisedRows(dataset.X.loc[index].copy(), dataset.y.loc[index].copy())

    return SourceFit(site, select(train), select(val), select(train.append(val)))


def target_inputs(site, dataset, ledger):
    """The inference payload carries predictors and a seasonal reference, with no test labels."""
    timeline = eligible_rows(ledger, "test")
    keys = timeline.index
    return TargetInputs(site, dataset.X.loc[keys].copy(), dataset.seasonal_naive.loc[keys].copy(), timeline.copy())


def validate_release(h, fit, target, source_ledger, target_ledger, windows):
    train, val, test = fit.train.y.index, fit.validation.y.index, target.X.index
    assert all(len(index) and index.is_unique for index in (train, val, test))
    earliest_val = source_ledger.loc[val, "forecast_issued_at"].min()
    earliest_test = target.timeline.forecast_issued_at.min()
    assert source_ledger.loc[train, "target_end"].max() <= earliest_val
    refit = fit.final_refit.y.index
    assert refit.equals(train.append(val)) and refit.equals(fit.final_refit.X.index)
    assert source_ledger.loc[refit, "target_end"].max() <= earliest_test
    assert train.intersection(val).empty and refit.intersection(test).empty
    for partition, index in zip(PARTITIONS, (train, val, test)):
        start, end = windows[partition]
        assert ((index >= start) & (index < end)).all()
    for part, following in (("train", "validation"), ("validation", "test")):
        release_at = windows[following][0] - (h - 1) * HOUR
        block = source_ledger.loc[source_ledger.partition.eq(part)]
        removed = block.loc[block.status.eq("label_release_purge")]
        assert len(removed) == h - 1 and removed.target_end.gt(release_at).all()
        assert block.loc[block.status.eq("eligible"), "target_end"].le(release_at).all()
    assert not target_ledger.loc[test, ["partial_target", "partial_history"]].any().any()
    assert not source_ledger.loc[refit, ["partial_target", "partial_history"]].any().any()


def summarize_direction(horizon, direction, fit, source_ledger, target_ledger, windows, overlap_start, overlap_end):
    for part in (*PARTITIONS, "final_refit"):
        is_test = part == "test"
        ledger = target_ledger if is_test else source_ledger
        partitions = ["train", "validation"] if part == "final_refit" else [part]
        candidates = ledger.loc[ledger.partition.isin(partitions)]
        usable = candidates.loc[candidates.status.eq("eligible")]
        counts = candidates.status.value_counts()
        if part == "final_refit":
            assert usable.index.equals(fit.final_refit.X.index)
        lows, highs = zip(*(windows[key] for key in partitions))
        yield {
            "direction": direction, "horizon_hours": horizon, "partition": part,
            "site": direction.split("->")[int(is_test)], "site_role": "target" if is_test else "source",
            "calendar_start": min(lows), "calendar_end_exclusive": max(highs),
            "common_history_start": overlap_start, "common_history_end_inclusive": overlap_end,
            "candidate_target_hours": len(candidates), "eligible_n": len(usable),
            "warmup": int(counts.get("warmup", 0)),
            "excluded_partial_target": int(counts.get("boundary_target", 0)),
            "excluded_partial_history": int(counts.get("boundary_history", 0)),
            "excluded_label_release": int(counts.get("label_release_purge", 0)),
            "target_start": usable.index.min(), "target_end_inclusive": usable.index.max(),
            "first_issue": usable.forecast_issued_at.min(), "last_issue": usable.forecast_issued_at.max(),
            "last_label_available_at": usable.target_end.max(),
            "target_timestamps_sha256": timestamp_hash(usable.index),
        }
