"""Read and protect prior experiments. All forecasting definitions come from Stage 3."""

from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/05_cross_site_audit")]

from validate_cross_site_splits import site_ledger, source_fit, target_inputs, timestamp_hash, validate_release
from src.feature_engineering import HOUR, feature_catalog
from src.forecasting_dataset import load_hourly
from src.temporal_split import calendar_windows, common_view

TABLE = Path("results/tables/adaptive_hybrid")
METRICS = Path("results/metrics/adaptive_hybrid")
FIGURES = Path("results/figures/adaptive_hybrid")
OLD_TABLE = Path("results/tables/advanced_forecasting")
OLD_METRICS = Path("results/metrics/advanced_forecasting")
CASES = [
    ("Caltech full", "Caltech", "Caltech"),
    ("JPL matched", "JPL", "JPL"),
    ("Caltech matched", "Caltech", "Caltech"),
    ("Caltech->JPL", "Caltech", "JPL"),
    ("JPL->Caltech", "JPL", "Caltech"),
]


def read(path, **options):
    return pd.read_csv(path, float_precision="round_trip", **options)


def config(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protect():
    new_parts = {"adaptive_hybrid", "08_adaptive_hybrid", "__pycache__"}
    names = {"adaptive_hybrid.yaml", "08_adaptive_hybrid.md", "07_robustness_supplement.md",
             "phase9_adaptive_hybrid.log"}
    return pd.DataFrame({"path": p.relative_to(ROOT.parent).as_posix(), "sha256": digest(p)}
                        for base in (ROOT.with_name("2"), ROOT) for p in sorted(base.rglob("*"))
                        if p.is_file() and not new_parts.intersection(p.relative_to(base).parts)
                        and not any(part.startswith(".exp08-") for part in p.relative_to(base).parts)
                        and p.name not in names)


def verify_protection(protected):
    for row in protected.itertuples(index=False):
        assert digest(ROOT.parent / row.path) == row.sha256, row.path


class Project:
    def __init__(self):
        self.cfg = config("config/adaptive_hybrid.yaml")
        self.advanced = config(self.cfg["advanced_config"])
        old_transfer = config(self.advanced["cross_site_training"])
        self.one_hour = config(old_transfer["unified_config"])
        cross = config(old_transfer["audit_config"])
        framework = config(cross["framework"])
        self.baseline = config(framework["legacy_config"])
        self.horizons = cross["horizons_hours"]
        self.settings = self.baseline["features"]
        self.period = self.baseline["models"]["seasonal_naive"]["period_hours"]
        self.policy = cross["boundary_policy"]
        self.windows = calendar_windows(self.baseline["split"])
        self.full, _ = load_hourly(ROOT / framework["data_view"], ROOT / framework["provenance"])
        self.common, start, end, self.partial = common_view(self.full)
        assert (start, end) == tuple(pd.Timestamp(cross["expected_common"][k]) for k in ("start", "end"))
        test = cross["expected_test"]
        self.keys = pd.date_range(test["start"], test["end"], freq=HOUR)
        assert len(self.keys) == test["hours"]
        self.previous_splits = read(ROOT / OLD_TABLE / "advanced_forecasting_split_summary.csv")
        self.previous_scores = read(ROOT / OLD_METRICS / "advanced_forecasting_comparison.csv")
        self.previous_forecasts = read(ROOT / OLD_TABLE / "advanced_forecasting_predictions.csv", parse_dates=["target_hour"])

    def site_rows(self, case, site):
        frame = self.full if case == "Caltech full" else self.common
        return frame.loc[frame.site.eq(site)]

    def dataset(self, case, site, h, windows, rows=None):
        if rows is None:
            rows = self.site_rows(case, site)
        partial = None if case == "Caltech full" else self.partial
        return site_ledger(rows, self.settings, h, self.period, self.policy, partial, windows)

    def payload(self, case, source, target, h, *, windows=None, poison=False):
        official = windows is None
        windows = self.windows if official else windows
        s, sl = self.dataset(case, source, h, windows)
        t, tl = (s, sl) if source == target else self.dataset(case, target, h, windows)
        predict = target_inputs(target, t, tl)
        keys = predict.X.index
        truth = t.y.loc[keys].copy()
        if poison:
            s = replace(s, y=s.y.mask(sl.partition.eq("test"), -97001))
            t = replace(t, y=t.y.mask(tl.partition.eq("test"), -97002))
        fit = source_fit(source, s, sl)
        assert not hasattr(predict, "y") and len(predict.X.columns) == 34
        assert list(fit.train.X) == list(predict.X)
        validate_release(h, fit, predict, sl, tl, windows)
        assert (predict.timeline.forecast_issued_at == keys - (h - 1) * HOUR).all()
        assert all(last <= 0 for _, kind, _, last in feature_catalog(self.settings, h, self.period)
                   if kind in {"history", "baseline"})
        assert not fit.final_refit.X.isna().any().any() and not predict.X.isna().any().any()
        serial = []
        for role, index, ledger in [("train", fit.train.y.index, sl), ("validation", fit.validation.y.index, sl),
                                   ("final_refit", fit.final_refit.y.index, sl), ("test", keys, tl)]:
            if official:
                prev = self.previous_splits
                expected = prev.loc[prev.case.eq(case) & prev.horizon_hours.eq(h) & prev.partition.eq(role)].iloc[0]
                assert len(index) == expected.n and timestamp_hash(index) == expected.timestamp_sha256
                assert keys.equals(self.keys)
            timing = ledger.loc[index]
            serial.append({"case": case, "horizon_hours": h, "partition": role, "n": len(index),
                           "timestamp_sha256": timestamp_hash(index), "start": index[0], "end": index[-1],
                           "first_issue": timing.forecast_issued_at.min(),
                           "last_label_available": timing.target_end.max()})
        return fit, s, sl, predict, truth, serial

    def historical_perturbation(self, case, target, h, windows, predict):
        changed = self.site_rows(case, target).copy()
        change = predict.X.index[0]
        changed.loc[changed.timestamp.eq(change), "session_start_count"] += 10000
        d, ledger = self.dataset(case, target, h, windows, changed)
        perturbed = target_inputs(target, d, ledger)
        before_release = predict.timeline.forecast_issued_at.lt(change + HOUR)
        assert before_release.any()
        pd.testing.assert_frame_equal(predict.X.loc[before_release], perturbed.X.loc[before_release])
        pd.testing.assert_series_equal(predict.seasonal_naive.loc[before_release], perturbed.seasonal_naive.loc[before_release])
        assert not predict.X.loc[~before_release].equals(perturbed.X.loc[~before_release])
        return perturbed, before_release


def publish(frame, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, na_rep="NA")


if __name__ == "__main__":
    import subprocess

    subprocess.run([sys.executable, "-X", "utf8", "-B",
                    str(ROOT / "experiments/07_advanced_forecasting/verify_experiment.py")], check=True)
    project = Project()
    for case, source, target in CASES:
        for h in project.horizons:
            project.payload(case, source, target, h)
    print("Experiment 07 saved results, official sample hashes, and clock alignment passed.")
