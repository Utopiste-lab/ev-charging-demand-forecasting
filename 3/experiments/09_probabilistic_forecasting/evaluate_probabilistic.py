"""Summarize probabilistic, point and regime performance using saved forecast columns."""

import numpy as np
import pandas as pd

from audit_stage3_baselines import point_summary
from src.probabilistic_forecasting.evaluation import interval_values, paired_blocks, summarize_interval
from src.probabilistic_forecasting.quantile_models import QUANTILES, pinball

KEY = ["window", "case", "horizon_hours"]
INTERVALS = {"raw": ("P10", "P90"), "CQR": ("CQR_lower", "CQR_upper")}
POINTS = ["Constant Zero", "Seasonal Naive", "Point GB pre-calibration raw", "Point GB pre-calibration", "P50"]


def evaluate(saved, cfg):
    intervals, quantiles, points, bootstrap, crossing = [], [], [], [], []
    nominal = cfg["coverage_nominal"]
    for key, data in saved.groupby(KEY + ["partition"], sort=False):
        meta = dict(zip(KEY + ["partition"], key))
        y = data.actual.to_numpy()
        crossing.append({**meta, "n": len(data), "crossing_raw_n": data.crossing_raw.sum(),
                         "crossing_raw_rate": data.crossing_raw.mean(), "crossing_after_n": data.crossing_after.sum(),
                         "negative_raw_rate": data.negative_raw.mean(), "cqr_empty_repairs": data.CQR_empty_repair.sum()})
        for label, (lo, hi) in INTERVALS.items():
            for regime, mask in (("all", np.ones(len(y), dtype=bool)), ("zero", y == 0), ("positive", y > 0)):
                scores = summarize_interval(y[mask], data[lo].to_numpy()[mask], data[hi].to_numpy()[mask], nominal)
                intervals.append({**meta, "method": label, "regime": regime, "n": mask.sum(), "fraction": mask.mean(), **scores})
        for name, q in QUANTILES.items():
            pred = data[name].to_numpy()
            raw = data[f"{name}_raw"].to_numpy()
            quantiles.append({**meta, "quantile": name, "q": q, "n": len(y),
                              "pinball": pinball(y, pred, q).mean(), "pinball_raw": pinball(y, raw, q).mean(),
                              "empirical_strict_below": (y < pred).mean(), "empirical_at_or_below": (y <= pred).mean()})
        for m in POINTS:
            points.append({**meta, "model": m, "n": len(y), **point_summary(y, data[m].to_numpy())})
        if key[-1] == "test":
            ref = interval_values(y, data.P10, data.P90, nominal)
            cqr = interval_values(y, data.CQR_lower, data.CQR_upper, nominal)
            bootstrap.extend({**meta, **r} for r in paired_blocks(ref, cqr, cfg["bootstrap"], cfg["seed"]))
    return {"interval_metrics": pd.DataFrame(intervals), "quantile_metrics": pd.DataFrame(quantiles),
            "point_metrics": pd.DataFrame(points), "interval_block_bootstrap": pd.DataFrame(bootstrap),
            "quantile_constraint_summary": pd.DataFrame(crossing)}
