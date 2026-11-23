"""Point scores and paired moving-block bootstrap on a regular hourly evaluation window."""

import numpy as np
import pandas as pd

from src.advanced_forecasting.evaluation import demand_metrics


def metrics(y, p):
    return demand_metrics(np.asarray(y), np.asarray(p))


def bootstrap(y, forecast, baseline, lengths, n_bootstrap, seed):
    errors = np.column_stack([forecast - y, baseline - y])
    losses = np.column_stack([np.abs(errors), errors**2])
    n = len(y)
    full = losses.mean(axis=0)
    for length in lengths:
        assert 1 <= length <= n
        # Linear moving blocks, with identical sampled indices for both model losses.
        # Cumulative sums avoid materializing n_bootstrap × n × 4 arrays.
        cumulative = np.vstack([np.zeros((1, 4)), losses.cumsum(axis=0)])
        rng = np.random.default_rng(seed + length)
        whole, tail = divmod(n, length)
        starts = rng.integers(0, n - length + 1, size=(n_bootstrap, whole + bool(tail)))
        first = starts[:, :whole]
        total = (cumulative[first + length] - cumulative[first]).sum(axis=1)
        if tail:
            last = starts[:, -1]
            total += cumulative[last + tail] - cumulative[last]
        means = total / n
        for name, estimate, stat in (
            ("MAE", full[0] - full[1], means[:, 0] - means[:, 1]),
            ("RMSE", np.sqrt(full[2]) - np.sqrt(full[3]), np.sqrt(means[:, 2]) - np.sqrt(means[:, 3])),
        ):
            low, high = np.quantile(stat, [0.025, 0.975])
            yield {"metric": name, "block_hours": length, "replicates": n_bootstrap, "n": n,
                   "difference": estimate, "bootstrap_mean_difference": stat.mean(),
                   "ci_low": low, "ci_high": high, "improvement_fraction": (stat < 0).mean()}


def score_groups(frame, group_keys):
    models = (c for c in frame.columns if c.startswith("model:"))
    overall, subgroup = [], []
    groups = list(frame.groupby(group_keys, sort=False))
    for model in models:
        for key, rows in groups:
            rows = rows.loc[rows[model].notna()]
            if rows.empty:
                continue
            y, p = rows.actual, rows[model]
            meta = dict(zip(group_keys, key), model=model.removeprefix("model:"))
            overall.append({**meta, "n": len(y), **metrics(y, p)})
            for label, mask in [("zero", y.eq(0)), ("positive", y.gt(0))]:
                if mask.any():
                    subgroup.append({**meta, "regime": label, "n": mask.sum(), "fraction": mask.mean(),
                                     **metrics(y[mask], p[mask])})
    return pd.DataFrame(overall), pd.DataFrame(subgroup)
