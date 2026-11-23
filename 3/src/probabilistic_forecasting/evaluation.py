"""Per-observation interval quantities and paired moving-block bootstrap."""

import numpy as np


def interval_values(y, lower, upper, nominal):
    y, lower, upper = map(np.asarray, (y, lower, upper))
    width = upper - lower
    assert np.all(width >= 0)
    below, above = np.maximum(lower - y, 0), np.maximum(y - upper, 0)
    covered = (below == 0) & (above == 0)
    return covered, width, width + 2 / (1 - nominal) * (below + above)


def summarize_interval(y, lower, upper, nominal):
    if not len(y):
        return dict(coverage=np.nan, width=np.nan, interval_score=np.nan, coverage_minus_nominal=np.nan,
                    above_upper=np.nan, below_lower=np.nan)
    hit, width, score = interval_values(y, lower, upper, nominal)
    return dict(coverage=hit.mean(), width=width.mean(), interval_score=score.mean(),
                coverage_minus_nominal=hit.mean() - nominal, above_upper=(y > upper).mean(), below_lower=(y < lower).mean())


def paired_blocks(base, adjusted, opts, seed):
    names = ("coverage", "width", "interval_score")
    data = np.column_stack([*base, *adjusted]).astype(float)
    denom = len(data)
    for length in opts["block_hours"]:
        full, tail = divmod(denom, length)
        n = opts["replicates"]
        rng = np.random.default_rng(seed + length)
        starts = rng.integers(0, denom - length + 1, size=(n, full + bool(tail)))
        sums = np.vstack([np.zeros((1, len(names)*2)), data.cumsum(axis=0)])
        left = starts[:, :full]
        totals = (sums[left + length] - sums[left]).sum(axis=1)
        if tail:
            last = starts[:, -1]
            totals += sums[last + tail] - sums[last]
        mean = totals / denom
        original = data.mean(axis=0)
        for i, metric in enumerate(names):
            for stat, observed, sample in (("raw", original[i], mean[:, i]),
                                           ("CQR", original[i+3], mean[:, i+3]),
                                           ("CQR minus raw", original[i+3] - original[i], mean[:, i+3] - mean[:, i])):
                lo, hi = np.quantile(sample, [.025, .975])
                yield {"metric": metric, "statistic": stat, "block_hours": length, "replicates": n,
                       "n": denom, "estimate": observed, "bootstrap_mean": sample.mean(), "ci_low": lo, "ci_high": hi}
