"""Fit simple mixtures on OOF data, then select candidates on official validation."""

import numpy as np

from src.adaptive_hybrid.experts import CORE, PROBABILITY
from src.adaptive_hybrid.hybrid_forecaster import combine, fit_static, simplex_grid


def mae(y, p):
    return np.abs(np.asarray(p) - np.asarray(y)).mean()


def train_static(project, oof, val, y_val):
    opts = project.cfg["static"]
    grid = simplex_grid(opts["grid_resolution"])
    x, y = oof[CORE].to_numpy(), oof.actual.to_numpy()
    v = val[CORE].to_numpy()
    static = []
    for ordinal, reg in enumerate(opts["regularization"], 1):
        weight = fit_static(x, y, grid, reg)
        static.append({"candidate": ordinal, "parameters": {"regularization": reg, "weight": weight.tolist()},
                       "validation_MAE": mae(y_val, combine(v, weight))})
    best_static = min(static, key=lambda row: (row["validation_MAE"], row["candidate"]))
    fallback = np.asarray(best_static["parameters"]["weight"])
    rule = []
    for ordinal, threshold in enumerate(project.cfg["rule"]["probability_thresholds"], 1):
        low = oof[PROBABILITY].to_numpy() <= threshold
        pairs = []
        for mask in (low, ~low):
            fitted = fit_static(x[mask], y[mask], grid, best_static["parameters"]["regularization"]) \
                if mask.sum() >= project.cfg["rule"]["regime_min_oof_rows"] else fallback
            pairs.append(fitted.tolist())
        weights = rule_weights(val, {"threshold": threshold, "regime_weights": pairs})
        rule.append({"candidate": ordinal, "parameters": {"threshold": threshold, "regime_weights": pairs,
                                                          "oof_low_n": int(low.sum()), "oof_high_n": int((~low).sum())},
                     "validation_MAE": mae(y_val, combine(v, weights))})
    best_rule = min(rule, key=lambda row: (row["validation_MAE"], row["candidate"]))
    output = []
    for family, rows, best in (("Static optimized", static, best_static), ("Rule gate", rule, best_rule)):
        output.extend(dict(r, family=family, selected=r is best) for r in rows)
    return best_static["parameters"], best_rule["parameters"], output


def rule_weights(pred, choices):
    return np.asarray(choices["regime_weights"])[(pred[PROBABILITY].to_numpy() > choices["threshold"]).astype(int)]
