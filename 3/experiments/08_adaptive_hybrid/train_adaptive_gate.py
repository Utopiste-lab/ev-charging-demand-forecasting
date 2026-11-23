"""Select and freeze static, rule and learned gates with a common OOF meta-dataset."""

import hashlib

import numpy as np

from train_static_hybrid import mae, rule_weights, train_static
from src.adaptive_hybrid import gating_network
from src.adaptive_hybrid.experts import CORE
from src.adaptive_hybrid.hybrid_forecaster import ABLATIONS, ADAPTIVE, EQUAL, RULE, STATIC, combine, full_weights, inputs


def learn(project, oof, oof_x, val, val_x, y_val, *, variants=ABLATIONS, seed=None):
    models, trials = {}, []
    opts = project.cfg["gate"]
    if seed is None:
        seed = project.cfg["seed"]
    for variant in variants:
        tx, tp, names = inputs(oof, oof_x, variant)
        vx, vp, _ = inputs(val, val_x, variant)
        candidates = []
        for ordinal, params in enumerate(opts["candidates"], 1):
            gate = gating_network.fit(tx, tp, oof.actual.to_numpy(), params, opts, seed)
            candidates.append((gate, {"family": variant, "candidate": ordinal, "parameters": dict(params, seed=seed),
                                      "validation_MAE": mae(y_val, combine(vp, gate.weights(vx))),
                                      "optimizer_status": gate.convergence, "iterations": gate.iterations}))
        best, info = min(candidates, key=lambda model: (model[1]["validation_MAE"], model[1]["candidate"]))
        models[variant] = (best, info["parameters"])
        trials.extend(dict(row, selected=gate is best) for gate, row in candidates)
    return models, trials


def frozen_sensitivity(project, oof, oof_x, params, seed):
    gx, gp, _ = inputs(oof, oof_x, ADAPTIVE)
    return gating_network.fit(gx, gp, oof.actual.to_numpy(), params, project.cfg["gate"], seed)


def train(project, oof, oof_x, val, val_x, y_val, *, ablate):
    static, rule, simple = train_static(project, oof, val, y_val)
    models, learned = learn(project, oof, oof_x, val, val_x, y_val, variants=ABLATIONS if ablate else [ADAPTIVE])
    best = min(CORE, key=lambda name: mae(y_val, val[name]))
    return static, rule, models, best, simple + learned


def predictions(p, x, static, rule, models, best):
    base = p[CORE].to_numpy()
    weights = {EQUAL: np.full(base.shape, 1 / len(CORE)),
               STATIC: np.broadcast_to(static["weight"], base.shape), RULE: rule_weights(p, rule)}
    for variant, (gate, _) in models.items():
        meta, _, names = inputs(p, x, variant)
        weights[variant] = full_weights(gate.weights(meta), names, len(p))
    forecasts = {name: combine(base, w) for name, w in weights.items()}
    forecasts["Best validation expert"] = p[best].to_numpy()
    return forecasts, weights


def parameter_digest(gate):
    fields = (gate.parameter, gate.scaler.mean_, gate.scaler.scale_)
    return hashlib.sha256(b"".join(np.asarray(p, dtype="<f8").tobytes() for p in fields)).hexdigest()
