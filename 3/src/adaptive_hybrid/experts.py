"""Reuse Experiment 07 model families, chronological selection and cold refits."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.advanced_forecasting.hurdle_model import components, refit_hurdle, select_hurdle
from src.advanced_forecasting.residual_model import refit_residual, select_residual

CORE = ["Seasonal Naive", "Hurdle clipped", "Residual clipped"]
PROBABILITY = "Hurdle probability"


@dataclass(frozen=True)
class ExpertFit:
    hurdle: tuple
    residual: object
    median_distribution: object | None


def select(payload, seasonal, config, seed):
    hb, _, ht = select_hurdle(payload, config["hurdle"], seed)
    rb, _, rt = select_residual(payload, seasonal, config["residual"]["candidates"], seed)
    choices = {"hurdle": hb["parameters"], "residual": rb["parameters"]}
    trials = [dict(row, family=family, selected=row is best)
              for family, rows, best in (("hurdle", ht, hb), ("residual", rt, rb)) for row in rows]
    return choices, trials


def fit(payload, seasonal, choices, seed, median_config=None):
    hurdle, positive, _ = refit_hurdle(payload, choices["hurdle"], seed)
    residual, signed_target, _ = refit_residual(payload, seasonal, choices["residual"], seed)
    assert positive.index.equals(payload.final_refit.y.loc[payload.final_refit.y.gt(0)].index)
    assert signed_target.index.equals(payload.final_refit.y.index)
    np.testing.assert_array_equal(signed_target, payload.final_refit.y - seasonal.loc[signed_target.index])
    model = None
    if median_config is not None:
        model = RandomForestClassifier(**median_config, random_state=seed, n_jobs=4)
        model.fit(payload.final_refit.X.loc[positive.index], positive)
        model.set_params(n_jobs=1)
        assert np.all(model.classes_ >= 1)
    return ExpertFit(hurdle, residual, model), positive.index


def predict(fitted, inputs, seasonal):
    assert inputs.index.equals(seasonal.index)
    p, magnitude = components(fitted.hurdle, inputs)
    correction = fitted.residual.predict(inputs)
    raw_h = p * magnitude
    raw_r = seasonal.to_numpy() + correction
    out = pd.DataFrame({
        "Seasonal Naive": seasonal, "Hurdle": raw_h, "Hurdle clipped": np.maximum(0, raw_h),
        "Residual": raw_r, "Residual clipped": np.maximum(0, raw_r), PROBABILITY: p,
        "conditional_magnitude": magnitude, "predicted_residual": correction,
    }, index=inputs.index)
    if fitted.median_distribution is not None:
        m = fitted.median_distribution
        cdf = m.predict_proba(inputs).cumsum(axis=1)
        np.testing.assert_allclose(cdf[:, -1], 1, atol=1e-12)
        assert (np.diff(cdf, axis=1) >= 0).all()
        positive = p > 0.5
        q = np.maximum(0, 1 - np.divide(0.5, p, out=np.full_like(p, np.inf), where=p > 0))
        position = (cdf < q[:, None]).sum(axis=1)
        row = np.arange(len(p))
        previous = np.where(position, cdf[row, np.maximum(position - 1, 0)], 0)
        below = (1 - p) + p * previous
        reached = (1 - p) + p * cdf[row, position]
        out["Hurdle median"] = np.where(positive, m.classes_[position], 0)
        out["median_cdf_below"] = np.where(positive, below, 0)
        out["median_cdf_at"] = np.where(positive, reached, 1 - p)
        assert (out.median_cdf_below < 0.5 + 1e-12).all()
        assert (out.median_cdf_at >= 0.5 - 1e-12).all()
    assert np.isfinite(out).all().all()
    return out
