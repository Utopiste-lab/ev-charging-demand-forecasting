"""Seasonal residual regression with signed corrections."""

from time import perf_counter

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import score


def regressor(config, seed):
    params = {k: v for k, v in config.items() if k != "family"}
    if config["family"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(solver="svd", **params))
    if config["family"] == "forest":
        return RandomForestRegressor(random_state=seed, n_jobs=4, **params)
    if config["family"] == "boosting":
        return GradientBoostingRegressor(random_state=seed, loss="squared_error", **params)
    raise ValueError(config["family"])


def fit_model(model, X, y):
    if hasattr(model, "n_features_in_"):
        raise ValueError("Expected a fresh regressor")
    assert X.index.equals(y.index) and len(y)
    start = perf_counter()
    model.fit(X, y)
    elapsed = perf_counter() - start
    if isinstance(model, RandomForestRegressor):
        model.set_params(n_jobs=1)
    return elapsed


def seasonal_residual(rows, seasonal):
    reference = seasonal.loc[rows.y.index]
    assert reference.index.equals(rows.y.index) and np.isfinite(reference).all()
    return rows.y.sub(reference).rename("residual_target")


def select_residual(payload, seasonal, candidates, seed):
    train, val = payload.train, payload.validation
    correction_target = seasonal_residual(train, seasonal)
    baseline = seasonal.loc[val.y.index].to_numpy()
    trials, predictions = [], []
    for number, config in enumerate(candidates, 1):
        estimator = regressor(config, seed)
        elapsed = fit_model(estimator, train.X, correction_target)
        correction = estimator.predict(val.X)
        predictions.append(correction)
        trials.append({"candidate": number, "parameters": config, "candidate_fit_seconds": elapsed,
                       "validation_MAE": score(val.y, baseline + correction)["MAE"]})
    best = min(trials, key=lambda r: (r["validation_MAE"], r["candidate"]))
    return best, predictions[best["candidate"] - 1], trials


def refit_residual(payload, seasonal, config, seed):
    final = payload.final_refit
    values = seasonal_residual(final, seasonal)
    estimator = regressor(config, seed)
    elapsed = fit_model(estimator, final.X, values)
    return estimator, values, elapsed
