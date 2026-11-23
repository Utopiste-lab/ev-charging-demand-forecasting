"""Two-stage occurrence and positive-only conditional magnitude estimation."""

from time import perf_counter

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import score
from src.advanced_forecasting.residual_model import fit_model, regressor


def classifier(config, seed):
    params = {k: v for k, v in config.items() if k != "family"}
    if config["family"] == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, **params))
    if config["family"] == "forest":
        return RandomForestClassifier(random_state=seed, n_jobs=4, **params)
    if config["family"] == "boosting":
        return GradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(config["family"])


def fit_occurrence(estimator, rows):
    if hasattr(estimator, "n_features_in_"):
        raise ValueError("Expected a fresh occurrence classifier")
    start = perf_counter()
    assert rows.X.index.equals(rows.y.index)
    binary = rows.y.gt(0).astype(int)
    assert binary.nunique() == 2
    estimator.fit(rows.X, binary)
    elapsed = perf_counter() - start
    if isinstance(estimator, RandomForestClassifier):
        estimator.set_params(n_jobs=1)
    return elapsed


def components(models, inputs):
    estimator, magnitude = models
    assert np.array_equal(estimator.classes_, [0, 1])
    probability = estimator.predict_proba(inputs)[:, 1]
    return probability, magnitude.predict(inputs)


def select_hurdle(payload, candidates, seed):
    train, val = payload.train, payload.validation
    positive = train.y.loc[train.y.gt(0)]
    classifiers, magnitudes = [], []
    for config in candidates["classifiers"]:
        estimator = classifier(config, seed)
        elapsed = fit_occurrence(estimator, train)
        assert np.array_equal(estimator.classes_, [0, 1])
        classifiers.append((config, elapsed, estimator.predict_proba(val.X)[:, 1]))
    for config in candidates["magnitudes"]:
        estimator = regressor(config, seed)
        elapsed = fit_model(estimator, train.X.loc[positive.index], positive)
        magnitudes.append((config, elapsed, estimator.predict(val.X)))
    trials, details = [], []
    for c, c_seconds, probability in classifiers:
        for m, m_seconds, magnitude in magnitudes:
            prediction = probability * magnitude
            details.append((probability, magnitude))
            trials.append({
                "candidate": len(trials) + 1, "parameters": {"classifier": c, "magnitude": m},
                "candidate_fit_seconds": c_seconds + m_seconds,
                "validation_MAE": score(val.y, prediction)["MAE"],
            })
    best = min(trials, key=lambda r: (r["validation_MAE"], r["candidate"]))
    return best, details[best["candidate"] - 1], trials


def refit_hurdle(payload, config, seed):
    final = payload.final_refit
    positive = final.y.loc[final.y.gt(0)]
    occurrence_model = classifier(config["classifier"], seed)
    magnitude_model = regressor(config["magnitude"], seed)
    elapsed = fit_occurrence(occurrence_model, final)
    elapsed += fit_model(magnitude_model, final.X.loc[positive.index], positive)
    return (occurrence_model, magnitude_model), positive, elapsed
