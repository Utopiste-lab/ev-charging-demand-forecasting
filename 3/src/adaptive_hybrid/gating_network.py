"""A small softmax mixture trained on out-of-fold forecasts with a smooth MAE objective."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Gate:
    scaler: StandardScaler
    parameter: np.ndarray
    hidden: int
    n: int
    k: int
    convergence: str
    iterations: int

    def weights(self, inputs):
        x = design(self.scaler.transform(inputs))
        return forward(self.parameter, x, self.hidden, self.k)[0]


def design(x):
    return np.column_stack([x, np.ones(len(x))])


def forward(parameter, x, hidden, k):
    if not hidden:
        return softmax(x @ parameter.reshape(x.shape[1], k), axis=1), None, None
    offset = x.shape[1] * hidden
    a = parameter[:offset].reshape(x.shape[1], hidden)
    b = parameter[offset:].reshape(hidden + 1, k)
    activated = np.tanh(x @ a)
    return softmax(design(activated) @ b, axis=1), activated, b


def objective(parameter, x, forecasts, y, hidden, strength, smoothing):
    k = forecasts.shape[1]
    weights, activated, b = forward(parameter, x, hidden, k)
    prediction = (weights * forecasts).sum(axis=1)
    e = prediction - y
    distance = np.sqrt(e * e + smoothing * smoothing)
    logit_derivative = (e / (len(y) * distance))[:, None] * weights * (forecasts - prediction[:, None])
    if hidden:
        grad_b = design(activated).T @ logit_derivative
        grad_a = x.T @ ((logit_derivative @ b[:-1].T) * (1 - activated**2))
        grad = np.concatenate([grad_a.ravel(), grad_b.ravel()])
    else:
        grad = (x.T @ logit_derivative).ravel()
    return distance.mean() + strength * np.dot(parameter, parameter) / 2, grad + strength * parameter


def fit(inputs, forecasts, y, options, settings, seed):
    scaler = StandardScaler().fit(inputs)
    x = design(scaler.transform(inputs))
    hidden, k = options["hidden"], forecasts.shape[1]
    size = x.shape[1] * k if not hidden else x.shape[1] * hidden + (hidden + 1) * k
    rng = np.random.default_rng(seed)
    start = np.zeros(size) if not hidden else rng.normal(0, 0.07, size)
    result = minimize(objective, start, method="L-BFGS-B", jac=True,
                      args=(x, forecasts, y, hidden, options["l2"], settings["smoothing"]),
                      options={"maxiter": settings["max_iterations"], "ftol": 1e-9, "gtol": 1e-6})
    if not result.success and result.status != 1:
        raise RuntimeError(f"Gating optimization failed: {result.message}")
    assert np.isfinite(result.x).all() and result.fun <= objective(start, x, forecasts, y, hidden,
                                                                  options["l2"], settings["smoothing"])[0] + 1e-10
    convergence = ("converged" if result.success else "limit_reached")
    return Gate(scaler, result.x, hidden, x.shape[1], k, convergence, result.nit)


def calibrate(weights, forecasts, y, strength, smoothing):
    logits = np.log(np.maximum(weights, 1e-300))

    def loss(offset):
        mix = softmax(logits + offset, axis=1)
        forecast = (mix * forecasts).sum(axis=1)
        e = forecast - y
        distance = np.sqrt(e * e + smoothing * smoothing)
        tangent = (e / distance)[:, None] * mix * (forecasts - forecast[:, None])
        return distance.mean() + strength * (offset @ offset) / 2, tangent.mean(axis=0) + strength * offset

    result = minimize(loss, np.zeros(weights.shape[1]), method="L-BFGS-B", jac=True,
                      options={"ftol": 1e-10, "gtol": 1e-7})
    if not result.success:
        raise RuntimeError(f"Target pretest calibration failed: {result.message}")
    return result.x


def apply_offset(weights, offset):
    return softmax(np.log(np.maximum(weights, 1e-300)) + offset, axis=1)
