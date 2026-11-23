"""Regression metrics reused from Stage 3 and fixed-threshold hurdle diagnostics."""

import numpy as np
from sklearn.metrics import (auc, average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from src.evaluation import score


def demand_metrics(actual, prediction):
    return {**score(actual, prediction),
            "bias": float(np.subtract(prediction, actual).mean()),
            "negative_predictions": int(np.count_nonzero(np.asarray(prediction) < 0))}


def occurrence_metrics(actual, probability, threshold):
    positive = np.asarray(actual) > 0
    p = np.asarray(probability)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert positive.any() and (~positive).any()
    decision = p >= threshold
    precision, recall, _ = precision_recall_curve(positive, p)
    return {
        "threshold": threshold, "n": len(p), "n_positive": int(positive.sum()),
        "precision": precision_score(positive, decision, zero_division=0),
        "recall": recall_score(positive, decision, zero_division=0),
        "F1": f1_score(positive, decision, zero_division=0),
        "ROC_AUC": roc_auc_score(positive, p), "PR_AUC": auc(recall, precision),
        "average_precision": average_precision_score(positive, p), "Brier": brier_score_loss(positive, p),
        "mean_probability": p.mean(), "actual_positive_rate": positive.mean(),
    }
