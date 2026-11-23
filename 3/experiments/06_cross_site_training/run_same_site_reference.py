"""Fit fresh matched controls without using any estimator or winner from a transfer run."""

from train_transfer_models import train_models


def train_reference(payload, settings, ridge, meta, first_issue, label_release):
    assert meta["scenario"] == "same_site" and meta["source_site"] == meta["target_site"] == payload.site
    return train_models(payload, settings, ridge, meta, first_issue, label_release)
