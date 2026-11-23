"""First-prefix selection and expanding chronological expert cross-fitting."""

from dataclasses import replace

import numpy as np
import pandas as pd

from src.adaptive_hybrid import experts


def rows_at(rows, keys):
    return replace(rows, X=rows.X.loc[keys], y=rows.y.loc[keys])


def build_oof(payload, dataset, ledger, config, advanced, hash_keys, block_count, prefix=None):
    train = payload.train
    keys = train.X.index
    season = dataset.seasonal_naive
    first = int(len(keys) * config["oof_warmup_fraction"])
    prefix_validation = keys[first - config["oof_inner_validation_hours"]:first]
    earliest_oof_issue = ledger.loc[keys[first], "forecast_issued_at"]
    prefix_validation = prefix_validation[ledger.loc[prefix_validation, "target_end"].le(earliest_oof_issue)]
    earlier = keys[:first - config["oof_inner_validation_hours"]]
    inner_issue = ledger.loc[prefix_validation[0], "forecast_issued_at"]
    earlier = earlier[ledger.loc[earlier, "target_end"].le(inner_issue)]
    assert len(earlier) and len(prefix_validation)
    assert earlier.intersection(prefix_validation).empty
    if prefix is None:
        inner = replace(payload, train=rows_at(train, earlier), validation=rows_at(train, prefix_validation))
        prefix = experts.select(inner, season, advanced, config["seed"])
    choices, _ = prefix
    records, features, manifests = [], [], []
    for block, positions in enumerate(np.array_split(np.arange(first, len(keys)), block_count), 1):
        forecast_keys = keys[positions]
        issue = ledger.loc[forecast_keys[0], "forecast_issued_at"]
        prior = keys[:positions[0]]
        fit_keys = prior[ledger.loc[prior, "target_end"].le(issue)]
        assert fit_keys.intersection(forecast_keys).empty and prefix_validation.difference(fit_keys).empty
        fitted, positive = experts.fit(replace(payload, final_refit=rows_at(train, fit_keys)),
                                       season, choices, config["seed"])
        x = train.X.loc[forecast_keys]
        out = experts.predict(fitted, x, season.loc[forecast_keys])
        out["forecast_issued_at"] = ledger.loc[forecast_keys, "forecast_issued_at"]
        out["actual"] = train.y.loc[forecast_keys]
        out["block"] = block
        records.append(out)
        features.append(x)
        manifests.append({
            "oof_blocks": block_count, "block": block, "fit_n": len(fit_keys),
            "fit_start": fit_keys[0], "fit_end": fit_keys[-1], "fit_sha256": hash_keys(fit_keys),
            "positive_fit_n": len(positive), "positive_fit_sha256": hash_keys(positive),
            "purged": len(prior) - len(fit_keys), "prediction_n": len(forecast_keys),
            "prediction_start": forecast_keys[0], "prediction_end": forecast_keys[-1],
            "prediction_sha256": hash_keys(forecast_keys), "first_issue": issue,
            "last_fit_label_available": ledger.loc[fit_keys, "target_end"].max(),
            "prefix_train_n": len(earlier), "prefix_train_sha256": hash_keys(earlier),
            "prefix_validation_n": len(prefix_validation), "prefix_validation_sha256": hash_keys(prefix_validation),
            "last_prefix_label_available": ledger.loc[prefix_validation, "target_end"].max(),
        })
    meta, x = pd.concat(records), pd.concat(features)
    assert meta.index.equals(keys[first:]) and x.index.equals(meta.index)
    assert not meta.index.has_duplicates
    return meta, x, manifests, prefix
