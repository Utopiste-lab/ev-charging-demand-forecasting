"""Search only on the earlier validation slice, followed by fresh pre-calibration fitting."""

import json
import numpy as np
import pandas as pd

from src.probabilistic_forecasting.quantile_models import QUANTILES, estimator, pinball


def train(dataset, keys, cfg):
    x, y = dataset.X, dataset.y
    train_x, train_y = x.loc[keys["train"]], y.loc[keys["train"]]
    val_x, val_y = x.loc[keys["validation"]], y.loc[keys["validation"]]
    refit_x, refit_y = x.loc[keys["final_fit"]], y.loc[keys["final_fit"]]
    models, trials, heldout = {}, [], []
    for label in (*QUANTILES, "Point GB pre-calibration"):
        group = []
        for i, options in enumerate(cfg["grid"], 1):
            candidate = estimator(label, options, cfg["seed"]).fit(train_x, train_y)
            p = candidate.predict(val_x)
            loss = pinball(val_y, p, QUANTILES[label]) if label in QUANTILES else np.abs(p - val_y)
            candidate_row = {"model": label, "candidate": i, "objective": "pinball" if label in QUANTILES else "MAE",
                             "validation_loss": np.mean(loss), "parameters": json.dumps(options, sort_keys=True)}
            heldout.append(pd.DataFrame({"model": label, "candidate": i, "target_hour": val_x.index,
                                         "actual": val_y.to_numpy(), "prediction_raw": p}))
            group.append(candidate_row)
        best = min(group, key=lambda trial: (trial["validation_loss"], trial["candidate"]))
        options = cfg["grid"][best["candidate"] - 1]
        models[label] = estimator(label, options, cfg["seed"]).fit(refit_x, refit_y)
        trials.extend({**entry, "selected": entry is best} for entry in group)
    return models, trials, pd.concat(heldout, ignore_index=True)
