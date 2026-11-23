# Stage 3 publication validation

Validated on 2026-11-23 against repository base `1e9ebf9aa22e386eb9e5eefced5b75f10c8aaaed` and the supplied local Stage 3 archive.

| Check | Outcome |
|---|---|
| Published artifact manifest | 254 Stage 3 files checked by byte size and SHA-256 |
| Saved official one-hour predictions | Six variants, validation/test MAE and RMSE recomputed; 3,383 test target keys per variant |
| Feature and split checks | 32 plans passed across both sites, four horizons, two boundary policies and full/common windows; includes future/target perturbations and label-release checks |
| Official one-hour training replay | Six tables passed at absolute/relative tolerance 2e-11; maximum absolute numerical difference 2.8421709430404007e-14; five of five regenerated figure hashes identical |
| Stage 2 historical training replay | Flat repository inputs reproduced the original 3,384-hour Seasonal Naive and Ridge MAE/RMSE |
| Original hourly data | Byte hash retained; Git attributes prevent Windows line-ending conversion |

One-hour retraining was checked with `python -B -X utf8 experiments/09_probabilistic_forecasting/replay_legacy_point.py 03` from `3/`. The wrapper runs the original trainer in temporary storage with the documented numerical tolerance and reports figure hashes separately. It does not replace official outputs. For read-only review use `verify_release.py` instead.

The validation environment used Python 3.12.3, NumPy 2.4.4, pandas 3.0.3, scikit-learn 1.8.0 and Matplotlib 3.10.8. The original research dependency file pins pandas 3.0.2; this migration validation used the available 3.0.3 installation and does not claim a newly installed exact-pins environment.

Full Experiments 04–09 retraining, full probability metric reconstruction, raw archive audits and Stage 4 validation were not performed for this publication. Their original research evidence remains in reports, with omitted inputs identified in the external artifact catalogue. The research completion checklist and original frozen manifests retain their historical scope.
