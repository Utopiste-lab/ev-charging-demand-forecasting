# Caltech One-Hour Unified Model Comparison

**Seasonal Naive** had the lowest test MAE and **Gradient Boosting Clipped** had the lowest test RMSE on the official boundary-aware test set. The six variants share all 3,383 target timestamps from 2020-07-15 00:00:00 to 2020-12-02 22:00:00 in the source clock.

## Selection and refit

Eight prespecified candidates per learned model were fitted on 15,790 training rows and ranked solely by raw MAE on 3,385 chronological validation rows. The earliest configured candidate resolves exact ties. Ridge includes alpha 1,000 in its search. The existing RF and GB candidate lists were reused.

After selection, every learned model was reinitialized and fitted on the same **19,175** ordered target timestamps, from 2018-05-08 01:00:00 through 2020-07-14 23:00:00. Ridge's StandardScaler was fitted inside the pipeline on the appropriate fit rows. Trees used unscaled inputs. Random Forest fitted in parallel and summed tree predictions in a fixed order for exact replay. The last refit label became available at 2020-07-15 00:00:00, the first test issue. All choices and refits finished before test prediction. Seasonal Naive uses the count 24 hours earlier and has no fit.

The shared timestamp SHA-256 is `c5bcf9e261724722a9be69a0aa6699dd02d5185c93a27506348b97ac91eb9698`. The [training verification table](../results/tables/training_sample_verification_1h.csv) records counts, endpoints, parameters and the hash for each model.

| Model | Selected parameters | Final fit seconds | Test prediction seconds |
|---|---|---:|---:|
| Ridge Regression | alpha=3000.0, solver=svd | 0.036 | 0.0028 |
| Random Forest | max_depth=10, max_features=0.75, min_samples_leaf=6, min_samples_split=12, n_estimators=160, n_jobs=4, random_state=42 | 0.843 | 0.0219 |
| Gradient Boosting | learning_rate=0.05, loss=squared_error, max_depth=3, min_samples_leaf=6, min_samples_split=12, n_estimators=160, n_iter_no_change=None, random_state=42, subsample=1.0 | 1.757 | 0.0036 |

## Official results

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE | Test R² | Negative test predictions |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal Naive | 0.241950 | 0.882624 | 0.278155 | 0.712417 | -0.761252 | 0 |
| Ridge | 0.660019 | 1.047654 | 0.543987 | 0.814958 | -1.304751 | 1,202 |
| Ridge Clipped | 0.579314 | 1.029747 | 0.464961 | 0.792302 | -1.178387 | 0 |
| Random Forest | 0.356619 | 0.698094 | 0.299418 | 0.569884 | -0.127006 | 0 |
| Gradient Boosting | 0.316020 | 0.734686 | 0.294247 | 0.568010 | -0.119604 | 172 |
| Gradient Boosting Clipped | 0.308562 | 0.734426 | 0.293063 | 0.567935 | -0.119308 | 0 |

Validation predictions come from the selected training-only candidate. Test predictions come from its fresh refit. MAE and RMSE are measured in recorded starts per hour. R² is supplementary and uses 1 − SSE/SST, with NA when the target variance is zero. Raw and clipped Ridge and Gradient Boosting predictions are preserved separately. Random Forest had no negative validation or test predictions, so clipping it leaves every prediction unchanged.

## Error by observed target

Zero counts made up 86.8% of test targets. These groups use observed test outcomes for retrospective diagnosis. The held-out errors estimate performance on this Caltech window, with operating coverage still unknown.

| Model | MAE on zero-count hours | MAE on positive-count hours |
|---|---:|---:|
| Seasonal Naive | 0.147720 | 1.139326 |
| Ridge | 0.488119 | 0.912838 |
| Ridge Clipped | 0.400639 | 0.889630 |
| Random Forest | 0.194381 | 0.992902 |
| Gradient Boosting | 0.192087 | 0.968736 |
| Gradient Boosting Clipped | 0.190742 | 0.968614 |

## Reproduce

Runtime versions were Python 3.12.10, scikit-learn 1.8.0, pandas 3.0.2, NumPy 2.4.4 and Matplotlib 3.10.8. Use the existing `experiments/03_tree_models_1h/requirements.txt` and run from Stage 3:

```powershell
python -B experiments/03_tree_models_1h/run_comparison.py --verify
```

The official runner checks the existing data SHA-256 and split ledger. Verification retrains every candidate and refit, compares the six generated tables apart from timings, and checks the hashes of the five regenerated charts. Results are published only for Caltech and the one-hour horizon.
