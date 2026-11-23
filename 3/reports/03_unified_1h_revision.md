# Caltech Unified One-Hour Revision

The sole official one-hour comparison is the boundary-aware experiment in [03_tree_models_1h.md](03_tree_models_1h.md). It evaluates six forecast variants on the same 3,383 targets from 2020-07-15 00:00:00 through 2020-12-02 22:00:00 in the source clock.

## What changed

The superseded comparison fitted Ridge on the training set alone and allowed the trees to refit with eligible training and validation labels. The official runner now selects Ridge, Random Forest and Gradient Boosting by raw validation MAE after training each candidate on the same 15,790 rows. Validation contains 3,385 rows. All three selected estimators were reinitialized and retrained on the same **19,175** ordered target timestamps, from 2018-05-08 01:00:00 through 2020-07-14 23:00:00. Ridge refits its StandardScaler inside its pipeline. The last fitting label became available at 2020-07-15, the first test issue. No test targets entered tuning or fitting. Seasonal Naive has no fitting step.

The shared timestamp SHA-256 is `c5bcf9e261724722a9be69a0aa6699dd02d5185c93a27506348b97ac91eb9698`. The [training verification table](../results/tables/training_sample_verification_1h.csv) records the ordered timestamp hash and fit boundaries separately for each learned model.

| Model | Selected hyperparameters |
|---|---|
| Ridge Regression | alpha=3000.0, solver=svd |
| Random Forest | max_depth=10, max_features=0.75, min_samples_leaf=6, min_samples_split=12, n_estimators=160, n_jobs=4, random_state=42 |
| Gradient Boosting | learning_rate=0.05, loss=squared_error, max_depth=3, min_samples_leaf=6, min_samples_split=12, n_estimators=160, n_iter_no_change=None, random_state=42, subsample=1.0 |

The Ridge search covered 0.1, 1, 10, 100, 1,000, 3,000, 10,000 and 30,000. RF and GB kept their previous search configurations and seeds. Random Forest used parallel fitting and serial prediction to avoid floating-point summation-order drift in the exact-replay check.

## Official scores and revision deltas

The first five metric columns are from the official run. The final three deltas are new minus superseded boundary-aware scores and describe the revision only.

| Variant | Val MAE | Val RMSE | Test MAE | Test RMSE | Test R² | Negative test forecasts | Δ MAE | Δ RMSE | Δ R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seasonal Naive | 0.241950 | 0.882624 | 0.278155 | 0.712417 | -0.761252 | 0 | +0.000000 | +0.000000 | +0.000000 |
| Ridge | 0.660019 | 1.047654 | 0.543987 | 0.814958 | -1.304751 | 1202 | -0.126839 | -0.175749 | +1.101240 |
| Ridge Clipped | 0.579314 | 1.029747 | 0.464961 | 0.792302 | -1.178387 | 0 | -0.090943 | -0.164022 | +0.995295 |
| Random Forest | 0.356619 | 0.698094 | 0.299418 | 0.569884 | -0.127006 | 0 | +0.000000 | +0.000000 | +0.000000 |
| Gradient Boosting | 0.316020 | 0.734686 | 0.294247 | 0.568010 | -0.119604 | 172 | +0.000000 | +0.000000 | +0.000000 |
| Gradient Boosting Clipped | 0.308562 | 0.734426 | 0.293063 | 0.567935 | -0.119308 | 0 | +0.000000 | +0.000000 | +0.000000 |

Ridge improved under the revised pipeline, which changes both alpha and the number of fit labels. The experiment cannot attribute the improvement to either change alone. Seasonal Naive and both tree estimators showed no changes at the six-decimal precision above. Seasonal Naive retained the lowest MAE. Both tree estimators beat both Ridge variants on MAE and RMSE, and Gradient Boosting Clipped had the lowest RMSE and least negative R². Clipping changed 1,202 Ridge and 172 Gradient Boosting test forecasts. Random Forest had no negative validation or test predictions, so clipping leaves its forecasts unchanged.

## Validation and reproduction

Independent checks recalculated MAE, RMSE and R² from all 40,608 validation and test prediction records, verified exact test keys and exact clipping, and matched the baseline files against the full comparison. All five figures were regenerated from the saved predictions and checked visually. The historical feature audit now reads the original Stage 2 code and predictions separately from official outputs. The main runner and baseline entry point both use the unified experiment. The baseline entry point reproduced all six saved tables apart from timings and matched all five figure hashes.

```powershell
python -B experiments/01_baseline_and_ridge/run_baseline.py --verify
```

The [execution log](../logs/phase4_tree_models.log) records before/after SHA-256 values for the existing data, provenance, data contract, feature code, split code and split table. All 28 Stage 2 file hashes and inventory matched before and after. Raw archives were not accessed. No multi-horizon or cross-site models were trained. The dependency versions are unchanged and both entry points now share the same requirements.

## Files changed relative to Stage 3

- `config/baseline_1h.yaml`
- `config/feature_validation.yaml`
- `config/tree_models_1h.yaml`
- `experiments/01_baseline_and_ridge/requirements.txt`
- `experiments/01_baseline_and_ridge/run_baseline.py`
- `experiments/02_feature_validation/run_validation.py`
- `experiments/02_feature_validation/test_feature_validation.py`
- `experiments/03_tree_models_1h/presentation.py`
- `experiments/03_tree_models_1h/requirements.txt`
- `experiments/03_tree_models_1h/run_comparison.py`
- `logs/phase4_tree_models.log`
- `reports/00_stage3_implementation_plan.md`
- `reports/01_data_contract_and_baseline.md`
- `reports/02_feature_and_split_validation.md`
- `reports/03_tree_models_1h.md`
- `reports/03_unified_1h_revision.md` (new)
- `results/figures/actual_vs_predicted_1h.png`
- `results/figures/model_comparison_1h_mae.png`
- `results/figures/model_comparison_1h_rmse.png`
- `results/figures/residual_comparison_1h.png`
- `results/figures/validation_vs_test_1h.png`
- `results/metrics/baseline_1h_reproduction.csv` (removed and replaced by the unified baseline table)
- `results/metrics/baseline_1h_unified.csv` (new)
- `results/metrics/tree_models_1h_comparison.csv`
- `results/tables/baseline_predictions.csv`
- `results/tables/training_sample_verification_1h.csv` (new)
- `results/tables/tree_models_1h_hyperparameters.csv`
- `results/tables/tree_models_1h_predictions.csv`
- `src/evaluation.py` (new)
