# Stage 3 数据口径与统一一小时基准

## 数据口径

目标为 **Hourly Recorded Session-Start Counts**。按归档文件名的会话开始时间，统计每个站点在左闭右开小时区间内的记录数。每个归档 CSV 包含一个源会话的多条遥测。运营覆盖和物理车辆到达需要独立证据。

`data/processed/hourly_counts_v1.csv` 保持原始站点、时间戳、计数和行序。字段为 `site`、`timestamp`、`session_start_count`、`clock_basis`、`timezone_status`、`coverage_status`、`is_first_hour` 和 `is_last_hour`。时间为 `source_clock`，时区和运营覆盖均为 `unknown`。所有零值及峰值保留于原视图。首尾小时可能只覆盖部分时段，经验证的边界策略仅限制实验样本。

| 站点 | 小时 | 起点 | 末小时 | 零小时 | 峰值 |
|---|---:|---|---|---:|---:|
| Caltech | 22,728 | 2018-05-01 00:00:00 | 2020-12-02 23:00:00 | 13,379 | 193 |
| JPL | 18,875 | 2018-10-08 13:00:00 | 2020-12-02 23:00:00 | 12,575 | 35 |

`data/processed/data_provenance.json` 记录 v1 创建时的输入和配置指纹。当前实验使用统一协议的配置，小时视图及其原始溯源文件保持原样。

## 正式一小时基准

Caltech 使用与完整比较相同的 3,383 个测试目标，范围为 2020-07-15 00:00:00 至 2020-12-02 22:00:00。Ridge 先在 15,790 个训练样本上拟合每个候选，在 3,385 个验证样本上按原始 MAE 选优，再将 StandardScaler 与 Ridge 从头拟合于同一组 19,175 个符合时序的最终样本。选中 alpha 为 **3000**。Seasonal Naive 取前 24 小时对应计数。截断版保留为独立预测版本。

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE | Test R² | Negative test predictions |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal Naive | 0.241950 | 0.882624 | 0.278155 | 0.712417 | -0.761252 | 0 |
| Ridge | 0.660019 | 1.047654 | 0.543987 | 0.814958 | -1.304751 | 1,202 |
| Ridge Clipped | 0.579314 | 1.029747 | 0.464961 | 0.792302 | -1.178387 | 0 |

正式基准文件为 [baseline_1h_unified.csv](../results/metrics/baseline_1h_unified.csv) 和 [baseline_predictions.csv](../results/tables/baseline_predictions.csv)，均由[完整统一实验](03_tree_models_1h.md)的结果筛出。旧 Stage 2 指标仅属于历史审查。基准入口调用统一运行器：

```powershell
python -B experiments/01_baseline_and_ridge/run_baseline.py --verify
```
