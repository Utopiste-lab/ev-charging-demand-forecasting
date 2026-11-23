# Stage 3 Research Index

数据入口是 `data/processed/hourly_counts_v1.csv`，时间与标签约定见 `config/data_contract.yaml`。全过程取 source_clock，时区及实际覆盖为 unknown。正式结果和复现入口如下。

| 阶段 | 研究报告 | 运行或复核入口 |
|---|---|---|
| 档案审查 | [Stage 2 审查](00_stage2_audit.md) · [复用决策](00_data_reuse_decision.md) | Stage 2 只读源与 Stage 3 数据来源 JSON |
| 01–03 | [数据合约](01_data_contract_and_baseline.md) · [特征/切分](02_feature_and_split_validation.md) · [统一 1h](03_unified_1h_revision.md) · [传统模型](03_tree_models_1h.md) | `experiments/02_feature_validation/run_validation.py --check-only` · `experiments/03_tree_models_1h/run_comparison.py --verify` |
| 04 | [四跨度直接预测](04_multi_horizon_analysis.md) | `experiments/04_multi_horizon/run_multi_horizon.py --verify` |
| 05–06 | [迁移设计](05_cross_site_experimental_design.md) · [双向迁移](06_cross_site_analysis.md) | `experiments/05_cross_site_audit/audit_common_window.py --verify` · `experiments/06_cross_site_training/run_cross_site.py --verify` |
| 07 | [Hurdle 与残差](07_advanced_forecasting.md) · [滚动补充](07_robustness_supplement.md) | `experiments/07_advanced_forecasting/verify_experiment.py` |
| 08 | [自适应混合](08_adaptive_hybrid.md) | `experiments/08_adaptive_hybrid/verify_experiment.py` |
| 09 | [概率预测](09_probabilistic_forecasting.md) | `experiments/09_probabilistic_forecasting/run_probabilistic_forecasting.py --verify` · `verify_experiment.py` 同目录 |
| 收官 | [最终学术报告](STAGE3_FINAL_REPORT.md) · [完成清单](STAGE3_COMPLETION_CHECKLIST.md) · [后续阶段就绪评估](STAGE4_READINESS_ASSESSMENT.md) | `experiments/09_probabilistic_forecasting/integrate_stage3.py --verify` · `verify_frozen_inputs.py` 同目录 |

上述入口在 Stage 3 根下用 `python -X utf8 -B` 调用。个别历史点模型逐位比较受浮点尾数影响时，可用本阶段 `replay_legacy_point.py 03|04|06` 分别核验。它保留非浮点精确比较，数值容差绝对和相对均为 2e−11，单列图哈希结果。

**原精度结果。** `results/metrics/stage3_final/` 提供恒零比较、各设定的全部同键点结果、最佳 MAE/RMSE、匹配迁移差值、适应性对照、概率指标副本、滚动比较和局限。逐小时恒零、历史指标审查、输入 SHA-256、原始元数据盘点、复现审查与冻结清单位于 `results/tables/stage3_final/`。

**概率细节。** `results/metrics/probabilistic_forecasting/` 为原精度点、分位数、区间、区块 bootstrap 与约束频率；`results/tables/probabilistic_forecasting/` 为逐行分位数、校准分数、候选预测、配置和各时间键哈希。图分别位于 `results/figures/probabilistic_forecasting/` 和 `results/figures/stage3_final/`。执行记录集中于 `logs/phase10_probabilistic_forecasting.log`。
