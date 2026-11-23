# Caltech 多时间跨度充电会话开始数预测

## 实验口径

目标为源时钟内每小时记录到的会话开始次数。令 t 为最后一个已完成小时的起点，在 t+1h 发布 y(t+h) 的预测。采用逐小时滚动起点和直接预测，每个 h 单独训练。使用同一套 34 列，其中 4 列需求历史锚定 t，30 列日历特征由目标小时 t+h 决定。Seasonal Naive 取 y(t+h−24)。测试期已完成且已可用的真实观测可供后续起点使用。

复用 Phase 3 边界及自动化验证。筛选按目标小时，剔除涉及不完整边界小时的样本，保留原视图的零值与峰值。各模型在相同初始训练段训练候选，仅用未截断的验证 MAE 选参。每个提前量重新选参，沿用正式 1h 的 8 个 Ridge alpha、8 组 RF 及 8 组 GB 设置。配置包含原 1h 优选设置和固定种子 42。最终三个模型用相同的训练与验证时间戳重新拟合，Ridge 标准化也重新拟合。测试标签未用于拟合或选择。

| h | 初始训练 | 验证 | 最终拟合 | 测试 | 首个测试发布时间 | 分界剔除 训练 / 验证 |
|---:|---:|---:|---:|---:|---|---:|
| 1 | 15,790 | 3,385 | 19,175 | 3,383 | 2020-07-15 00:00:00 | 0 / 0 |
| 3 | 15,786 | 3,383 | 19,169 | 3,383 | 2020-07-14 22:00:00 | 2 / 2 |
| 6 | 15,780 | 3,380 | 19,160 | 3,383 | 2020-07-14 19:00:00 | 5 / 5 |
| 24 | 15,744 | 3,362 | 19,106 | 3,383 | 2020-07-14 01:00:00 | 23 / 23 |

测试目标范围为 2020-07-15 00:00 至 2020-12-02 22:00。经逐键核对，各提前量实际选出的目标小时完全相同。时间戳哈希、历史边界剔除与预热数量见切分表。正式 1h 结果的逐行最大预测绝对差为 0，指标、候选选取、最终拟合时间戳均通过参考检查。

## 原始预测与模型排名

| h | 模型 | MAE | RMSE | R² |
|---:|---|---:|---:|---:|
| 1 | Seasonal Naive | 0.278155 | 0.712417 | -0.761252 |
| 1 | Ridge | 0.543987 | 0.814958 | -1.304751 |
| 1 | Random Forest | 0.299418 | 0.569884 | -0.127006 |
| 1 | Gradient Boosting | 0.294247 | 0.568010 | -0.119604 |
| 3 | Seasonal Naive | 0.278155 | 0.712417 | -0.761252 |
| 3 | Ridge | 0.824737 | 1.246749 | -4.394001 |
| 3 | Random Forest | 0.390595 | 0.841813 | -1.459144 |
| 3 | Gradient Boosting | 0.403004 | 0.772396 | -1.070299 |
| 6 | Seasonal Naive | 0.278155 | 0.712417 | -0.761252 |
| 6 | Ridge | 0.926026 | 1.344622 | -5.274132 |
| 6 | Random Forest | 0.417097 | 0.897348 | -1.794314 |
| 6 | Gradient Boosting | 0.480471 | 0.881842 | -1.698576 |
| 24 | Seasonal Naive | 0.278155 | 0.712417 | -0.761252 |
| 24 | Ridge | 0.795967 | 1.120351 | -3.355734 |
| 24 | Random Forest | 0.459774 | 0.771447 | -1.065213 |
| 24 | Gradient Boosting | 0.495178 | 0.817273 | -1.317861 |

| h | 原始 MAE 最低 | 原始 RMSE 最低 | 含截断 MAE 最低 | 含截断 RMSE 最低 |
|---:|---|---|---|---|
| 1 | Seasonal Naive (0.278155) | Gradient Boosting (0.568010) | Seasonal Naive | Gradient Boosting 截断 |
| 3 | Seasonal Naive (0.278155) | Seasonal Naive (0.712417) | Seasonal Naive | Seasonal Naive |
| 6 | Seasonal Naive (0.278155) | Seasonal Naive (0.712417) | Seasonal Naive | Seasonal Naive |
| 24 | Seasonal Naive (0.278155) | Seasonal Naive (0.712417) | Seasonal Naive | Seasonal Naive |

- **Seasonal Naive** 的 MAE 为 0.2782 → 0.2782 → 0.2782 → 0.2782，RMSE 为 0.7124 → 0.7124 → 0.7124 → 0.7124。
- **Ridge** 的 MAE 为 0.5440 → 0.8247 → 0.9260 → 0.7960，RMSE 为 0.8150 → 1.2467 → 1.3446 → 1.1204。
- **Random Forest** 的 MAE 为 0.2994 → 0.3906 → 0.4171 → 0.4598，RMSE 为 0.5699 → 0.8418 → 0.8973 → 0.7714。
- **Gradient Boosting** 的 MAE 为 0.2942 → 0.4030 → 0.4805 → 0.4952，RMSE 为 0.5680 → 0.7724 → 0.8818 → 0.8173。

Seasonal Naive 的参考始终是目标小时之前 24 小时的观测。对相同目标小时，不同提前量的这个参考值相同，且在各发布时点已可用。因此它的误差曲线保持平坦。学习模型的短期历史与目标间距随 h 改变，并且候选选择和可用训练样本也随 h 改变。

- Random Forest 的原始 MAE 低于原始 Ridge 的提前量为 1、3、6、24 小时。
- Random Forest 的原始 RMSE 低于原始 Ridge 的提前量为 1、3、6、24 小时。
- Gradient Boosting 的原始 MAE 低于原始 Ridge 的提前量为 1、3、6、24 小时。
- Gradient Boosting 的原始 RMSE 低于原始 Ridge 的提前量为 1、3、6、24 小时。

下表用原始 Gradient Boosting 误差减去原始 Random Forest 误差。负值表示 Gradient Boosting 更低，可以直接检查其 1h 优势在较长提前量是否持续。

| h | GB − RF MAE | GB − RF RMSE |
|---:|---:|---:|
| 1 | -0.005171 | -0.001874 |
| 3 | +0.012408 | -0.069417 |
| 6 | +0.063374 | -0.015506 |
| 24 | +0.035404 | +0.045826 |

R² 的零点以测试窗口实际均值作参照，负值表示平方误差高于该参照。该参照均值用来定义指标，不是可部署的测试期间预测规则。

## 零值、正值和截断

切片根据事后实测目标定义，只用于解释误差。下表显示样本比例和原始模型的切片赢家。各模型的完整 MAE、RMSE、样本数与占比在 `horizon_error_breakdown.csv`，包含适用的截断变体。

| h | 子集 | 小时 | 占比 | 原始 MAE 最低 | 原始 RMSE 最低 |
|---:|---|---:|---:|---|---|
| 1 | zero | 2,938 | 86.85% | Seasonal Naive (0.147720) | Gradient Boosting (0.355937) |
| 1 | positive | 445 | 13.15% | Ridge (0.912838) | Ridge (1.238020) |
| 3 | zero | 2,938 | 86.85% | Seasonal Naive (0.147720) | Seasonal Naive (0.525717) |
| 3 | positive | 445 | 13.15% | Ridge (0.917280) | Ridge (1.338248) |
| 6 | zero | 2,938 | 86.85% | Seasonal Naive (0.147720) | Seasonal Naive (0.525717) |
| 6 | positive | 445 | 13.15% | Ridge (0.947829) | Ridge (1.359910) |
| 24 | zero | 2,938 | 86.85% | Seasonal Naive (0.147720) | Seasonal Naive (0.525717) |
| 24 | positive | 445 | 13.15% | Random Forest (0.866624) | Random Forest (1.198891) |

所有提前量匹配同一组实际值，因此零值比例相同。总体 MAE 是两个子集 MAE 按小时比例的加权平均，整体 RMSE 则对分组平方误差先加权再开方。零值占比较高时，零值误差对总体 MAE 排名权重较大，正值小时的较大误差仍可显著影响 RMSE。

| h | 模型 | 截断前负测试预测 | 截断后 − 原始 MAE | 截断后 − 原始 RMSE |
|---:|---|---:|---:|---:|
| 1 | Ridge | 1,202 | -0.079026 | -0.022656 |
| 1 | Gradient Boosting | 172 | -0.001184 | -0.000075 |
| 3 | Ridge | 568 | -0.046980 | -0.010413 |
| 3 | Gradient Boosting | 973 | -0.019252 | -0.002430 |
| 6 | Ridge | 542 | -0.054805 | -0.012967 |
| 6 | Gradient Boosting | 752 | -0.021756 | -0.002613 |
| 24 | Ridge | 1,065 | -0.147292 | -0.058760 |
| 24 | Gradient Boosting | 497 | -0.018741 | -0.004170 |

对于非负实测值，把负预测截为零会在该点减少绝对与平方误差。表中只列有负验证或测试预测的模型与提前量。Random Forest 的逐行原始预测也已保存，可核查负值数量。截断规则固定，未参与选参。

## 选参与局限

| h | 模型 | 选中参数 | 最终拟合时间戳 SHA-256 前缀 |
|---:|---|---|---|
| 1 | Ridge | alpha=3000.0 | c5bcf9e26172 |
| 1 | Random Forest | max_depth=10、max_features=0.75、min_samples_leaf=6、min_samples_split=12、n_estimators=160 | c5bcf9e26172 |
| 1 | Gradient Boosting | learning_rate=0.05、max_depth=3、min_samples_leaf=6、min_samples_split=12、n_estimators=160、subsample=1.0 | c5bcf9e26172 |
| 3 | Ridge | alpha=10000.0 | 0e8fafec5974 |
| 3 | Random Forest | max_depth=12、max_features=1.0、min_samples_leaf=4、min_samples_split=8、n_estimators=160 | 0e8fafec5974 |
| 3 | Gradient Boosting | learning_rate=0.05、max_depth=4、min_samples_leaf=6、min_samples_split=12、n_estimators=160、subsample=0.85 | 0e8fafec5974 |
| 6 | Ridge | alpha=10000.0 | eff4841cf94e |
| 6 | Random Forest | max_depth=None、max_features=0.5、min_samples_leaf=2、min_samples_split=4、n_estimators=160 | eff4841cf94e |
| 6 | Gradient Boosting | learning_rate=0.05、max_depth=4、min_samples_leaf=6、min_samples_split=12、n_estimators=160、subsample=0.85 | eff4841cf94e |
| 24 | Ridge | alpha=3000.0 | 83edffdc67f9 |
| 24 | Random Forest | max_depth=6、max_features=0.75、min_samples_leaf=16、min_samples_split=24、n_estimators=120 | 83edffdc67f9 |
| 24 | Gradient Boosting | learning_rate=0.05、max_depth=2、min_samples_leaf=8、min_samples_split=16、n_estimators=150、subsample=0.8 | 83edffdc67f9 |

| h | 模型 | 选中候选训练 MAE | 选中候选验证 MAE | 训练误差最低候选的验证 MAE |
|---:|---|---:|---:|---:|
| 1 | Ridge | 1.206281 | 0.660019 | 0.675247 |
| 1 | Random Forest | 0.971711 | 0.356619 | 0.442929 |
| 1 | Gradient Boosting | 1.056999 | 0.316020 | 0.355785 |
| 3 | Ridge | 1.521421 | 1.086448 | 1.239249 |
| 3 | Random Forest | 0.980300 | 0.445309 | 0.556668 |
| 3 | Gradient Boosting | 1.096237 | 0.550433 | 0.550433 |
| 6 | Ridge | 1.552937 | 1.212669 | 1.332933 |
| 6 | Random Forest | 0.748727 | 0.563496 | 0.563496 |
| 6 | Gradient Boosting | 1.146465 | 0.749528 | 0.749528 |
| 24 | Ridge | 1.363682 | 0.970932 | 1.023430 |
| 24 | Random Forest | 1.194439 | 0.605423 | 0.867678 |
| 24 | Gradient Boosting | 1.212088 | 0.689926 | 0.722671 |

最小训练误差的候选与最小验证误差的候选若不同，说明进一步降低样本内误差未带来相同的后期收益。训练、验证和测试的需求分布不同，且正式测试使用训练加验证重拟合。绝对误差差距无法单独归因为过拟合。指标只覆盖一个站点的一个测试窗口，没有提供跨站或跨时期的不确定性估计。归档覆盖状态和时区均未知，零值表示未记录新会话，结果依赖完整小时计数在发布时及时可用的假设。

## 复现

依赖沿用正式一小时版本，运行环境为 Python 3.12.10、scikit-learn 1.8.0、pandas 3.0.2、NumPy 2.4.4、Matplotlib 3.10.8。

```powershell
python -B experiments/04_multi_horizon/run_multi_horizon.py --verify
```

全部图表依据本阶段落盘的预测与统计表重画。为保留既有一小时图，10 张图统一放在 `results/figures/multi_horizon/`。趋势、模型比较和分组误差图展示四个原始模型，逐日实际与预测图另外标出适用的截断预测。分组图的两个子图纵轴独立，数值大小以坐标轴和 CSV 为准。实验范围限于 Caltech。
