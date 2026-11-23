# Caltech 与 JPL 跨站预测及匹配同站对照

## 结论

- **Caltech → JPL** 的 12 个学习模型原始组合中，4 个 MAE 低于同站对照，4 个 RMSE 低于同站对照。MAE 差范围 -0.1885 至 +0.3276，RMSE 差范围 -0.2137 至 +0.4047。
- **JPL → Caltech** 的 12 个学习模型原始组合中，0 个 MAE 低于同站对照，0 个 RMSE 低于同站对照。MAE 差范围 +0.0490 至 +0.7306，RMSE 差范围 +0.0293 至 +2.1019。

原始变体中，最大 MAE 损失是 JPL → Caltech 24h 的 Random Forest（+0.730583），最低差值出现在 Caltech → JPL 3h 的 Ridge（-0.188505）。最大 RMSE 损失是 JPL → Caltech 24h 的 Random Forest（+2.101886，相对 +266.46%）。两个方向呈现不同的迁移代价。负值说明迁移在该目标窗口的这一指标低于匹配同站误差。

## 设计与公平性

两方向与两组匹配同站对照使用 2018-10-08 13:00 至 2020-12-02 23:00 的共同源时钟历史，以及完全相同的 3,383 个测试目标小时，范围为 2020-07-15 00:00 至 2020-12-02 22:00。模型在 1、3、6、24h 分别独立拟合，最终样本数依次为 15,322、15,316、15,307、15,253。每组的三个学习模型均在同站训练段拟合八个候选，按该站原始验证 MAE 选参，然后从头拟合可用的训练及验证数据。两个同站对照独立选参及拟合，未将其结果反馈给迁移模型。因为某些设置使用同一站点的相同样本、网格和种子，独立训练得到同样参数属于预期。

迁移的训练标签仅来自源站。模型冻结后，以目标站在发布时已完成的历史作输入。目标站的 Seasonal Naive 取目标小时减 24 小时的该站记录，并在该目标的迁移与同站对照中复用。日历特征始终使用目标源时钟。Ridge 标准化只在对应训练站的拟合行内学习，树模型保留计数尺度。预测冻结后才连接测试真值。此实验使用目标历史的滚动起点，严格无目标历史的 zero-shot 将需要另外的协议。

核实了审计中的可用样本、每处分界按跨度剔除 0/2/5/23 小时、最终拟合时间戳及标签哈希、目标键和 34 列 schema。扰动首个目标测试观测，48 个已冻结模型在该观测可用之前的预测均逐行不变，对之后预测的实际影响已记录。两站正式同站参照均使用与迁移相同的共同历史，已发布的完整 Caltech 同站结果只提供背景。

## 两个方向的原始结果

| 方向 | h | 模型 | 迁移 MAE | 迁移 RMSE | 迁移 R² | 同站 MAE | 同站 RMSE | 同站 R² |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Caltech → JPL | 1 | Seasonal Naive | 0.665386 | 1.388586 | 0.129323 | 0.665386 | 1.388586 | 0.129323 |
| Caltech → JPL | 1 | Ridge | 0.686405 | 1.107445 | 0.446195 | 0.688186 | 1.113133 | 0.440492 |
| Caltech → JPL | 1 | Random Forest | 0.631230 | 1.137351 | 0.415882 | 0.574837 | 1.020020 | 0.530183 |
| Caltech → JPL | 1 | Gradient Boosting | 0.611827 | 1.081693 | 0.471652 | 0.573236 | 1.028563 | 0.522279 |
| Caltech → JPL | 3 | Seasonal Naive | 0.665386 | 1.388586 | 0.129323 | 0.665386 | 1.388586 | 0.129323 |
| Caltech → JPL | 3 | Ridge | 0.939029 | 1.452627 | 0.047160 | 1.127534 | 1.409955 | 0.102319 |
| Caltech → JPL | 3 | Random Forest | 0.967513 | 1.922099 | -0.668258 | 0.639901 | 1.524756 | -0.049815 |
| Caltech → JPL | 3 | Gradient Boosting | 0.959690 | 1.828280 | -0.509375 | 0.683286 | 1.423590 | 0.084873 |
| Caltech → JPL | 6 | Seasonal Naive | 0.665386 | 1.388586 | 0.129323 | 0.665386 | 1.388586 | 0.129323 |
| Caltech → JPL | 6 | Ridge | 0.991245 | 1.472030 | 0.021535 | 1.140224 | 1.419664 | 0.089913 |
| Caltech → JPL | 6 | Random Forest | 0.946056 | 1.782929 | -0.435423 | 0.630426 | 1.530004 | -0.057054 |
| Caltech → JPL | 6 | Gradient Boosting | 0.937479 | 1.647612 | -0.225807 | 0.725805 | 1.450613 | 0.049800 |
| Caltech → JPL | 24 | Seasonal Naive | 0.665386 | 1.388586 | 0.129323 | 0.665386 | 1.388586 | 0.129323 |
| Caltech → JPL | 24 | Ridge | 0.836336 | 1.250237 | 0.294175 | 0.976893 | 1.345590 | 0.182407 |
| Caltech → JPL | 24 | Random Forest | 0.788365 | 1.343044 | 0.185497 | 0.650248 | 1.556716 | -0.094285 |
| Caltech → JPL | 24 | Gradient Boosting | 0.723255 | 1.226819 | 0.320370 | 0.700083 | 1.330272 | 0.200915 |
| JPL → Caltech | 1 | Seasonal Naive | 0.278155 | 0.712417 | -0.761252 | 0.278155 | 0.712417 | -0.761252 |
| JPL → Caltech | 1 | Ridge | 0.577956 | 0.838291 | -1.438612 | 0.528943 | 0.809011 | -1.271235 |
| JPL → Caltech | 1 | Random Forest | 0.358207 | 0.645801 | -0.447272 | 0.299593 | 0.576286 | -0.152469 |
| JPL → Caltech | 1 | Gradient Boosting | 0.376165 | 0.630365 | -0.378914 | 0.299338 | 0.567104 | -0.116036 |
| JPL → Caltech | 3 | Seasonal Naive | 0.278155 | 0.712417 | -0.761252 | 0.278155 | 0.712417 | -0.761252 |
| JPL → Caltech | 3 | Ridge | 1.275645 | 1.620943 | -8.117773 | 0.764051 | 1.205695 | -4.044611 |
| JPL → Caltech | 3 | Random Forest | 0.923744 | 2.644406 | -23.266626 | 0.387317 | 0.839770 | -1.447222 |
| JPL → Caltech | 3 | Gradient Boosting | 0.879935 | 2.004686 | -12.945878 | 0.389585 | 0.786718 | -1.147787 |
| JPL → Caltech | 6 | Seasonal Naive | 0.278155 | 0.712417 | -0.761252 | 0.278155 | 0.712417 | -0.761252 |
| JPL → Caltech | 6 | Ridge | 1.301315 | 1.647834 | -8.422799 | 0.846574 | 1.275604 | -4.646568 |
| JPL → Caltech | 6 | Random Forest | 1.039756 | 2.971405 | -29.639154 | 0.415923 | 0.897682 | -1.796394 |
| JPL → Caltech | 6 | Gradient Boosting | 1.009764 | 2.523122 | -21.091720 | 0.455874 | 0.888723 | -1.740853 |
| JPL → Caltech | 24 | Seasonal Naive | 0.278155 | 0.712417 | -0.761252 | 0.278155 | 0.712417 | -0.761252 |
| JPL → Caltech | 24 | Ridge | 1.099967 | 1.495699 | -6.763211 | 0.696845 | 0.929312 | -1.996927 |
| JPL → Caltech | 24 | Random Forest | 1.187172 | 2.890692 | -27.997247 | 0.456589 | 0.788806 | -1.159204 |
| JPL → Caltech | 24 | Gradient Boosting | 1.177938 | 2.540093 | -21.389908 | 0.485217 | 0.793623 | -1.185652 |

下表在各自比较中纳入适用的截断变体，MAE 与 RMSE 分别排序。

| 方向 | h | 迁移最低 MAE | 迁移最低 RMSE | 匹配同站最低 MAE | 匹配同站最低 RMSE |
|---|---:|---|---|---|---|
| Caltech → JPL | 1 | Gradient Boosting 截断 (0.6109) | Gradient Boosting 截断 (1.0817) | Gradient Boosting 截断 (0.5729) | Random Forest (1.0200) |
| Caltech → JPL | 3 | Seasonal Naive (0.6654) | Seasonal Naive (1.3886) | Random Forest (0.6399) | Seasonal Naive (1.3886) |
| Caltech → JPL | 6 | Seasonal Naive (0.6654) | Seasonal Naive (1.3886) | Random Forest (0.6304) | Seasonal Naive (1.3886) |
| Caltech → JPL | 24 | Seasonal Naive (0.6654) | Gradient Boosting 截断 (1.2258) | Random Forest (0.6502) | Ridge 截断 (1.3292) |
| JPL → Caltech | 1 | Seasonal Naive (0.2782) | Gradient Boosting 截断 (0.6302) | Seasonal Naive (0.2782) | Gradient Boosting 截断 (0.5666) |
| JPL → Caltech | 3 | Seasonal Naive (0.2782) | Seasonal Naive (0.7124) | Seasonal Naive (0.2782) | Seasonal Naive (0.7124) |
| JPL → Caltech | 6 | Seasonal Naive (0.2782) | Seasonal Naive (0.7124) | Seasonal Naive (0.2782) | Seasonal Naive (0.7124) |
| JPL → Caltech | 24 | Seasonal Naive (0.2782) | Seasonal Naive (0.7124) | Seasonal Naive (0.2782) | Seasonal Naive (0.7124) |

Caltech→JPL 的 1h 迁移可优于目标站 Seasonal Naive，且 24h 的 Gradient Boosting 在 RMSE 上也领先季节基准。3h 和 6h 的总体赢家是目标站 Seasonal Naive。JPL→Caltech 的 1h Gradient Boosting 在 RMSE 上优于 Seasonal Naive，同时其 MAE 仍较高。较长跨度两个指标均由 Seasonal Naive 领先。三个学习模型在两个方向的 1h 原始 RMSE 都比 3h 和 6h 低。Caltech→JPL 三个学习模型的 24h 原始 RMSE 又都低于各自 6h，误差随跨度变化并不单调。

## 迁移代价

差值为迁移减同站，正值表示误差上升，百分比以同站误差为分母。下表列原始变体。CSV 另外包含三个学习模型的截断配对。Seasonal Naive 在对应目标站两种协议的差为零。

| 方向 | h | 模型 | ΔMAE | 相对 MAE | ΔRMSE | 相对 RMSE |
|---|---:|---|---:|---:|---:|---:|
| Caltech → JPL | 1 | Seasonal Naive | +0.000000 | +0.00% | +0.000000 | +0.00% |
| Caltech → JPL | 1 | Ridge | -0.001781 | -0.26% | -0.005688 | -0.51% |
| Caltech → JPL | 1 | Random Forest | +0.056393 | +9.81% | +0.117331 | +11.50% |
| Caltech → JPL | 1 | Gradient Boosting | +0.038592 | +6.73% | +0.053130 | +5.17% |
| Caltech → JPL | 3 | Seasonal Naive | +0.000000 | +0.00% | +0.000000 | +0.00% |
| Caltech → JPL | 3 | Ridge | -0.188505 | -16.72% | +0.042672 | +3.03% |
| Caltech → JPL | 3 | Random Forest | +0.327612 | +51.20% | +0.397342 | +26.06% |
| Caltech → JPL | 3 | Gradient Boosting | +0.276404 | +40.45% | +0.404690 | +28.43% |
| Caltech → JPL | 6 | Seasonal Naive | +0.000000 | +0.00% | +0.000000 | +0.00% |
| Caltech → JPL | 6 | Ridge | -0.148979 | -13.07% | +0.052367 | +3.69% |
| Caltech → JPL | 6 | Random Forest | +0.315630 | +50.07% | +0.252925 | +16.53% |
| Caltech → JPL | 6 | Gradient Boosting | +0.211674 | +29.16% | +0.196999 | +13.58% |
| Caltech → JPL | 24 | Seasonal Naive | +0.000000 | +0.00% | +0.000000 | +0.00% |
| Caltech → JPL | 24 | Ridge | -0.140557 | -14.39% | -0.095352 | -7.09% |
| Caltech → JPL | 24 | Random Forest | +0.138117 | +21.24% | -0.213671 | -13.73% |
| Caltech → JPL | 24 | Gradient Boosting | +0.023172 | +3.31% | -0.103454 | -7.78% |
| JPL → Caltech | 1 | Seasonal Naive | +0.000000 | +0.00% | +0.000000 | +0.00% |
| JPL → Caltech | 1 | Ridge | +0.049012 | +9.27% | +0.029280 | +3.62% |
| JPL → Caltech | 1 | Random Forest | +0.058614 | +19.56% | +0.069515 | +12.06% |
| JPL → Caltech | 1 | Gradient Boosting | +0.076828 | +25.67% | +0.063261 | +11.16% |
| JPL → Caltech | 3 | Seasonal Naive | +0.000000 | +0.00% | +0.000000 | +0.00% |
| JPL → Caltech | 3 | Ridge | +0.511595 | +66.96% | +0.415249 | +34.44% |
| JPL → Caltech | 3 | Random Forest | +0.536427 | +138.50% | +1.804637 | +214.90% |
| JPL → Caltech | 3 | Gradient Boosting | +0.490350 | +125.86% | +1.217968 | +154.82% |
| JPL → Caltech | 6 | Seasonal Naive | +0.000000 | +0.00% | +0.000000 | +0.00% |
| JPL → Caltech | 6 | Ridge | +0.454741 | +53.72% | +0.372230 | +29.18% |
| JPL → Caltech | 6 | Random Forest | +0.623833 | +149.99% | +2.073723 | +231.01% |
| JPL → Caltech | 6 | Gradient Boosting | +0.553890 | +121.50% | +1.634399 | +183.90% |
| JPL → Caltech | 24 | Seasonal Naive | +0.000000 | +0.00% | +0.000000 | +0.00% |
| JPL → Caltech | 24 | Ridge | +0.403122 | +57.85% | +0.566387 | +60.95% |
| JPL → Caltech | 24 | Random Forest | +0.730583 | +160.01% | +2.101886 | +266.46% |
| JPL → Caltech | 24 | Gradient Boosting | +0.692721 | +142.77% | +1.746470 | +220.06% |

判断哪个模型迁移更稳健时，应同时看绝对误差和匹配差值。较小的迁移差值也可能来自较高的同站误差。两方向的目标站分布不同，直接比较两方向的绝对 MAE 会混合两个站点的计数规模。比较差值和相对变化能回答各自目标站上更换训练来源的影响。负差值可能反映两站训练时期的模式差异、有限样本或窗口特定效应，现有设计无法确定具体原因。

## 偏差与分布

共同测试期 Caltech 平均 0.1779 次/小时，JPL 为 0.7455。标准差分别为 0.5369 和 1.4884。在此计数差异下，Caltech→JPL 低估和 JPL→Caltech 高估是可检验的方向性假设。下表报告每个学习模型的有符号偏差，正值表示高估。偏差的方向以各行结果为准，数据规模差异本身不构成因果解释。

| 方向 | 模型 | 1h | 3h | 6h | 24h |
|---|---|---:|---:|---:|---:|
| Caltech → JPL | Ridge | +0.071640 | +0.156713 | +0.181638 | +0.194316 |
| Caltech → JPL | Random Forest | +0.165293 | +0.339489 | +0.250613 | +0.226205 |
| Caltech → JPL | Gradient Boosting | +0.101242 | +0.311488 | +0.237779 | +0.170365 |
| JPL → Caltech | Ridge | -0.005397 | +1.173094 | +1.201027 | +0.792217 |
| JPL → Caltech | Random Forest | +0.112650 | +0.648218 | +0.759321 | +1.002285 |
| JPL → Caltech | Gradient Boosting | +0.119901 | +0.584029 | +0.738788 | +0.990655 |

- **Caltech → JPL** 的学习模型中，12/12 个组合高估，0/12 个组合低估。原始偏差范围为 +0.0716 至 +0.3395 次/小时。
- **JPL → Caltech** 的学习模型中，11/12 个组合高估，1/12 个组合低估。原始偏差范围为 -0.0054 至 +1.2010 次/小时。

Caltech→JPL 的低估假设未获得这组原始预测支持。JPL→Caltech 多数组合高估，尤其是 3h 及 6h 的 Ridge。模型同时使用站点历史和历法，本实验没有单独干预规模，不能把这些偏差直接归因为两站均值差异。

## 零值与正值小时

每一目标站的四个跨度共享相同的实际标签。分组用于事后解释，未参与选参。完整误差分组表包含迁移、同站和截断版本。

| 方向 | h | 实测组 | 小时与比例 | 最低原始 MAE | 最低原始 RMSE |
|---|---:|---|---|---|---|
| Caltech → JPL | 1 | zero | 2,361 (69.79%) | Seasonal Naive (0.2685) | Gradient Boosting (0.5988) |
| Caltech → JPL | 1 | positive | 1,022 (30.21%) | Ridge (1.2303) | Ridge (1.7003) |
| Caltech → JPL | 3 | zero | 2,361 (69.79%) | Seasonal Naive (0.2685) | Seasonal Naive (0.8624) |
| Caltech → JPL | 3 | positive | 1,022 (30.21%) | Ridge (1.5397) | Seasonal Naive (2.1597) |
| Caltech → JPL | 6 | zero | 2,361 (69.79%) | Seasonal Naive (0.2685) | Seasonal Naive (0.8624) |
| Caltech → JPL | 6 | positive | 1,022 (30.21%) | Ridge (1.5227) | Seasonal Naive (2.1597) |
| Caltech → JPL | 24 | zero | 2,361 (69.79%) | Seasonal Naive (0.2685) | Gradient Boosting (0.7639) |
| Caltech → JPL | 24 | positive | 1,022 (30.21%) | Ridge (1.2861) | Ridge (1.8868) |
| JPL → Caltech | 1 | zero | 2,938 (86.85%) | Seasonal Naive (0.1477) | Gradient Boosting (0.4563) |
| JPL → Caltech | 1 | positive | 445 (13.15%) | Gradient Boosting (0.9999) | Gradient Boosting (1.2831) |
| JPL → Caltech | 3 | zero | 2,938 (86.85%) | Seasonal Naive (0.1477) | Seasonal Naive (0.5257) |
| JPL → Caltech | 3 | positive | 445 (13.15%) | Ridge (0.8408) | Ridge (1.2055) |
| JPL → Caltech | 6 | zero | 2,938 (86.85%) | Seasonal Naive (0.1477) | Seasonal Naive (0.5257) |
| JPL → Caltech | 6 | positive | 445 (13.15%) | Ridge (0.8560) | Ridge (1.2194) |
| JPL → Caltech | 24 | zero | 2,938 (86.85%) | Seasonal Naive (0.1477) | Seasonal Naive (0.5257) |
| JPL → Caltech | 24 | positive | 445 (13.15%) | Ridge (0.9858) | Ridge (1.3881) |

总体 MAE 是两个组 MAE 按样本比例加权。正值小时数量较少，但大误差对 RMSE 的贡献可以更高。零值的含义是没有归档的新会话开始记录，零值表无法确认设备是否在线。

Seasonal Naive 在两个方向所有跨度的零值小时 MAE 最低。正值小时的 MAE 赢家不同，Caltech→JPL 各跨度由 Ridge 领先，JPL→Caltech 的 1h 由 Gradient Boosting 领先，其余跨度由 Ridge 领先。这些结果解释了某个模型可能在正值小时更准，却因零值小时的误差在整体排名落后。

## 非负截断

下表对同一方向的四个提前量取截断前负预测总数与误差变化范围。对非负目标，负预测移至零会减少该小时绝对和平方误差，已非负的预测保持原值。原值与截断值同时保存在逐时预测表中。

| 方向 | 模型 | 负预测总数 | 截断后减原始 MAE | 截断后减原始 RMSE |
|---|---|---:|---|---|
| Caltech → JPL | Ridge | 2,360 | -0.080161 至 -0.015863 | -0.018197 至 -0.001523 |
| Caltech → JPL | Random Forest | 0 | +0.000000 至 +0.000000 | +0.000000 至 +0.000000 |
| Caltech → JPL | Gradient Boosting | 1,206 | -0.007804 至 -0.000976 | -0.001066 至 -0.000031 |
| JPL → Caltech | Ridge | 2,318 | -0.160402 至 +0.000000 | -0.096787 至 +0.000000 |
| JPL → Caltech | Random Forest | 0 | +0.000000 至 +0.000000 | +0.000000 至 +0.000000 |
| JPL → Caltech | Gradient Boosting | 1,897 | -0.021077 至 -0.001406 | -0.001674 至 -0.000156 |

## 局限与复现

结论限于两个归档站点和这一测试窗口。时区未知，共同源时钟标签无法证明处于相同当地时间，日历分布差异需据此解读。持续运营覆盖未知，归档计数不等同于实际车辆到达或用电量。滚动推理假定完整小时计数可及时取得。未用目标测试均值做规模转换或校准。R² 按目标测试均值定义，负值表示较该参考有更高的平方误差。所有结果均为单窗口点估计。

有 4 组 Ridge 独立选参命中 α 网格上界 30,000。结果对应本阶段预定网格，不能断言更强正则化无益。16/24 个原始跨站学习模型组合的测试 R² 为负。

图表由落盘预测和指标重算。实际与预测图以 1h 和 24h 的逐日均值展示两个方向，逐小时结果含全部四个跨度。Python 3.12.10、scikit-learn 1.8.0、pandas 3.0.2、NumPy 2.4.4。重新执行并核对非计时数据与图表哈希：

```powershell
python -B experiments/06_cross_site_training/run_cross_site.py --verify
```
