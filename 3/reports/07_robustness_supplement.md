# Experiment 07 补充验证

原实验保存官方固定窗结果。本补充用两个站点各三段等长滚动测试补充时间外证据。预先固定的三个测试起点为 2020-01-04 00:00:00, 2020-02-15 00:00:00, 2020-03-28 00:00:00（source clock）。每个窗口的历法训练、验证、测试长度分别为 365、42、42 天。三段测试互不重叠，均结束于官方测试前。训练期重叠，故窗口比较仍有依赖。每窗口、站点和跨度所有方法对齐 1,008 个目标小时。Ridge、Random Forest、Gradient Boosting 使用既有训练协议，搜索使用预先固定的原候选子集。

## 各窗口的稳定性

下表列出截断方法相对 Seasonal Naive 改善的跨度数。每个站点–方法的跨窗口计数以 12 个窗口–跨度组合为分母，每个组合独立评分。表中的 R1–R3 列为各窗口的 MAE 改善数。

| 站点 | 方法 | MAE 改善 | RMSE 改善 | R1 MAE | R2 MAE | R3 MAE |
|---|---|---|---|---|---|---|
| Caltech full | Hurdle clipped | 8/12 | 9/12 | 4/4 | 4/4 | 0/4 |
| Caltech full | Residual clipped | 5/12 | 9/12 | 3/4 | 2/4 | 0/4 |
| JPL matched | Hurdle clipped | 7/12 | 8/12 | 4/4 | 3/4 | 0/4 |
| JPL matched | Residual clipped | 8/12 | 8/12 | 4/4 | 4/4 | 0/4 |

Caltech 的逐窗口修正诊断见下表。偏差为预测减真实均值；恶化比例计算截断后残差模型绝对误差超过同一小时季节基准的小时数。

| 窗口 | 跨度 | 修正后偏差 | 季节偏差 | 误差恶化比例 |
|---|---|---|---|---|
| R1 | 1 | -0.0512 | -0.0278 | 65.6% |
| R1 | 3 | +0.0978 | -0.0278 | 54.0% |
| R1 | 6 | +0.1000 | -0.0278 | 54.7% |
| R1 | 24 | +0.0825 | -0.0278 | 66.9% |
| R2 | 1 | +0.1117 | +0.0417 | 68.4% |
| R2 | 3 | +0.1847 | +0.0417 | 59.1% |
| R2 | 6 | +0.2213 | +0.0417 | 61.6% |
| R2 | 24 | +0.2503 | +0.0417 | 71.6% |
| R3 | 1 | +0.1424 | +0.0010 | 94.5% |
| R3 | 3 | +0.1487 | +0.0010 | 87.0% |
| R3 | 6 | +0.2425 | +0.0010 | 92.1% |
| R3 | 24 | +0.5247 | +0.0010 | 97.1% |

各窗口最低误差模型如下。不同窗口的负载水平不同，原始绝对误差应在同一窗口内解释。

| 站点 | 窗口 | 跨度 | 最低 MAE | 最低 RMSE |
|---|---|---|---|---|
| Caltech full | R1 | 1 | Hurdle median | Hurdle clipped |
| Caltech full | R1 | 3 | Hurdle median | Random Forest |
| Caltech full | R1 | 6 | Hurdle median | Hurdle clipped |
| Caltech full | R1 | 24 | Hurdle median | Gradient Boosting |
| JPL matched | R1 | 1 | Residual clipped | Residual clipped |
| JPL matched | R1 | 3 | Residual clipped | Residual clipped |
| JPL matched | R1 | 6 | Residual clipped | Residual clipped |
| JPL matched | R1 | 24 | Residual clipped | Random Forest |
| Caltech full | R2 | 1 | Hurdle median | Random Forest |
| Caltech full | R2 | 3 | Hurdle median | Random Forest |
| Caltech full | R2 | 6 | Hurdle median | Random Forest |
| Caltech full | R2 | 24 | Hurdle median | Hurdle clipped |
| JPL matched | R2 | 1 | Hurdle median | Residual clipped |
| JPL matched | R2 | 3 | Hurdle median | Hurdle clipped |
| JPL matched | R2 | 6 | Hurdle median | Hurdle clipped |
| JPL matched | R2 | 24 | Hurdle median | Hurdle clipped |
| Caltech full | R3 | 1 | Hurdle median | Hurdle median |
| Caltech full | R3 | 3 | Hurdle median | Hurdle median |
| Caltech full | R3 | 6 | Seasonal Naive | Seasonal Naive |
| Caltech full | R3 | 24 | Seasonal Naive | Hurdle median |
| JPL matched | R3 | 1 | Hurdle median | Seasonal Naive |
| JPL matched | R3 | 3 | Seasonal Naive | Seasonal Naive |
| JPL matched | R3 | 6 | Seasonal Naive | Seasonal Naive |
| JPL matched | R3 | 24 | Seasonal Naive | Seasonal Naive |

## 固定窗成对区块重采样

按每一对预测的共同目标键做线性 moving-block bootstrap，每次重采样成对连续块并截取到原长度。每组 2,500 次，主分析为 168h，敏感性为 24h。RMSE 先在每次样本内算均方误差再开方。CSV 记录观测差、bootstrap 差均值、95% 百分位区间及获胜比例。下表仅计完全低于零的区间数。结果依赖当前测试窗口、块长和近似平稳性，多个比较为描述性。

| 设定 | 方法 | 指标 | 168h 区间改善 | 24h 区间改善 |
|---|---|---|---|---|
| Caltech full | Hurdle clipped | MAE | 0/4 | 0/4 |
| Caltech full | Hurdle clipped | RMSE | 2/4 | 1/4 |
| Caltech full | Residual clipped | MAE | 0/4 | 0/4 |
| Caltech full | Residual clipped | RMSE | 1/4 | 1/4 |
| JPL matched | Hurdle clipped | MAE | 2/4 | 1/4 |
| JPL matched | Hurdle clipped | RMSE | 3/4 | 2/4 |
| JPL matched | Residual clipped | MAE | 1/4 | 1/4 |
| JPL matched | Residual clipped | RMSE | 4/4 | 4/4 |
| Caltech matched | Hurdle clipped | MAE | 0/4 | 0/4 |
| Caltech matched | Hurdle clipped | RMSE | 2/4 | 2/4 |
| Caltech matched | Residual clipped | MAE | 0/4 | 0/4 |
| Caltech matched | Residual clipped | RMSE | 1/4 | 1/4 |
| Caltech → JPL | Hurdle clipped | MAE | 1/4 | 1/4 |
| Caltech → JPL | Hurdle clipped | RMSE | 2/4 | 2/4 |
| Caltech → JPL | Residual clipped | MAE | 1/4 | 1/4 |
| Caltech → JPL | Residual clipped | RMSE | 2/4 | 2/4 |
| JPL → Caltech | Hurdle clipped | MAE | 0/4 | 0/4 |
| JPL → Caltech | Hurdle clipped | RMSE | 1/4 | 1/4 |
| JPL → Caltech | Residual clipped | MAE | 0/4 | 0/4 |
| JPL → Caltech | Residual clipped | RMSE | 1/4 | 1/4 |

## MAE 导向的 Hurdle

发生概率沿用相应窗口选出的 Experiment 07 发生分类器。只使用已释放的正需求训练行拟合多类别随机森林，在已观察到的整数支持上估计条件正值概率质量函数。若 p≤0.5，最小中位数为 0。否则使用 q=1−0.5/p，取条件正值 CDF 首次达到 q 的整数。每行验证无条件 CDF 在中位数前小于 0.5、在中位数处达到 0.5。保存所选位置两侧的 CDF。有限样本、训练支持上限和发生概率的校准误差均影响估计。此变体作为分布诊断独立比较，主混合保持原定三个专家。

| 固定窗设定 | Median 胜于期望 MAE | Median 胜于期望 RMSE | Median 胜于 Seasonal Naive MAE |
|---|---|---|---|
| Caltech full | 4/4 | 3/4 | 4/4 |
| JPL matched | 3/4 | 1/4 | 4/4 |
| Caltech → JPL | 4/4 | 3/4 | 4/4 |
| JPL → Caltech | 4/4 | 4/4 | 4/4 |

| 同站 | 真实需求组 | Median 胜于期望 MAE | Median 胜于期望 RMSE |
|---|---|---|---|
| Caltech full | zero | 4/4 | 4/4 |
| Caltech full | positive | 0/4 | 0/4 |
| JPL matched | zero | 4/4 | 4/4 |
| JPL matched | positive | 0/4 | 0/4 |

以下为实际标签严格低于预测中位数及不高于预测中位数的频率。对可校准的离散中位数，0.5 应落在两者之间；分组覆盖只是样本诊断。正预测组样本较少时不宜据此认证条件分布。

| 同站 | 跨度 | 概率组 | 小时 | 实际低于 | 实际不高于 |
|---|---|---|---|---|---|
| Caltech full | 1 | 全体 | 3383 | 0.009 | 0.871 |
| Caltech full | 1 | p>0.5 | 46 | 0.674 | 0.870 |
| Caltech full | 3 | 全体 | 3383 | 0.020 | 0.873 |
| Caltech full | 3 | p>0.5 | 85 | 0.776 | 0.941 |
| Caltech full | 6 | 全体 | 3383 | 0.050 | 0.881 |
| Caltech full | 6 | p>0.5 | 217 | 0.774 | 0.949 |
| Caltech full | 24 | 全体 | 3383 | 0.051 | 0.887 |
| Caltech full | 24 | p>0.5 | 265 | 0.653 | 0.894 |
| JPL matched | 1 | 全体 | 3383 | 0.043 | 0.747 |
| JPL matched | 1 | p>0.5 | 604 | 0.242 | 0.475 |
| JPL matched | 3 | 全体 | 3383 | 0.053 | 0.759 |
| JPL matched | 3 | p>0.5 | 738 | 0.243 | 0.486 |
| JPL matched | 6 | 全体 | 3383 | 0.055 | 0.754 |
| JPL matched | 6 | p>0.5 | 651 | 0.284 | 0.524 |
| JPL matched | 24 | 全体 | 3383 | 0.073 | 0.783 |
| JPL matched | 24 | p>0.5 | 729 | 0.340 | 0.569 |

为辨别零值比例对 MAE 的影响，加入固定输出 0 的事后诊断对照。该对照无需估计。Caltech 同站有 4/4 个跨度的中位数 Hurdle MAE 高于恒零。Caltech 的 p>0.5 组有 4/4 个跨度实际值低于预测中位数的比例超过一半。估计分布的中位数计算已通过验证，经验条件分布在这些子组仍有校准偏差。

| 设定 | 跨度 | 中位数 MAE | 中位数 RMSE | 恒零 MAE | 恒零 RMSE | 完整门 MAE |
|---|---|---|---|---|---|---|
| Caltech full | 1 | 0.182678 | 0.565016 | 0.177949 | 0.565539 | 0.257248 |
| Caltech full | 3 | 0.198049 | 0.597066 | 0.177949 | 0.565539 | 0.293472 |
| Caltech full | 6 | 0.218150 | 0.609803 | 0.177949 | 0.565539 | 0.327065 |
| Caltech full | 24 | 0.203370 | 0.574356 | 0.177949 | 0.565539 | 0.292448 |
| JPL matched | 1 | 0.596512 | 1.350491 | 0.745492 | 1.664428 | 0.528661 |
| JPL matched | 3 | 0.565770 | 1.285219 | 0.745492 | 1.664428 | 0.573223 |
| JPL matched | 6 | 0.587644 | 1.311853 | 0.745492 | 1.664428 | 0.591255 |
| JPL matched | 24 | 0.533550 | 1.207793 | 0.745492 | 1.664428 | 0.618722 |
| Caltech matched | 1 | 0.182974 | 0.567366 | 0.177949 | 0.565539 | 0.254577 |
| Caltech matched | 3 | 0.203665 | 0.624412 | 0.177949 | 0.565539 | 0.289271 |
| Caltech matched | 6 | 0.212238 | 0.609803 | 0.177949 | 0.565539 | 0.306236 |
| Caltech matched | 24 | 0.202187 | 0.572294 | 0.177949 | 0.565539 | 0.290654 |
| Caltech → JPL | 1 | 0.524682 | 1.146780 | 0.745492 | 1.664428 | 0.562580 |
| Caltech → JPL | 3 | 0.623116 | 1.384642 | 0.745492 | 1.664428 | 0.750836 |
| Caltech → JPL | 6 | 0.639965 | 1.448597 | 0.745492 | 1.664428 | 0.669295 |
| Caltech → JPL | 24 | 0.609223 | 1.269134 | 0.745492 | 1.664428 | 0.642616 |
| JPL → Caltech | 1 | 0.183860 | 0.570224 | 0.177949 | 0.565539 | 0.302821 |
| JPL → Caltech | 3 | 0.243275 | 0.639382 | 0.177949 | 0.565539 | 0.404945 |
| JPL → Caltech | 6 | 0.258351 | 0.679065 | 0.177949 | 0.565539 | 0.382196 |
| JPL → Caltech | 24 | 0.213716 | 0.596819 | 0.177949 | 0.565539 | 0.472528 |

两个站点均使用 source clock，时区和实际运营覆盖未知。零值表示零条归档会话。上述结论适用于归档记录及其选择的时间窗口。
