# Experiment 08 自适应混合充电需求预测

## 研究问题与设定

Experiment 07 显示 Seasonal Naive 的 MAE 强，两个站点零值比例差异明显，Hurdle 和季节修正的排名随站点、跨度和需求组变化。本实验检验样本级门控是否能利用这种互补性。专家为 Seasonal Naive y(t+h−24)、期望 Hurdle pμ 的最终截断版、季节值加带符号残差的最终截断版。保留两种专家的原始值；中位数 Hurdle 作为独立诊断。既有 34 个特征、完成小时释放规则、1/3/6/24h 直接预测及边界排除规则均沿用。

混合预测为 Σᵢwᵢŷᵢ，w≥0 且 Σw=1。等权为固定对照。静态权重在 0.05 间隔的单纯形网格上最小化 OOF MAE 加等权收缩惩罚 λ||w−1/3||²，再按官方验证 MAE 选 λ。规则门控按 Hurdle 的预测概率划分高低组，权重只从对应 OOF 行拟合，阈值由验证 MAE 选择。学习门控用线性 logits 或 6 单元 tanh 层，softmax 输出；输入为专家值、发生概率和已有的历史/日历特征。以 sqrt(e²+0.01²) 平滑绝对误差并加 L2，复杂度和惩罚按精确验证 MAE 选择。线性门从零开始，非线性门使用固定种子。最佳单专家同样按验证 MAE 选出。

## 时间顺序与适配

主 OOF 从训练期 45% 暖启动后开始，划分 4 个预测块。在首块前再划分 504h 前缀验证窗口选择 OOF 专家配置，相关标签均先于首块发布。每块用已释放的更早训练标签扩展拟合，移除 h−1 个未释放标签。门控只拟合 OOF 预测和标签。正式验证专家用训练行拟合，最终专家从头用合规训练加验证标签拟合，门控冻结。另用 3/5 个 OOF 块和种子 [7, 19, 42] 检查敏感性，固定主门架构和超参数。

Caltech 使用完整历史，JPL 及双向迁移使用共同历史；迁移损失用相同日历的 Caltech 或 JPL 同站对照。跨站只用源站拟合与选参，目标站已完成的历史构造输入。单列的 target pretest calibration 仅用目标站已释放的验证标签拟合三个正则化 logit 偏移量。生成该校准输入的专家只拟合源站初始训练，配置在源站验证结束后确定，因此该步骤属于测试发布前的回溯校准。目标测试参与事后评分。

## 固定窗结果

数值为 MAE / RMSE，胜者从此表相关的新方法、专家和对照中选择。每设定均有 3,383 个共同目标小时。CSV 另列 R²、平均预测偏差、零/正值分组、验证分数和相对改善百分比；相对改善的正值代表误差下降。

| 设定 | 跨度 | 季节 | 验证最佳专家 | 静态优化 | 规则门 | 完整自适应 | 最低 MAE | 最低 RMSE |
|---|---|---|---|---|---|---|---|---|
| Caltech full | 1 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.2841 / 0.5524 | 0.2814 / 0.7106 | 0.2572 / 0.5779 | Adaptive full | Static optimized |
| Caltech full | 3 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.3476 / 0.6883 | 0.3331 / 0.7226 | 0.2935 / 0.6667 | Seasonal Naive | Adaptive full |
| Caltech full | 6 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.3819 / 0.6579 | 0.3508 / 0.6827 | 0.3271 / 0.6681 | Seasonal Naive | Static optimized |
| Caltech full | 24 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.3309 / 0.6402 | 0.2779 / 0.7026 | 0.2924 / 0.6326 | Rule gate | Adaptive full |
| JPL matched | 1 | 0.6654 / 1.3886 | 0.5614 / 1.0393 | 0.5462 / 1.0096 | 0.5289 / 1.0105 | 0.5287 / 1.0314 | Adaptive full | Static optimized |
| JPL matched | 3 | 0.6654 / 1.3886 | 0.6654 / 1.3886 | 0.6317 / 1.0621 | 0.5626 / 1.0542 | 0.5732 / 1.1096 | Rule gate | Rule gate |
| JPL matched | 6 | 0.6654 / 1.3886 | 0.6654 / 1.3886 | 0.6456 / 1.0782 | 0.6240 / 1.0789 | 0.5913 / 1.1405 | Adaptive full | Static optimized |
| JPL matched | 24 | 0.6654 / 1.3886 | 0.6654 / 1.3886 | 0.7351 / 1.1327 | 0.6868 / 1.1185 | 0.6187 / 1.1730 | Adaptive full | Rule gate |
| Caltech matched | 1 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.2929 / 0.5524 | 0.2822 / 0.5864 | 0.2546 / 0.5720 | Adaptive full | Static optimized |
| Caltech matched | 3 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.3541 / 0.7101 | 0.3436 / 0.7494 | 0.2893 / 0.6718 | Seasonal Naive | Adaptive full |
| Caltech matched | 6 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.3811 / 0.7131 | 0.3471 / 0.7104 | 0.3062 / 0.6642 | Seasonal Naive | Adaptive full |
| Caltech matched | 24 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.3587 / 0.6420 | 0.2791 / 0.7048 | 0.2907 / 0.6277 | Seasonal Naive | Adaptive full |
| Caltech → JPL | 1 | 0.6654 / 1.3886 | 0.6654 / 1.3886 | 0.5903 / 1.0592 | 0.5813 / 1.0746 | 0.5626 / 1.0949 | Adaptive full | Static optimized |
| Caltech → JPL | 3 | 0.6654 / 1.3886 | 0.6654 / 1.3886 | 0.7773 / 1.4564 | 0.7857 / 1.5271 | 0.7508 / 1.5180 | Seasonal Naive | Seasonal Naive |
| Caltech → JPL | 6 | 0.6654 / 1.3886 | 0.6654 / 1.3886 | 0.7059 / 1.2954 | 0.7040 / 1.3402 | 0.6693 / 1.3039 | Adaptive target-calibrated | Static optimized |
| Caltech → JPL | 24 | 0.6654 / 1.3886 | 0.6654 / 1.3886 | 0.7033 / 1.2410 | 0.6376 / 1.2975 | 0.6426 / 1.2436 | Adaptive target-calibrated | Residual clipped |
| JPL → Caltech | 1 | 0.2782 / 0.7124 | 0.3424 / 0.6254 | 0.3373 / 0.5899 | 0.2947 / 0.5683 | 0.3028 / 0.5906 | Seasonal Naive | Rule gate |
| JPL → Caltech | 3 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.6430 / 0.9991 | 0.5090 / 0.9111 | 0.4049 / 0.6746 | Seasonal Naive | Adaptive target-calibrated |
| JPL → Caltech | 6 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.6372 / 0.9993 | 0.5734 / 0.9137 | 0.3822 / 0.6495 | Seasonal Naive | Adaptive target-calibrated |
| JPL → Caltech | 24 | 0.2782 / 0.7124 | 0.2782 / 0.7124 | 0.7845 / 1.1803 | 0.7237 / 1.1794 | 0.4725 / 0.8336 | Seasonal Naive | Seasonal Naive |

完整门在固定窗 Caltech 的 MAE 优势集中于 1h，在 JPL 同站四个跨度均降低了三个原始专家中的最小 MAE。JPL 的静态优化和规则门在四个跨度的 RMSE 都低于完整门。在固定窗之外，完整门与静态权重的相对结果可随窗口变化，见下文。

**独立中位数对照。** 主门在本实验预定三个专家之间混合。单独训练的中位数 Hurdle 在 19/20 个固定窗设定–跨度中取得比完整门更低的 MAE。Caltech 同站的恒零对照更低，且 Caltech p>0.5 组的经验中位数覆盖存在偏差。因此同时检查 MAE、RMSE、分组覆盖和正需求误差。详见 Experiment 07 补充报告。

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

## 门控行为与消融

下表比较全体、真实零/正需求（仅事后）及预测概率两组的权重。相邻变化是权重向量 L1 距离除以 2。末列是某个专家获得至少 95% 权重的最大小时占比。均值随具体小时、星期及跨度变化的完整分组分位数在权重汇总表。

| 设定 | 跨度 | 权重最高专家 | 均值 | 季节零值 | 季节正值 | Hurdle 低 p | Hurdle 高 p | 相邻变化 | ≥95%占比 |
|---|---|---|---|---|---|---|---|---|---|
| Caltech full | 1 | Seasonal Naive | 0.79 | 0.81 | 0.69 | 0.11 | 0.33 | 0.102 | 3.5% |
| Caltech full | 3 | Seasonal Naive | 0.77 | 0.78 | 0.67 | 0.17 | 0.39 | 0.117 | 14.2% |
| Caltech full | 6 | Seasonal Naive | 0.77 | 0.79 | 0.67 | 0.15 | 0.59 | 0.127 | 25.1% |
| Caltech full | 24 | Seasonal Naive | 0.74 | 0.77 | 0.60 | 0.15 | 0.58 | 0.132 | 2.5% |
| JPL matched | 1 | Seasonal Naive | 0.44 | 0.53 | 0.26 | 0.23 | 0.37 | 0.124 | 0.0% |
| JPL matched | 3 | Hurdle clipped | 0.38 | 0.41 | 0.27 | 0.37 | 0.42 | 0.161 | 4.1% |
| JPL matched | 6 | Seasonal Naive | 0.43 | 0.43 | 0.43 | 0.33 | 0.37 | 0.169 | 3.7% |
| JPL matched | 24 | Seasonal Naive | 0.53 | 0.61 | 0.35 | 0.22 | 0.33 | 0.209 | 11.7% |
| Caltech matched | 1 | Seasonal Naive | 0.78 | 0.80 | 0.65 | 0.11 | 0.37 | 0.111 | 21.3% |
| Caltech matched | 3 | Seasonal Naive | 0.81 | 0.82 | 0.73 | 0.13 | 0.33 | 0.097 | 22.8% |
| Caltech matched | 6 | Seasonal Naive | 0.79 | 0.80 | 0.71 | 0.14 | 0.46 | 0.098 | 28.1% |
| Caltech matched | 24 | Seasonal Naive | 0.75 | 0.77 | 0.60 | 0.13 | 0.48 | 0.122 | 3.2% |
| Caltech → JPL | 1 | Seasonal Naive | 0.58 | 0.72 | 0.25 | 0.13 | 0.56 | 0.180 | 13.2% |
| Caltech → JPL | 3 | Seasonal Naive | 0.59 | 0.70 | 0.36 | 0.19 | 0.54 | 0.181 | 6.7% |
| Caltech → JPL | 6 | Seasonal Naive | 0.60 | 0.67 | 0.44 | 0.22 | 0.53 | 0.170 | 8.7% |
| Caltech → JPL | 24 | Seasonal Naive | 0.59 | 0.68 | 0.39 | 0.16 | 0.56 | 0.170 | 1.0% |
| JPL → Caltech | 1 | Seasonal Naive | 0.51 | 0.52 | 0.41 | 0.23 | 0.38 | 0.122 | 0.0% |
| JPL → Caltech | 3 | Seasonal Naive | 0.38 | 0.40 | 0.30 | 0.36 | 0.32 | 0.156 | 4.2% |
| JPL → Caltech | 6 | Seasonal Naive | 0.45 | 0.46 | 0.38 | 0.32 | 0.20 | 0.175 | 4.5% |
| JPL → Caltech | 24 | Seasonal Naive | 0.59 | 0.60 | 0.48 | 0.21 | 0.28 | 0.204 | 15.9% |

| 设定 | 真实组 | 自适应改善 MAE 跨度 | 自适应改善 RMSE 跨度 |
|---|---|---|---|
| Caltech full | zero | 1/4 | 4/4 |
| Caltech full | positive | 4/4 | 4/4 |
| JPL matched | zero | 1/4 | 4/4 |
| JPL matched | positive | 4/4 | 4/4 |
| Caltech → JPL | zero | 1/4 | 4/4 |
| Caltech → JPL | positive | 3/4 | 3/4 |
| JPL → Caltech | zero | 0/4 | 2/4 |
| JPL → Caltech | positive | 4/4 | 4/4 |

消融用各变体自己的验证 MAE 选参。去除一个专家时，相关预测与 Hurdle 概率也从门输入移除。移除显式历史或日历特征后，专家仍由原始 34 特征产生，因此这些消融针对门控输入。

| 变体 | Caltech MAE 胜过完整门 | Caltech RMSE | JPL MAE | JPL RMSE |
|---|---|---|---|---|
| Seasonal Naive | 3/4 | 0/4 | 0/4 | 0/4 |
| Hurdle clipped | 0/4 | 1/4 | 0/4 | 0/4 |
| Residual clipped | 0/4 | 1/4 | 0/4 | 0/4 |
| Static equal | 0/4 | 2/4 | 0/4 | 3/4 |
| Static optimized | 0/4 | 2/4 | 0/4 | 4/4 |
| Adaptive without Hurdle | 1/4 | 0/4 | 0/4 | 0/4 |
| Adaptive without Residual | 3/4 | 2/4 | 3/4 | 0/4 |
| Adaptive without history | 0/4 | 2/4 | 2/4 | 4/4 |
| Adaptive without calendar | 4/4 | 4/4 | 2/4 | 3/4 |

Caltech 完整历史设定中，去掉显式日历的门在四个跨度均有更低的 MAE 与 RMSE；JPL 的对应改善仅见于部分跨度。移除 Hurdle 使 JPL 四个跨度的 MAE 和 RMSE 均升高。该结果支持专家互补性与门控特征的作用随站点变化。

各专家互补性可由事后逐行选择的理论参照观察。该参照使用已经揭晓的真实值，不能用于部署或选参。末三列统计该专家在同行独自取得最小绝对误差的小时比例，平局不计入。

| 站点 | 跨度 | 事后最佳单专家 MAE | 完整门 MAE | 逐行专家最小 MAE | 季节独赢 | Hurdle 独赢 | 残差独赢 |
|---|---|---|---|---|---|---|---|
| Caltech full | 1 | 0.2782 | 0.2572 | 0.1464 | 79.3% | 11.4% | 8.2% |
| Caltech full | 3 | 0.2782 | 0.2935 | 0.1493 | 67.9% | 17.5% | 2.2% |
| Caltech full | 6 | 0.2782 | 0.3271 | 0.1447 | 58.9% | 15.4% | 3.7% |
| Caltech full | 24 | 0.2782 | 0.2924 | 0.1716 | 70.7% | 10.9% | 6.8% |
| JPL matched | 1 | 0.5614 | 0.5287 | 0.2831 | 69.6% | 20.5% | 9.9% |
| JPL matched | 3 | 0.6160 | 0.5732 | 0.3097 | 55.8% | 22.1% | 9.7% |
| JPL matched | 6 | 0.6418 | 0.5913 | 0.3092 | 56.6% | 21.6% | 9.9% |
| JPL matched | 24 | 0.6654 | 0.6187 | 0.3478 | 56.9% | 18.4% | 12.7% |

## 稳健性与误差集中

| 站点 | 滚动窗口 | 完整门胜过季节 MAE | 完整门胜过季节 RMSE | 完整门胜过静态 MAE |
|---|---|---|---|---|
| Caltech full | R1 | 4/4 | 4/4 | 4/4 |
| Caltech full | R2 | 4/4 | 4/4 | 4/4 |
| Caltech full | R3 | 0/4 | 2/4 | 4/4 |
| JPL matched | R1 | 4/4 | 4/4 | 3/4 |
| JPL matched | R2 | 4/4 | 4/4 | 4/4 |
| JPL matched | R3 | 0/4 | 0/4 | 4/4 |

下表统计 95% 区间完全低于零的跨度数。每组 2500 次成对移动区块重采样，主块长 168h，24h 为敏感性分析。区间描述当前固定窗时间依赖下的差值，无跨时期保证及多重比较校正。

| 设定 | 基准 | 168h MAE | 168h RMSE | 24h MAE | 24h RMSE |
|---|---|---|---|---|---|
| Caltech full | Seasonal Naive | 1/4 | 4/4 | 1/4 | 2/4 |
| Caltech full | Best validation expert | 1/4 | 4/4 | 1/4 | 2/4 |
| JPL matched | Seasonal Naive | 4/4 | 4/4 | 4/4 | 4/4 |
| JPL matched | Best validation expert | 4/4 | 3/4 | 4/4 | 3/4 |
| Caltech → JPL | Seasonal Naive | 2/4 | 3/4 | 1/4 | 3/4 |
| Caltech → JPL | Best validation expert | 2/4 | 3/4 | 1/4 | 3/4 |
| JPL → Caltech | Seasonal Naive | 0/4 | 2/4 | 0/4 | 2/4 |
| JPL → Caltech | Best validation expert | 1/4 | 2/4 | 1/4 | 2/4 |

固定窗各设定–跨度测试 MAE 在种子间的最大极差为 **0.000958**，3/4/5 个 OOF 块的最大极差为 **0.012235**。OOF 至最终测试输入的最大绝对标准化均值差（用 OOF 标准差）为 Caltech full 1.10, Caltech matched 0.99, Caltech → JPL 0.56, JPL matched 0.48, JPL → Caltech 0.71。该诊断结合了时期/站点负载变化、OOF 拟合样本规模变化和选中专家配置变化，不能归因于单一因素。

迭代上限影响 4 个候选，其中 3 个被验证 MAE 选中。这些候选均在滚动窗口，日志记录了延长上限后的选参和误差对照。

按相邻 168h 划分预测增益，末列表示移除改善最大的周后，全窗 MAE 优势是否消失。测试最后一个不足全周的区块按实际小时计。

| 设定 | 跨度 | 全窗 ΔMAE | 改善周 | 周数 | 去掉最好周后反转 |
|---|---|---|---|---|---|
| Caltech full | 1 | -0.02091 | 18 | 21 | 否 |
| Caltech full | 3 | +0.01532 | 5 | 21 | 否 |
| Caltech full | 6 | +0.04891 | 2 | 21 | 否 |
| Caltech full | 24 | +0.01429 | 3 | 21 | 否 |
| JPL matched | 1 | -0.13672 | 21 | 21 | 否 |
| JPL matched | 3 | -0.09216 | 19 | 21 | 否 |
| JPL matched | 6 | -0.07413 | 18 | 21 | 否 |
| JPL matched | 24 | -0.04666 | 18 | 21 | 否 |
| Caltech matched | 1 | -0.02358 | 19 | 21 | 否 |
| Caltech matched | 3 | +0.01112 | 6 | 21 | 否 |
| Caltech matched | 6 | +0.02808 | 4 | 21 | 否 |
| Caltech matched | 24 | +0.01250 | 3 | 21 | 否 |
| Caltech → JPL | 1 | -0.10281 | 20 | 21 | 否 |
| Caltech → JPL | 3 | +0.08545 | 2 | 21 | 否 |
| Caltech → JPL | 6 | +0.00391 | 9 | 21 | 否 |
| Caltech → JPL | 24 | -0.02277 | 15 | 21 | 否 |
| JPL → Caltech | 1 | +0.02467 | 7 | 21 | 否 |
| JPL → Caltech | 3 | +0.12679 | 1 | 21 | 否 |
| JPL → Caltech | 6 | +0.10404 | 1 | 21 | 否 |
| JPL → Caltech | 24 | +0.19437 | 0 | 21 | 否 |

## 跨站差值

正差表示误差高于匹配的目标站同站门控。同一源站配对共享冻结的专家与门，校准分支仅增加验证窗估计的偏移。

| 方向 | 跨度 | 模式 | MAE | RMSE | ΔMAE | ΔRMSE |
|---|---|---|---|---|---|---|
| Caltech → JPL | 1 | source-only transfer | 0.56258 | 1.09492 | +0.03392 | +0.06357 |
| Caltech → JPL | 3 | source-only transfer | 0.75084 | 1.51798 | +0.17761 | +0.40843 |
| Caltech → JPL | 6 | source-only transfer | 0.66930 | 1.30386 | +0.07804 | +0.16340 |
| Caltech → JPL | 24 | source-only transfer | 0.64262 | 1.24356 | +0.02389 | +0.07054 |
| JPL → Caltech | 1 | source-only transfer | 0.30282 | 0.59056 | +0.04824 | +0.01860 |
| JPL → Caltech | 3 | source-only transfer | 0.40494 | 0.67462 | +0.11567 | +0.00279 |
| JPL → Caltech | 6 | source-only transfer | 0.38220 | 0.64954 | +0.07596 | -0.01471 |
| JPL → Caltech | 24 | source-only transfer | 0.47253 | 0.83364 | +0.18187 | +0.20595 |
| Caltech → JPL | 1 | target pretest calibration | 0.56265 | 1.09992 | +0.03399 | +0.06856 |
| Caltech → JPL | 3 | target pretest calibration | 0.73858 | 1.50704 | +0.16536 | +0.39748 |
| Caltech → JPL | 6 | target pretest calibration | 0.65617 | 1.30413 | +0.06491 | +0.16367 |
| Caltech → JPL | 24 | target pretest calibration | 0.63581 | 1.24489 | +0.01709 | +0.07187 |
| JPL → Caltech | 1 | target pretest calibration | 0.29857 | 0.59781 | +0.04400 | +0.02584 |
| JPL → Caltech | 3 | target pretest calibration | 0.35706 | 0.65760 | +0.06779 | -0.01422 |
| JPL → Caltech | 6 | target pretest calibration | 0.33468 | 0.63799 | +0.02844 | -0.02626 |
| JPL → Caltech | 24 | target pretest calibration | 0.41141 | 0.78588 | +0.12076 | +0.15819 |

## 适用边界与后续研究

本研究贡献是把原先的三个预测策略置于严格时间 OOF、可追溯标签释放、跨站迁移和配对误差比较下，识别在哪些条件下简单规则、静态权重与实例门控产生可量化的差异。对同一官方验证窗的分层专家和门选择可能在验证排名上偏乐观，解释以冻结模型的测试结果为据。所有时间沿用 source clock，时区和运营覆盖未知。零值表示未记录到会话，预测假设完成小时的计数及时可用。中位数诊断只能分配训练见过的正计数支持。不同滚动窗口训练期重叠，证据仍局限于两站和观察期。

Experiment 09 可评估分位数回归与 quantile gradient boosting，以 pinball loss 评价分位数，并用分离的时间校准窗检验 conformal 覆盖率及区间宽度。建议按跨度、站点和零/正值分别报告，在跨站漂移下单独检验覆盖。可把校准后的不确定性用作后续门控变量。

复现环境为 python 3.12.10, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8, PyYAML 6.0.3。官方核对入口为 `experiments/08_adaptive_hybrid/verify_experiment.py`。完整干净重跑为 `evaluate_models.py --verify`，评分专用测试标签会被扰动，已完成历史观测仍随发布时点可用。
