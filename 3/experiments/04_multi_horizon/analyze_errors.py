"""Post-hoc subgroup analysis, figures, and research report."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import score


STYLE = {
    "Seasonal Naive": ("Seasonal Naive", "#24657D", "o"),
    "Ridge Regression": ("Ridge", "#A65328", "s"),
    "Random Forest": ("Random Forest", "#287557", "^"),
    "Gradient Boosting": ("Gradient Boosting", "#655099", "D"),
}
FIGURE_ROOT = "results/figures/multi_horizon"
FIGURE_NAMES = [
    "mae_vs_horizon.png", "rmse_vs_horizon.png", "r2_vs_horizon.png", "model_comparison_by_horizon.png",
    *[f"actual_vs_predicted_{h}h.png" for h in (1, 3, 6, 24)],
    "zero_vs_positive_mae.png", "zero_vs_positive_rmse.png",
]


def error_breakdown(predictions):
    test = predictions.loc[predictions.partition.eq("test")]
    rows = []
    for (h, model, variant), data in test.groupby(["horizon_hours", "model", "prediction_variant"], sort=False):
        for group, mask in [("zero", data.actual.eq(0)), ("positive", data.actual.gt(0))]:
            part = data.loc[mask]
            metrics = score(part.actual, part.prediction)
            rows.append({
                "horizon_hours": h, "model": model, "prediction_variant": variant, "group": group,
                "n": len(part), "n_test": len(data), "percentage_of_test": 100 * len(part) / len(data),
                "MAE": metrics["MAE"], "RMSE": metrics["RMSE"],
            })
    detailed = pd.DataFrame(rows)
    totals = []
    primary = detailed.loc[detailed.prediction_variant.eq("raw")]
    for (h, group), block in primary.groupby(["horizon_hours", "group"], sort=False):
        for column in ["n", "n_test", "percentage_of_test"]:
            assert block[column].nunique() == 1
        top = block.iloc[0]
        mae, rmse = block.loc[block.MAE.idxmin()], block.loc[block.RMSE.idxmin()]
        totals.append({"horizon_hours": h, "group": group, "n": top.n, "n_test": top.n_test,
                       "percentage_of_test": top.percentage_of_test, "best_raw_MAE_model": mae.model,
                       "best_raw_MAE": mae.MAE, "best_raw_RMSE_model": rmse.model, "best_raw_RMSE": rmse.RMSE})
    return detailed, pd.DataFrame(totals)


def create_figures(directory, primary, clipped, predictions, subgroup, horizons):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True, "grid.alpha": .22,
                         "figure.titlesize": 15, "axes.titlesize": 11})
    axis_x = np.arange(len(horizons))
    ticks = [f"{h}h" for h in horizons]

    def save(fig, filename):
        fig.savefig(directory / filename, dpi=180, facecolor="white")
        plt.close(fig)

    for col, metric, file in [("test_MAE", "MAE", "mae_vs_horizon.png"),
                              ("test_RMSE", "RMSE", "rmse_vs_horizon.png"),
                              ("test_R2", "R²", "r2_vs_horizon.png")]:
        fig, ax = plt.subplots(figsize=(8.6, 4.4), layout="constrained")
        for model, (label, color, marker) in STYLE.items():
            y = primary.loc[primary.model.eq(model)].set_index("horizon_hours").loc[horizons, col]
            ax.plot(axis_x, y, color=color, marker=marker, lw=1.8, label=label)
        ax.set(xticks=axis_x, xticklabels=ticks, xlabel="Forecast horizon",
               ylabel=metric if col == "test_R2" else f"{metric} (recorded starts/hour)",
               title=f"Caltech test {metric} by forecast horizon · Raw forecasts")
        ax.grid(axis="y")
        ax.legend(loc="best", ncols=2, frameon=False)
        if col == "test_R2":
            ax.axhline(0, color="#6B7179", lw=.8, ls=":")
        save(fig, file)

    fig, panels = plt.subplots(2, len(horizons), figsize=(13, 6.2), sharex="row", layout="constrained")
    fig.suptitle("Caltech model comparison by horizon · Raw forecasts")
    for i, metric in enumerate(("MAE", "RMSE")):
        maximum = primary[f"test_{metric}"].max()
        for j, h in enumerate(horizons):
            ax = panels[i, j]
            values = primary.loc[primary.horizon_hours.eq(h)].set_index("model").loc[list(STYLE), f"test_{metric}"]
            bars = ax.barh(range(4), values, color=[item[1] for item in STYLE.values()])
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
            ax.set_yticks(range(4), [item[0] for item in STYLE.values()] if j == 0 else [""] * 4)
            ax.invert_yaxis()
            ax.set_xlim(0, maximum * 1.28)
            ax.set(title=f"{h}h · {metric}", xlabel="Recorded starts/hour")
            ax.grid(axis="x")
    save(fig, "model_comparison_by_horizon.png")

    test = predictions.loc[predictions.partition.eq("test")]
    for h in horizons:
        fig, panels = plt.subplots(2, 2, figsize=(11.5, 6.0), sharex=True, sharey=True, layout="constrained")
        fig.suptitle(f"Caltech {h}-hour forecasts · Daily means of all held-out hourly targets")
        for ax, (model, (label, color, _)) in zip(panels.flat, STYLE.items()):
            blocks = test.loc[test.horizon_hours.eq(h) & test.model.eq(model)]
            raw = blocks.loc[blocks.prediction_variant.eq("raw")].set_index("target_hour")
            daily = raw[["actual", "prediction"]].resample("D").mean()
            ax.plot(daily.index, daily.actual, color="#272C34", label="Actual", lw=1.2)
            ax.plot(daily.index, daily.prediction, color=color, label="Raw", lw=1.1)
            if (clipped.horizon_hours.eq(h) & clipped.model.eq(model)).any():
                extra = blocks.loc[blocks.prediction_variant.eq("nonnegative")].set_index("target_hour")
                daily_clip = extra.prediction.resample("D").mean()
                ax.plot(daily_clip.index, daily_clip, color=color, ls="--", label="Clipped", lw=1.)
            ax.set_title(label, loc="left")
            ax.grid(axis="y")
            ax.legend(loc="upper left", frameon=False, ncols=3, fontsize=8)
        for ax in panels[:, 0]:
            ax.set_ylabel("Recorded starts/hour")
        for ax in panels[-1]:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.set_xlabel("Target day (2020 source clock)")
        save(fig, f"actual_vs_predicted_{h}h.png")

    sub = subgroup.loc[subgroup.prediction_variant.eq("raw")]
    for metric in ("MAE", "RMSE"):
        fig, panels = plt.subplots(1, 2, figsize=(11.2, 4.5), layout="constrained")
        fig.suptitle(f"Caltech {metric} on zero-count and positive-count hours · Raw forecasts")
        for ax, group in zip(panels, ("zero", "positive")):
            rows = sub.loc[sub.group.eq(group)]
            for model, (label, color, marker) in STYLE.items():
                values = rows.loc[rows.model.eq(model)].set_index("horizon_hours").loc[horizons, metric]
                ax.plot(axis_x, values, color=color, marker=marker, lw=1.55, label=label)
            ax.set(xticks=axis_x, xticklabels=ticks, xlabel="Forecast horizon",
                   ylabel=f"{metric} (recorded starts/hour)", title=f"{group.capitalize()} observed count")
            ax.grid(axis="y")
        panels[1].legend(loc="best", frameon=False, fontsize=8)
        save(fig, f"zero_vs_positive_{metric.lower()}.png")


def research_report(primary, clipped, split, fit, trials, subset, subset_summary, reference, env):
    labels = {key: value[0] for key, value in STYLE.items()}
    horizons = sorted(primary.horizon_hours.unique())
    all_metrics = pd.concat([primary, clipped], ignore_index=True)

    def display(row):
        return labels[row.model] + (" 截断" if row.prediction_variant == "nonnegative" else "")

    scores, winners, patterns, dem, zeros, clip_rows, params, gaps, rf_gb = [], [], [], [], [], [], [], [], []
    for h in horizons:
        raw = primary.loc[primary.horizon_hours.eq(h)].set_index("model")
        mixed = all_metrics.loc[all_metrics.horizon_hours.eq(h)]
        for model in STYLE:
            r = raw.loc[model]
            scores.append(f"| {h} | {labels[model]} | {r.test_MAE:.6f} | {r.test_RMSE:.6f} | {r.test_R2:.6f} |")
        winners.append(f"| {h} | {labels[raw.test_MAE.idxmin()]} ({raw.test_MAE.min():.6f}) | "
                       f"{labels[raw.test_RMSE.idxmin()]} ({raw.test_RMSE.min():.6f}) | "
                       f"{display(mixed.loc[mixed.test_MAE.idxmin()])} | {display(mixed.loc[mixed.test_RMSE.idxmin()])} |")
        count = split.loc[split.horizon_hours.eq(h)].set_index("partition")
        dem.append(f"| {h} | {count.at['train','eligible_n']:,} | {count.at['validation','eligible_n']:,} | "
                   f"{fit.loc[fit.horizon_hours.eq(h),'n_final_refit'].iloc[0]:,} | "
                   f"{count.at['test','eligible_n']:,} | {count.at['test','first_issue']} | "
                   f"{count.at['train','excluded_label_release']} / {count.at['validation','excluded_label_release']} |")
        for r in subset_summary.loc[subset_summary.horizon_hours.eq(h)].itertuples():
            zeros.append(f"| {h} | {r.group} | {r.n:,} | {r.percentage_of_test:.2f}% | "
                         f"{labels[r.best_raw_MAE_model]} ({r.best_raw_MAE:.6f}) | "
                         f"{labels[r.best_raw_RMSE_model]} ({r.best_raw_RMSE:.6f}) |")
        for r in clipped.loc[clipped.horizon_hours.eq(h)].itertuples():
            clip_rows.append(f"| {h} | {labels[r.model]} | {r.raw_test_negative_predictions:,} | "
                             f"{r.delta_test_MAE:+.6f} | {r.delta_test_RMSE:+.6f} |")
        for r in fit.loc[fit.horizon_hours.eq(h)].itertuples():
            js = json.loads(r.selected_hyperparameters)
            parameters = "、".join(f"{k}={v}" for k, v in js.items()
                                   if k not in {"loss", "n_iter_no_change", "random_state", "n_jobs", "solver"})
            params.append(f"| {h} | {labels[r.model]} | {parameters} | {r.final_training_timestamps_sha256[:12]} |")
        for model in list(STYLE)[1:]:
            options = trials.loc[trials.horizon_hours.eq(h) & trials.model.eq(model)]
            chosen = options.loc[options.selected].iloc[0]
            best_train = options.loc[options.train_MAE.idxmin()]
            gaps.append(f"| {h} | {labels[model]} | {chosen.train_MAE:.6f} | {chosen.validation_MAE:.6f} | "
                        f"{best_train.validation_MAE:.6f} |")
        rf, gb = raw.loc[["Random Forest", "Gradient Boosting"]].itertuples(index=False)
        rf_gb.append(f"| {h} | {gb.test_MAE-rf.test_MAE:+.6f} | {gb.test_RMSE-rf.test_RMSE:+.6f} |")
    for model, data in primary.groupby("model", sort=False):
        seq = data.sort_values("horizon_hours")
        patterns.append(f"- **{labels[model]}** 的 MAE 为 " + " → ".join(f"{v:.4f}" for v in seq.test_MAE)
                        + "，RMSE 为 " + " → ".join(f"{v:.4f}" for v in seq.test_RMSE) + "。")
    relative = []
    ridge = primary.loc[primary.model.eq("Ridge Regression")].set_index("horizon_hours")
    for model in ("Random Forest", "Gradient Boosting"):
        tree = primary.loc[primary.model.eq(model)].set_index("horizon_hours")
        for metric in ("MAE", "RMSE"):
            wins = [str(h) for h in horizons if tree.at[h, f"test_{metric}"] < ridge.at[h, f"test_{metric}"]]
            relative.append(f"- {labels[model]} 的原始 {metric} 低于原始 Ridge 的提前量为 "
                            + ("、".join(wins) + " 小时。" if wins else "无。"))
    return f"""# Caltech 多时间跨度充电会话开始数预测

## 实验口径

目标为源时钟内每小时记录到的会话开始次数。令 t 为最后一个已完成小时的起点，在 t+1h 发布 y(t+h) 的预测。采用逐小时滚动起点和直接预测，每个 h 单独训练。使用同一套 34 列，其中 4 列需求历史锚定 t，30 列日历特征由目标小时 t+h 决定。Seasonal Naive 取 y(t+h−24)。测试期已完成且已可用的真实观测可供后续起点使用。

复用 Phase 3 边界及自动化验证。筛选按目标小时，剔除涉及不完整边界小时的样本，保留原视图的零值与峰值。各模型在相同初始训练段训练候选，仅用未截断的验证 MAE 选参。每个提前量重新选参，沿用正式 1h 的 8 个 Ridge alpha、8 组 RF 及 8 组 GB 设置。配置包含原 1h 优选设置和固定种子 42。最终三个模型用相同的训练与验证时间戳重新拟合，Ridge 标准化也重新拟合。测试标签未用于拟合或选择。

| h | 初始训练 | 验证 | 最终拟合 | 测试 | 首个测试发布时间 | 分界剔除 训练 / 验证 |
|---:|---:|---:|---:|---:|---|---:|
{chr(10).join(dem)}

测试目标范围为 2020-07-15 00:00 至 2020-12-02 22:00。经逐键核对，各提前量实际选出的目标小时完全相同。时间戳哈希、历史边界剔除与预热数量见切分表。正式 1h 结果的逐行最大预测绝对差为 {reference:.3g}，指标、候选选取、最终拟合时间戳均通过参考检查。

## 原始预测与模型排名

| h | 模型 | MAE | RMSE | R² |
|---:|---|---:|---:|---:|
{chr(10).join(scores)}

| h | 原始 MAE 最低 | 原始 RMSE 最低 | 含截断 MAE 最低 | 含截断 RMSE 最低 |
|---:|---|---|---|---|
{chr(10).join(winners)}

{chr(10).join(patterns)}

Seasonal Naive 的参考始终是目标小时之前 24 小时的观测。对相同目标小时，不同提前量的这个参考值相同，且在各发布时点已可用。因此它的误差曲线保持平坦。学习模型的短期历史与目标间距随 h 改变，并且候选选择和可用训练样本也随 h 改变。

{chr(10).join(relative)}

下表用原始 Gradient Boosting 误差减去原始 Random Forest 误差。负值表示 Gradient Boosting 更低，可以直接检查其 1h 优势在较长提前量是否持续。

| h | GB − RF MAE | GB − RF RMSE |
|---:|---:|---:|
{chr(10).join(rf_gb)}

R² 的零点以测试窗口实际均值作参照，负值表示平方误差高于该参照。该参照均值用来定义指标，不是可部署的测试期间预测规则。

## 零值、正值和截断

切片根据事后实测目标定义，只用于解释误差。下表显示样本比例和原始模型的切片赢家。各模型的完整 MAE、RMSE、样本数与占比在 `horizon_error_breakdown.csv`，包含适用的截断变体。

| h | 子集 | 小时 | 占比 | 原始 MAE 最低 | 原始 RMSE 最低 |
|---:|---|---:|---:|---|---|
{chr(10).join(zeros)}

所有提前量匹配同一组实际值，因此零值比例相同。总体 MAE 是两个子集 MAE 按小时比例的加权平均，整体 RMSE 则对分组平方误差先加权再开方。零值占比较高时，零值误差对总体 MAE 排名权重较大，正值小时的较大误差仍可显著影响 RMSE。

| h | 模型 | 截断前负测试预测 | 截断后 − 原始 MAE | 截断后 − 原始 RMSE |
|---:|---|---:|---:|---:|
{chr(10).join(clip_rows)}

对于非负实测值，把负预测截为零会在该点减少绝对与平方误差。表中只列有负验证或测试预测的模型与提前量。Random Forest 的逐行原始预测也已保存，可核查负值数量。截断规则固定，未参与选参。

## 选参与局限

| h | 模型 | 选中参数 | 最终拟合时间戳 SHA-256 前缀 |
|---:|---|---|---|
{chr(10).join(params)}

| h | 模型 | 选中候选训练 MAE | 选中候选验证 MAE | 训练误差最低候选的验证 MAE |
|---:|---|---:|---:|---:|
{chr(10).join(gaps)}

最小训练误差的候选与最小验证误差的候选若不同，说明进一步降低样本内误差未带来相同的后期收益。训练、验证和测试的需求分布不同，且正式测试使用训练加验证重拟合。绝对误差差距无法单独归因为过拟合。指标只覆盖一个站点的一个测试窗口，没有提供跨站或跨时期的不确定性估计。归档覆盖状态和时区均未知，零值表示未记录新会话，结果依赖完整小时计数在发布时及时可用的假设。

## 复现

依赖沿用正式一小时版本，运行环境为 Python {env['python']}、scikit-learn {env['sklearn']}、pandas {env['pandas']}、NumPy {env['numpy']}、Matplotlib {env['matplotlib']}。

```powershell
python -B experiments/04_multi_horizon/run_multi_horizon.py --verify
```

全部图表依据本阶段落盘的预测与统计表重画。为保留既有一小时图，10 张图统一放在 `results/figures/multi_horizon/`。趋势、模型比较和分组误差图展示四个原始模型，逐日实际与预测图另外标出适用的截断预测。分组图的两个子图纵轴独立，数值大小以坐标轴和 CSV 为准。实验范围限于 Caltech。
"""
