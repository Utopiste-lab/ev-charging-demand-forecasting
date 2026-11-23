"""Matched transfer gaps, publication figures and results-based interpretation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as dates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


STYLE = {
    "Seasonal Naive": ("Seasonal Naive", "#266883", "o"),
    "Ridge Regression": ("Ridge", "#AE542B", "s"),
    "Random Forest": ("Random Forest", "#267552", "^"),
    "Gradient Boosting": ("Gradient Boosting", "#634D9B", "D"),
}
DIRECTIONS = {"Caltech->JPL": "Caltech → JPL", "JPL->Caltech": "JPL → Caltech"}
HORIZONS = [1, 3, 6, 24]


def analyze(transfer, control, errors):
    keys = ["target_site", "horizon_hours", "model", "variant"]
    metrics = ["test_MAE", "test_RMSE", "test_R2", "mean_prediction_error", "n_test", "test_start", "test_end"]
    paired = transfer[["direction", "source_site", *keys, *metrics]].merge(
        control[[*keys, *metrics]].rename(columns={k: f"same_{k}" for k in metrics}),
        on=keys, validate="one_to_one")
    assert len(paired) == len(transfer) == len(control)
    for k in ["n_test", "test_start", "test_end"]:
        assert paired[k].eq(paired[f"same_{k}"]).all()
    for metric in ["MAE", "RMSE"]:
        original, matched = paired[f"test_{metric}"], paired[f"same_test_{metric}"]
        paired[f"transfer_{metric}_gap"] = original - matched
        paired[f"relative_{metric}_change_pct"] = 100 * (original - matched) / matched.replace(0, np.nan)
    naive = paired.loc[paired.model.eq("Seasonal Naive")]
    assert naive.transfer_MAE_gap.eq(0).all() and naive.transfer_RMSE_gap.eq(0).all()
    groups = []
    for (scenario, direction, h, target, subset), frame in errors.loc[errors.variant.eq("raw")].groupby(
            ["scenario", "direction", "horizon_hours", "target_site", "subset"], sort=False):
        assert frame.n.nunique() == frame.percentage.nunique() == 1
        mae, rmse = frame.loc[frame.MAE.idxmin()], frame.loc[frame.RMSE.idxmin()]
        groups.append({"scenario": scenario, "direction": direction, "horizon_hours": h, "target_site": target,
                       "subset": subset, "n": mae.n, "percentage": mae.percentage,
                       "best_raw_MAE_model": mae.model, "best_raw_MAE": mae.MAE,
                       "best_raw_RMSE_model": rmse.model, "best_raw_RMSE": rmse.RMSE})
    return paired, pd.DataFrame(groups)


def render_figures(folder, transfer, control, paired, errors, signed, test):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True, "grid.alpha": .21,
                         "figure.titlesize": 15})
    x = np.arange(4)
    ticks = [f"{h}h" for h in HORIZONS]

    def save(figure, file):
        figure.savefig(folder / (file + ".png"), dpi=175, facecolor="white")
        plt.close(figure)

    def curve(ax, data, metric, models=STYLE, ls="-"):
        for name in models:
            label, color, marker = STYLE[name]
            points = data.loc[data.model.eq(name)].set_index("horizon_hours").loc[HORIZONS, metric]
            ax.plot(x, points, label=label, color=color, marker=marker, ms=4.5, lw=1.65, ls=ls)
        ax.set(xticks=x, xticklabels=ticks, xlabel="Forecast horizon")
        ax.grid(axis="y")

    for col, metric, name in [("test_MAE", "MAE", "mae"), ("test_RMSE", "RMSE", "rmse"),
                               ("test_R2", "R²", "r2")]:
        fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.3), sharey=True, layout="constrained")
        fig.suptitle(f"Cross-site test {metric} by horizon · Raw forecasts")
        for ax, (direction, label) in zip(axes, DIRECTIONS.items()):
            curve(ax, transfer.loc[transfer.direction.eq(direction)], col)
            ax.set_title(label)
            if metric == "R²":
                ax.axhline(0, color="#777D82", lw=.9, ls=":")
        axes[0].set_ylabel(metric if metric == "R²" else f"{metric} (recorded starts/hour)")
        axes[1].legend(frameon=False, ncols=2, fontsize=8)
        save(fig, f"cross_site_{name}_comparison")

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), sharex=True, layout="constrained")
    fig.suptitle("Transfer and matched same-site reference · Raw forecasts")
    for i, metric in enumerate(["MAE", "RMSE"]):
        for j, (direction, label) in enumerate(DIRECTIONS.items()):
            ax = axes[i, j]
            site = direction.split("->")[1]
            curve(ax, transfer.loc[transfer.direction.eq(direction)], f"test_{metric}")
            curve(ax, control.loc[control.target_site.eq(site) & control.variant.eq("raw")],
                  f"test_{metric}", ls="--")
            ax.set(title=f"{label} and {site} → {site}", ylabel=f"{metric} (starts/hour)")
            if i == 0:
                ax.set_xlabel("")
    handles = [Line2D([0], [0], color=c, marker=m, lw=1.65, label=l) for l, c, m in STYLE.values()]
    handles += [Line2D([0], [0], color="#464B52", ls="-", label="Transfer"),
                Line2D([0], [0], color="#464B52", ls="--", label="Same site")]
    fig.legend(handles=handles, loc="outside lower center", ncols=6, frameon=False, fontsize=8)
    save(fig, "transfer_vs_same_site")

    width = .19
    for (direction, label), filename in zip(DIRECTIONS.items(), ["caltech_to_jpl", "jpl_to_caltech"]):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), layout="constrained")
        fig.suptitle(f"{label} · Cross-site test error by horizon")
        selected = transfer.loc[transfer.direction.eq(direction)]
        for ax, metric in zip(axes, ("MAE", "RMSE")):
            for i, (model, (display, color, _)) in enumerate(STYLE.items()):
                y = selected.loc[selected.model.eq(model)].set_index("horizon_hours").loc[HORIZONS, f"test_{metric}"]
                ax.bar(x + (i - 1.5) * width, y, width=width, color=color, label=display)
            ax.set(xticks=x, xticklabels=ticks, xlabel="Forecast horizon",
                   ylabel=f"{metric} (recorded starts/hour)")
            ax.grid(axis="y")
        axes[1].legend(frameon=False, fontsize=8)
        save(fig, filename)

    raw_gaps = paired.loc[paired.variant.eq("raw")]
    for metric in ("MAE", "RMSE"):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True, layout="constrained")
        fig.suptitle(f"Transfer {metric} gap · Transfer minus matched same-site error")
        for ax, (direction, label) in zip(axes, DIRECTIONS.items()):
            curve(ax, raw_gaps.loc[raw_gaps.direction.eq(direction)], f"transfer_{metric}_gap",
                  models=list(STYLE)[1:])
            ax.axhline(0, color="#5E6871", lw=1, ls=":")
            ax.set_title(label)
        axes[0].set_ylabel(f"Δ{metric} (recorded starts/hour)")
        axes[1].legend(frameon=False, fontsize=8)
        save(fig, f"transfer_{metric.lower()}_gap")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.1), sharex=True, layout="constrained")
    fig.suptitle("Cross-site actual and predicted demand · Daily means of hourly forecasts")
    selected = test.loc[test.scenario.eq("transfer")]
    for i, h in enumerate([1, 24]):
        for j, (direction, label) in enumerate(DIRECTIONS.items()):
            ax = axes[i, j]
            series = selected.loc[selected.direction.eq(direction) & selected.horizon_hours.eq(h)]
            actual = series.loc[series.model.eq("Seasonal Naive")].set_index("target_hour").actual.resample("D").mean()
            ax.plot(actual.index, actual, color="#20242A", lw=1.55, label="Actual")
            for model, (display, color, _) in STYLE.items():
                line = series.loc[series.model.eq(model)].set_index("target_hour").prediction_raw.resample("D").mean()
                ax.plot(line.index, line, color=color, lw=.95, alpha=.90, label=display)
            ax.set(title=f"{label} · {h}h", ylabel="Recorded starts/hour")
            ax.grid(axis="y")
    for ax in axes[-1]:
        ax.set_xlabel("Target day (2020 source clock)")
        ax.xaxis.set_major_locator(dates.MonthLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter("%b"))
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=5, frameon=False, fontsize=8)
    save(fig, "actual_vs_predicted_cross_site")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), sharey=True, layout="constrained")
    fig.suptitle("Cross-site signed prediction bias · Raw forecasts")
    bias = signed.loc[signed.scenario.eq("transfer") & signed.variant.eq("raw")]
    for ax, (direction, label) in zip(axes, DIRECTIONS.items()):
        curve(ax, bias.loc[bias.direction.eq(direction)], "mean_prediction_error")
        ax.axhline(0, color="#777D82", lw=.9, ls=":")
        ax.set_title(label)
    axes[0].set_ylabel("Mean forecast minus actual (starts/hour)")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "prediction_bias")

    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.9), layout="constrained")
    fig.suptitle("Cross-site error on zero-count and positive-count hours · Raw forecasts")
    sub = errors.loc[errors.scenario.eq("transfer") & errors.variant.eq("raw")]
    for i, metric in enumerate(["MAE", "RMSE"]):
        for j, (direction, label) in enumerate(DIRECTIONS.items()):
            for k, subset in enumerate(["zero", "positive"]):
                ax = axes[i, j * 2 + k]
                curve(ax, sub.loc[sub.direction.eq(direction) & sub.subset.eq(subset)], metric)
                ax.set(title=f"{label} · {subset}", ylabel=f"{metric} (starts/hour)")
                if i == 0:
                    ax.set_xlabel("")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncols=4, frameon=False, fontsize=8)
    save(fig, "zero_vs_positive_performance")


def report(raw, clip, same, gaps, errors, bias, subsets, fit, dist, env):
    names = {k: v[0] for k, v in STYLE.items()}
    arrows = {**DIRECTIONS, "Caltech->Caltech": "Caltech → Caltech", "JPL->JPL": "JPL → JPL"}
    def label(row):
        return names[row.model] + (" 截断" if row.variant == "nonnegative" else "")

    tables, winners, gap_rows, bias_rows, subgroup_rows, clip_rows, summary, bias_summary = [], [], [], [], [], [], [], []
    all_transfer = pd.concat([raw, clip], ignore_index=True)
    for direction, arrow in DIRECTIONS.items():
        target = direction.split("->")[1]
        paired = gaps.loc[gaps.direction.eq(direction) & gaps.variant.eq("raw")]
        paired_learned = paired.loc[~paired.model.eq("Seasonal Naive")]
        summary.append(
            f"- **{arrow}** 的 12 个学习模型原始组合中，{paired_learned.transfer_MAE_gap.lt(0).sum()} 个 MAE "
            f"低于同站对照，{paired_learned.transfer_RMSE_gap.lt(0).sum()} 个 RMSE 低于同站对照。"
            f"MAE 差范围 {paired_learned.transfer_MAE_gap.min():+.4f} 至 "
            f"{paired_learned.transfer_MAE_gap.max():+.4f}，RMSE 差范围 "
            f"{paired_learned.transfer_RMSE_gap.min():+.4f} 至 {paired_learned.transfer_RMSE_gap.max():+.4f}。")
        for h in HORIZONS:
            tm = all_transfer.loc[all_transfer.direction.eq(direction) & all_transfer.horizon_hours.eq(h)]
            controls = same.loc[same.target_site.eq(target) & same.horizon_hours.eq(h)]
            m, r = tm.loc[tm.test_MAE.idxmin()], tm.loc[tm.test_RMSE.idxmin()]
            sm, sr = controls.loc[controls.test_MAE.idxmin()], controls.loc[controls.test_RMSE.idxmin()]
            winners.append(f"| {arrow} | {h} | {label(m)} ({m.test_MAE:.4f}) | {label(r)} ({r.test_RMSE:.4f}) | "
                           f"{label(sm)} ({sm.test_MAE:.4f}) | {label(sr)} ({sr.test_RMSE:.4f}) |")
        for row in paired.itertuples():
            tables.append(f"| {arrow} | {row.horizon_hours} | {names[row.model]} | {row.test_MAE:.6f} | "
                          f"{row.test_RMSE:.6f} | {row.test_R2:.6f} | {row.same_test_MAE:.6f} | {row.same_test_RMSE:.6f} | "
                          f"{row.same_test_R2:.6f} |")
            gap_rows.append(f"| {arrow} | {row.horizon_hours} | {names[row.model]} | {row.transfer_MAE_gap:+.6f} | "
                            f"{row.relative_MAE_change_pct:+.2f}% | {row.transfer_RMSE_gap:+.6f} | "
                            f"{row.relative_RMSE_change_pct:+.2f}% |")
        signed = bias.loc[bias.direction.eq(direction) & bias.variant.eq("raw") & ~bias.model.eq("Seasonal Naive")]
        bias_summary.append(
            f"- **{arrow}** 的学习模型中，{signed.mean_prediction_error.gt(0).sum()}/12 个组合高估，"
            f"{signed.mean_prediction_error.lt(0).sum()}/12 个组合低估。"
            f"原始偏差范围为 {signed.mean_prediction_error.min():+.4f} 至 "
            f"{signed.mean_prediction_error.max():+.4f} 次/小时。")
        for model, rows in signed.groupby("model", sort=False):
            items = rows.sort_values("horizon_hours").mean_prediction_error
            bias_rows.append(f"| {arrow} | {names[model]} | " + " | ".join(f"{v:+.6f}" for v in items) + " |")
        for row in subsets.loc[subsets.direction.eq(direction)].itertuples():
            subgroup_rows.append(f"| {arrow} | {row.horizon_hours} | {row.subset} | {row.n:,} ({row.percentage:.2f}%) | "
                                 f"{names[row.best_raw_MAE_model]} ({row.best_raw_MAE:.4f}) | "
                                 f"{names[row.best_raw_RMSE_model]} ({row.best_raw_RMSE:.4f}) |")
        base = raw.loc[raw.direction.eq(direction)].set_index(["horizon_hours", "model"])
        for model, block in clip.loc[clip.direction.eq(direction)].groupby("model", sort=False):
            changes = [(row.test_MAE - base.at[(row.horizon_hours, model), "test_MAE"],
                        row.test_RMSE - base.at[(row.horizon_hours, model), "test_RMSE"])
                       for row in block.itertuples()]
            clip_rows.append(f"| {arrow} | {names[model]} | {block.negative_raw_predictions.sum():,} | "
                             f"{min(x[0] for x in changes):+.6f} 至 {max(x[0] for x in changes):+.6f} | "
                             f"{min(x[1] for x in changes):+.6f} 至 {max(x[1] for x in changes):+.6f} |")
    sites = dist.loc[dist.scope.eq("test_targets") & dist.group_type.eq("overall")].set_index("site")
    sizes = fit.groupby("horizon_hours").n_final_refit.first()
    largest = gaps.loc[gaps.variant.eq("raw")].sort_values("transfer_MAE_gap", ascending=False).iloc[0]
    best = gaps.loc[gaps.variant.eq("raw")].sort_values("transfer_MAE_gap").iloc[0]
    worst_rmse = gaps.loc[gaps.variant.eq("raw")].sort_values("transfer_RMSE_gap", ascending=False).iloc[0]
    learned = raw.loc[~raw.model.eq("Seasonal Naive")]
    selection = fit.loc[fit.model.eq("Ridge Regression")]
    upper = selection.loc[selection.selected_hyperparameters.str.contains('"alpha": 30000.0', regex=False)]
    return f"""# Caltech 与 JPL 跨站预测及匹配同站对照

## 结论

{chr(10).join(summary)}

原始变体中，最大 MAE 损失是 {arrows[largest.direction]} {largest.horizon_hours}h 的 {names[largest.model]}（{largest.transfer_MAE_gap:+.6f}），最低差值出现在 {arrows[best.direction]} {best.horizon_hours}h 的 {names[best.model]}（{best.transfer_MAE_gap:+.6f}）。最大 RMSE 损失是 {arrows[worst_rmse.direction]} {worst_rmse.horizon_hours}h 的 {names[worst_rmse.model]}（{worst_rmse.transfer_RMSE_gap:+.6f}，相对 {worst_rmse.relative_RMSE_change_pct:+.2f}%）。两个方向呈现不同的迁移代价。负值说明迁移在该目标窗口的这一指标低于匹配同站误差。

## 设计与公平性

两方向与两组匹配同站对照使用 2018-10-08 13:00 至 2020-12-02 23:00 的共同源时钟历史，以及完全相同的 3,383 个测试目标小时，范围为 2020-07-15 00:00 至 2020-12-02 22:00。模型在 1、3、6、24h 分别独立拟合，最终样本数依次为 {"、".join(f"{n:,}" for n in sizes)}。每组的三个学习模型均在同站训练段拟合八个候选，按该站原始验证 MAE 选参，然后从头拟合可用的训练及验证数据。两个同站对照独立选参及拟合，未将其结果反馈给迁移模型。因为某些设置使用同一站点的相同样本、网格和种子，独立训练得到同样参数属于预期。

迁移的训练标签仅来自源站。模型冻结后，以目标站在发布时已完成的历史作输入。目标站的 Seasonal Naive 取目标小时减 24 小时的该站记录，并在该目标的迁移与同站对照中复用。日历特征始终使用目标源时钟。Ridge 标准化只在对应训练站的拟合行内学习，树模型保留计数尺度。预测冻结后才连接测试真值。此实验使用目标历史的滚动起点，严格无目标历史的 zero-shot 将需要另外的协议。

核实了审计中的可用样本、每处分界按跨度剔除 0/2/5/23 小时、最终拟合时间戳及标签哈希、目标键和 34 列 schema。扰动首个目标测试观测，48 个已冻结模型在该观测可用之前的预测均逐行不变，对之后预测的实际影响已记录。两站正式同站参照均使用与迁移相同的共同历史，已发布的完整 Caltech 同站结果只提供背景。

## 两个方向的原始结果

| 方向 | h | 模型 | 迁移 MAE | 迁移 RMSE | 迁移 R² | 同站 MAE | 同站 RMSE | 同站 R² |
|---|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(tables)}

下表在各自比较中纳入适用的截断变体，MAE 与 RMSE 分别排序。

| 方向 | h | 迁移最低 MAE | 迁移最低 RMSE | 匹配同站最低 MAE | 匹配同站最低 RMSE |
|---|---:|---|---|---|---|
{chr(10).join(winners)}

Caltech→JPL 的 1h 迁移可优于目标站 Seasonal Naive，且 24h 的 Gradient Boosting 在 RMSE 上也领先季节基准。3h 和 6h 的总体赢家是目标站 Seasonal Naive。JPL→Caltech 的 1h Gradient Boosting 在 RMSE 上优于 Seasonal Naive，同时其 MAE 仍较高。较长跨度两个指标均由 Seasonal Naive 领先。三个学习模型在两个方向的 1h 原始 RMSE 都比 3h 和 6h 低。Caltech→JPL 三个学习模型的 24h 原始 RMSE 又都低于各自 6h，误差随跨度变化并不单调。

## 迁移代价

差值为迁移减同站，正值表示误差上升，百分比以同站误差为分母。下表列原始变体。CSV 另外包含三个学习模型的截断配对。Seasonal Naive 在对应目标站两种协议的差为零。

| 方向 | h | 模型 | ΔMAE | 相对 MAE | ΔRMSE | 相对 RMSE |
|---|---:|---|---:|---:|---:|---:|
{chr(10).join(gap_rows)}

判断哪个模型迁移更稳健时，应同时看绝对误差和匹配差值。较小的迁移差值也可能来自较高的同站误差。两方向的目标站分布不同，直接比较两方向的绝对 MAE 会混合两个站点的计数规模。比较差值和相对变化能回答各自目标站上更换训练来源的影响。负差值可能反映两站训练时期的模式差异、有限样本或窗口特定效应，现有设计无法确定具体原因。

## 偏差与分布

共同测试期 Caltech 平均 {sites.at['Caltech','mean']:.4f} 次/小时，JPL 为 {sites.at['JPL','mean']:.4f}。标准差分别为 {sites.at['Caltech','std_ddof1']:.4f} 和 {sites.at['JPL','std_ddof1']:.4f}。在此计数差异下，Caltech→JPL 低估和 JPL→Caltech 高估是可检验的方向性假设。下表报告每个学习模型的有符号偏差，正值表示高估。偏差的方向以各行结果为准，数据规模差异本身不构成因果解释。

| 方向 | 模型 | 1h | 3h | 6h | 24h |
|---|---|---:|---:|---:|---:|
{chr(10).join(bias_rows)}

{chr(10).join(bias_summary)}

Caltech→JPL 的低估假设未获得这组原始预测支持。JPL→Caltech 多数组合高估，尤其是 3h 及 6h 的 Ridge。模型同时使用站点历史和历法，本实验没有单独干预规模，不能把这些偏差直接归因为两站均值差异。

## 零值与正值小时

每一目标站的四个跨度共享相同的实际标签。分组用于事后解释，未参与选参。完整误差分组表包含迁移、同站和截断版本。

| 方向 | h | 实测组 | 小时与比例 | 最低原始 MAE | 最低原始 RMSE |
|---|---:|---|---|---|---|
{chr(10).join(subgroup_rows)}

总体 MAE 是两个组 MAE 按样本比例加权。正值小时数量较少，但大误差对 RMSE 的贡献可以更高。零值的含义是没有归档的新会话开始记录，零值表无法确认设备是否在线。

Seasonal Naive 在两个方向所有跨度的零值小时 MAE 最低。正值小时的 MAE 赢家不同，Caltech→JPL 各跨度由 Ridge 领先，JPL→Caltech 的 1h 由 Gradient Boosting 领先，其余跨度由 Ridge 领先。这些结果解释了某个模型可能在正值小时更准，却因零值小时的误差在整体排名落后。

## 非负截断

下表对同一方向的四个提前量取截断前负预测总数与误差变化范围。对非负目标，负预测移至零会减少该小时绝对和平方误差，已非负的预测保持原值。原值与截断值同时保存在逐时预测表中。

| 方向 | 模型 | 负预测总数 | 截断后减原始 MAE | 截断后减原始 RMSE |
|---|---|---:|---|---|
{chr(10).join(clip_rows)}

## 局限与复现

结论限于两个归档站点和这一测试窗口。时区未知，共同源时钟标签无法证明处于相同当地时间，日历分布差异需据此解读。持续运营覆盖未知，归档计数不等同于实际车辆到达或用电量。滚动推理假定完整小时计数可及时取得。未用目标测试均值做规模转换或校准。R² 按目标测试均值定义，负值表示较该参考有更高的平方误差。所有结果均为单窗口点估计。

有 {len(upper)} 组 Ridge 独立选参命中 α 网格上界 30,000。结果对应本阶段预定网格，不能断言更强正则化无益。{learned.test_R2.lt(0).sum()}/{len(learned)} 个原始跨站学习模型组合的测试 R² 为负。

图表由落盘预测和指标重算。实际与预测图以 1h 和 24h 的逐日均值展示两个方向，逐小时结果含全部四个跨度。Python {env['python']}、scikit-learn {env['sklearn']}、pandas {env['pandas']}、NumPy {env['numpy']}。重新执行并核对非计时数据与图表哈希：

```powershell
python -B experiments/06_cross_site_training/run_cross_site.py --verify
```
"""
