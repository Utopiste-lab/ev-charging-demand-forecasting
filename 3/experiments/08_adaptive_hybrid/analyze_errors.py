"""Render publication artifacts exclusively from saved prediction and metric tables."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_previous_results import FIGURES, METRICS, TABLE, read
from src.adaptive_hybrid.experts import CORE
from src.adaptive_hybrid.hybrid_forecaster import ABLATIONS, ADAPTIVE, EQUAL, RULE, STATIC

SN, HURDLE, RESIDUAL = CORE
BEST = "Best validation expert"
ADAPTED = "Adaptive target-calibrated"
SITE = ["Caltech full", "JPL matched"]
TRANSFER = ["Caltech->JPL", "JPL->Caltech"]
HORIZONS = [1, 3, 6, 24]
MAIN = [SN, HURDLE, RESIDUAL, STATIC, RULE, ADAPTIVE]
PALETTE = ["#444d5f", "#247b76", "#bd6b30", "#7957a5", "#ad4a77", "#225fb0", "#557f37", "#82705f"]
COLORS = dict(zip([*MAIN, ADAPTED, EQUAL], PALETTE))


def title(case):
    return case.replace("->", " → ")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|",
                      *["| " + " | ".join(map(str, r)) + " |" for r in rows]])


def fig_sub(n, size=(11.6, 5.0), rows=1):
    columns = int(np.ceil(n / rows))
    fig, axs = plt.subplots(rows, columns, figsize=size, squeeze=False, layout="constrained")
    return fig, axs.ravel()


def finish(fig, axes, heading, name, folder, *, grid=True):
    fig.suptitle(heading, fontweight="semibold")
    if grid:
        for ax in axes:
            ax.grid(axis="y", alpha=.23, lw=.7)
            ax.set_axisbelow(True)
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return FIGURES / name


def lineplot(scores, models, groups, metric, name, heading, out):
    fig, axes = fig_sub(len(groups), (12, 4.9))
    for ax, case in zip(axes, groups):
        part = scores.loc[scores.case.eq(case)].pivot(index="horizon_hours", columns="model", values=metric)
        for i, model in enumerate(models):
            ax.plot(range(4), part.loc[HORIZONS, model], label=model, color=COLORS.get(model, PALETTE[i % len(PALETTE)]),
                    marker=["o","s","^","D","v","P","X"][i % 7], linewidth=1.7)
        ax.set(title=title(case), xlabel="Forecast horizon", ylabel=f"{metric} (starts/hour)" if metric in ("MAE", "RMSE") else metric)
        ax.set_xticks(range(4), [f"{h}h" for h in HORIZONS])
    axes[-1].legend(fontsize=8, loc="best")
    return finish(fig, axes, heading, name, out)


def figures(out, pred, score, w, wstat, rolling, bootstrap, cross, subgroup):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "semibold", "legend.frameon": False})
    folder = out / FIGURES
    products = [
        lineplot(score, MAIN, SITE, "MAE", "01_mae_by_horizon.png", "Official fixed window · mean absolute error", folder),
        lineplot(score, MAIN, SITE, "RMSE", "02_rmse_by_horizon.png", "Official fixed window · root mean squared error", folder),
        lineplot(score, [EQUAL, STATIC, RULE, ADAPTIVE], SITE, "MAE", "03_static_vs_adaptive.png",
                 "Static and conditional mixtures · fixed-window MAE", folder),
        lineplot(score, [*CORE, ADAPTIVE], SITE, "MAE", "04_experts_vs_hybrid.png", "Experts and full adaptive gate", folder),
    ]
    fig, axes = fig_sub(2, (11.6, 6.0), rows=2)
    for ax, case in zip(axes, SITE):
        sample = w.loc[w.case.eq(case) & w.horizon_hours.eq(1) & w.model.eq(ADAPTIVE)].set_index("target_hour")
        days = sample[["seasonal_weight", "hurdle_weight", "residual_weight"]].resample("D").mean()
        ax.stackplot(days.index, days.to_numpy().T, labels=CORE, colors=[COLORS[x] for x in CORE], alpha=.85)
        ax.set(title=title(case), ylim=(0, 1), ylabel="Daily mean weight")
        ax.xaxis.set_major_formatter(dates.DateFormatter("%b %d"))
    axes[0].legend(ncol=3, loc="upper left", fontsize=8, facecolor="white", frameon=True)
    products.append(finish(fig, axes, "Weight allocation through the test window · 1h", "05_expert_weights_over_time.png", folder))
    fig, axes = fig_sub(2, (11.8, 4.6))
    sw = wstat.loc[wstat.model.eq(ADAPTIVE) & wstat.group_type.eq("overall")]
    for ax, case in zip(axes, SITE):
        cols = sw.loc[sw.case.eq(case)].pivot(index="horizon_hours", columns="expert", values="mean_weight").loc[HORIZONS]
        lower = np.zeros(4)
        for expert in CORE:
            ax.bar(range(4), cols[expert], bottom=lower, color=COLORS[expert], label=expert)
            lower += cols[expert]
        ax.set(title=title(case), ylabel="Mean weight", ylim=(0, 1), xlabel="Forecast horizon")
        ax.set_xticks(range(4), [f"{h}h" for h in HORIZONS])
    axes[-1].legend(loc="upper right", frameon=True)
    products.append(finish(fig, axes, "Average adaptive weights · site and horizon", "06_weights_by_site_horizon.png", folder))
    fig, axes = fig_sub(2, (11.8, 4.8))
    group = wstat.loc[wstat.model.eq(ADAPTIVE) & wstat.group_type.eq("observed_regime_posthoc") & wstat.horizon_hours.eq(1)]
    for ax, case in zip(axes, SITE):
        m = group.loc[group.case.eq(case)].pivot(index="expert", columns="group", values="mean_weight").loc[CORE]
        for i, regime in enumerate(["zero", "positive"]):
            ax.bar(np.arange(3) + (i - 0.5) * .32, m[regime], width=.32, label=regime.capitalize(), color=PALETTE[i])
        ax.set_xticks(range(3), CORE, rotation=13)
        ax.set(title=title(case), ylabel="Mean weight", ylim=(0, 1))
    axes[1].legend()
    products.append(finish(fig, axes, "Expert weights by observed demand · 1h, post-hoc", "07_weights_by_demand.png", folder))
    fig, axes = fig_sub(2, (11.7, 6.2), rows=2)
    for ax, case in zip(axes, SITE):
        sample = pred.loc[pred.case.eq(case) & pred.horizon_hours.eq(1)].set_index("target_hour")
        series = sample[["actual", f"model:{ADAPTIVE}", f"model:{SN}"]].resample("D").mean()
        for col, lab, color in zip(series, ["Actual", ADAPTIVE, SN], ["#202124", COLORS[ADAPTIVE], "#ad673d"]):
            ax.plot(series.index, series[col], label=lab, color=color, linewidth=1.25, alpha=.87)
        ax.set(title=title(case), ylabel="Daily mean starts/hour")
        ax.xaxis.set_major_formatter(dates.DateFormatter("%b %d"))
    axes[0].legend(ncol=3, fontsize=8)
    products.append(finish(fig, axes, "Actual and forecast demand · full 1h test period", "08_actual_vs_predicted.png", folder))
    fig, axes = fig_sub(4, (11.8, 7.0), rows=2)
    for ax, (case, metric) in zip(axes, [(site, field) for site in SITE for field in ("MAE", "RMSE")]):
        r = rolling.loc[rolling.case.eq(case)]
        r = r.pivot(index=["window", "horizon_hours"], columns="model", values=metric)
        for model in [HURDLE, RESIDUAL, STATIC, ADAPTIVE]:
            delta = (r[model] - r[SN]).unstack("horizon_hours")
            ax.plot(range(12), delta.to_numpy().ravel(), label=model, lw=1.35, marker="o", ms=3, color=COLORS[model])
        ax.axhline(0, color="#657080", linewidth=.8)
        ax.set_xticks([1.5, 5.5, 9.5], ["R1", "R2", "R3"])
        ax.set(title=f"{case} · {metric}", ylabel=f"Δ{metric} vs Seasonal Naive")
    axes[0].legend(ncol=2, fontsize=7)
    products.append(finish(fig, axes, "Rolling tests · each group shows 1h, 3h, 6h and 24h", "09_rolling_window_performance.png", folder))
    fig, axes = fig_sub(4, (11.8, 7.3), rows=2)
    for ax, (case, metric) in zip(axes, [(site, field) for site in SITE for field in ("MAE", "RMSE")]):
        block = bootstrap.loc[bootstrap.model.eq(ADAPTIVE) & bootstrap.case.eq(case) & bootstrap.metric.eq(metric) & bootstrap.baseline.eq(SN)]
        for i, length in enumerate([168, 24]):
            r = block.loc[block.block_hours.eq(length)].set_index("horizon_hours").loc[HORIZONS]
            pos = np.arange(4) + (i - 0.5) * .17
            ax.scatter(r.difference, pos, marker="o" if length == 168 else "s", color=PALETTE[i], label=f"{length}h blocks")
            ax.hlines(pos, r.ci_low, r.ci_high, color=PALETTE[i], lw=1.5)
        ax.axvline(0, color="#414957", lw=.8)
        ax.set_yticks(range(4), [f"{h}h" for h in HORIZONS])
        ax.set(title=f"{case} · {metric}", xlabel=f"Adaptive minus seasonal {metric}")
    axes[0].legend(fontsize=8)
    products.append(finish(fig, axes, "Paired moving-block bootstrap · 95% percentile intervals", "10_bootstrap_intervals.png", folder))
    fig, axes = fig_sub(2, (12.2, 5.3))
    for ax, case in zip(axes, SITE):
        r = score.loc[score.case.eq(case)].pivot(index="horizon_hours", columns="model", values="MAE").loc[HORIZONS]
        variants = [*CORE, EQUAL, STATIC, *ABLATIONS]
        matrix = (r[variants].sub(r[ADAPTIVE], axis=0)).to_numpy().T
        limit = max(np.abs(matrix).max(), .001)
        c = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(4), [f"{h}h" for h in HORIZONS])
        ax.set_yticks(range(len(variants)), [v.replace("Adaptive without ", "No ") for v in variants], fontsize=8)
        ax.set_title(case)
        fig.colorbar(c, ax=ax, fraction=.045, label="ΔMAE vs full gate")
    products.append(finish(fig, axes, "Ablations and simple controls · negative values favor the variant", "11_ablation_comparison.png", folder, grid=False))
    fig, axes = fig_sub(2, (11.8, 5.1))
    for ax, case in zip(axes, TRANSFER):
        r = cross.loc[cross.case.eq(case) & cross.model.isin([STATIC, RULE, ADAPTIVE, ADAPTED])]
        for model in [STATIC, RULE, ADAPTIVE, ADAPTED]:
            sample = r.loc[r.model.eq(model)].set_index("horizon_hours").loc[HORIZONS]
            ax.plot(range(4), sample.MAE_gap, color=COLORS[model], label=model, marker="o")
        ax.axhline(0, color="#414957", lw=.8)
        ax.set(title=title(case), ylabel="Transfer MAE minus matched MAE")
        ax.set_xticks(range(4), [f"{h}h" for h in HORIZONS])
    axes[-1].legend(fontsize=8)
    products.append(finish(fig, axes, "Matched transfer gaps · source-only and target pretest calibration", "12_cross_site_transfer.png", folder))
    fig, axes = fig_sub(2, (11.7, 6.4), rows=2)
    for ax, case in zip(axes, SITE):
        sample = pred.loc[pred.case.eq(case) & pred.horizon_hours.eq(1)].sort_values("target_hour").iloc[:168]
        for model in CORE:
            ax.plot(sample.target_hour, sample[f"model:{model}"], color=COLORS[model], alpha=.62, lw=.9, label=model)
        ax.plot(sample.target_hour, sample[f"model:{ADAPTIVE}"], color=COLORS[ADAPTIVE], lw=1.75, label=ADAPTIVE)
        ax.scatter(sample.target_hour, sample.actual, s=7, alpha=.50, color="#202124", label="Actual")
        ax.xaxis.set_major_formatter(dates.DateFormatter("%b %d"))
        ax.set(title=case, ylabel="Recorded starts/hour")
    axes[0].legend(ncol=5, fontsize=7)
    products.append(finish(fig, axes, "Gate examples · first chronological test week, 1h", "13_gate_prediction_examples.png", folder))
    fig, axes = fig_sub(2, (11.8, 4.8))
    for ax, (case, metric) in zip(axes, [(SITE[0], "MAE"), (SITE[1], "MAE")]):
        s = subgroup.loc[subgroup.case.eq(case) & subgroup.horizon_hours.eq(1) & subgroup.window.eq("official")
                         & subgroup.partition.eq("test") & subgroup.model.isin([*CORE, ADAPTIVE])]
        values = s.pivot(index="model", columns="regime", values=metric).loc[[*CORE, ADAPTIVE]]
        for i, group in enumerate(["zero", "positive"]):
            ax.bar(np.arange(4) + (i - .5) * .35, values[group], width=.35, label=group, color=PALETTE[i])
        ax.set_xticks(range(4), values.index, rotation=14)
        ax.set(title=case, ylabel="Mean absolute error (starts/hour)")
    axes[1].legend()
    products.append(finish(fig, axes, "Demand-regime errors · 1h, observed post-hoc", "14_zero_positive_errors.png", folder))
    return products


def improved(frame, method, baseline, metric):
    values = frame.pivot(index=["window", "case", "horizon_hours"], columns="model", values=metric)
    paired = values[[method, baseline]].dropna()
    return f"{(paired[method] < paired[baseline]).sum()}/{len(paired)}"


def median_diagnostics(prediction):
    rows = []
    for (case, horizon), frame in prediction.groupby(["case", "horizon_hours"], sort=False):
        y = frame.actual.to_numpy()
        zero_mae = np.abs(y).mean()
        zero_rmse = np.sqrt(np.square(y.astype(float)).mean())
        error = frame["model:Hurdle median"].to_numpy() - y
        mae, rmse = np.abs(error).mean(), np.sqrt(np.square(error).mean())
        rows.append([title(case), horizon, f"{mae:.6f}", f"{rmse:.6f}", f"{zero_mae:.6f}", f"{zero_rmse:.6f}",
                     f"{np.abs(frame[f'model:{ADAPTIVE}'].to_numpy() - y).mean():.6f}"])
    return table(["设定", "跨度", "中位数 MAE", "中位数 RMSE", "恒零 MAE", "恒零 RMSE", "完整门 MAE"], rows)


def supplement_text(project, rolling, saved, prediction, boot, median, subgroup):
    fixed = boot.loc[boot.suite.eq("Experiment 07") & boot.model.isin([HURDLE, RESIDUAL])]
    md = median.loc[median.partition.eq("test") & median.window.eq("official")]
    diagnostics = []
    for site in SITE:
        s = rolling.loc[rolling.case.eq(site)]
        for method in (HURDLE, RESIDUAL):
            line = [site, method, improved(s, method, SN, "MAE"), improved(s, method, SN, "RMSE")]
            for window in ["R1", "R2", "R3"]:
                line.append(improved(s.loc[s.window.eq(window)], method, SN, "MAE"))
            diagnostics.append(line)
    intervals = []
    for (site, model, metric), g in fixed.groupby(["case","model","metric"], sort=False):
        counts = []
        for length in (168, 24):
            q = g.loc[g.block_hours.eq(length)]
            counts.append(f"{(q.ci_high < 0).sum()}/4")
        intervals.append([title(site), model, metric, *counts])
    medians = []
    for site in SITE + TRANSFER:
        rows = md.loc[md.case.eq(site)]
        if not rows.empty:
            medians.append([title(site), improved(rows, "Hurdle median", HURDLE, "MAE"),
                            improved(rows, "Hurdle median", HURDLE, "RMSE"),
                            improved(rows, "Hurdle median", SN, "MAE")])
    regimes = []
    for site in SITE:
        rows = subgroup.loc[subgroup.case.eq(site) & subgroup.window.eq("official") & subgroup.partition.eq("test")
                            & subgroup.regime.isin(["zero","positive"]) & subgroup.model.isin([HURDLE, "Hurdle median"])]
        for group in ("zero", "positive"):
            g = rows.loc[rows.regime.eq(group)]
            regimes.append([site, group, improved(g, "Hurdle median", HURDLE, "MAE"),
                            improved(g, "Hurdle median", HURDLE, "RMSE")])
    corrections, ranks = [], []
    for (window, case, h), rows in saved.groupby(["window","case","horizon_hours"], sort=False):
        if case == SITE[0]:
            e = rows[f"model:{RESIDUAL}"] - rows.actual
            b = rows[f"model:{SN}"] - rows.actual
            corrections.append([window, h, f"{e.mean():+.4f}", f"{b.mean():+.4f}", f"{(e.abs() > b.abs()).mean():.1%}"])
        scoreboard = rolling.loc[rolling.window.eq(window) & rolling.case.eq(case) & rolling.horizon_hours.eq(h)]
        ranks.append([case, window, h, *(scoreboard.loc[scoreboard[metric].idxmin(), "model"] for metric in ("MAE","RMSE"))])
    empirical = []
    conditional_failures = 0
    for (site, h), sample in prediction.loc[prediction.case.isin(SITE)].groupby(["case","horizon_hours"], sort=False):
        for label, chosen in (("全体", sample), ("p>0.5", sample.loc[sample["Hurdle probability"].gt(0.5)])):
            if len(chosen):
                demand, median_prediction = chosen.actual, chosen["model:Hurdle median"]
                conditional_failures += int(site == SITE[0] and label == "p>0.5"
                                            and demand.lt(median_prediction).mean() > 0.5)
                empirical.append([site, h, label, len(chosen), f"{demand.lt(median_prediction).mean():.3f}",
                                  f"{demand.le(median_prediction).mean():.3f}"])
    caltech_zero = sum((frame["model:Hurdle median"] - frame.actual).abs().mean() > frame.actual.mean()
                       for _, frame in prediction.loc[prediction.case.eq(SITE[0])].groupby("horizon_hours"))
    return f"""# Experiment 07 补充验证

原实验保存官方固定窗结果。本补充用两个站点各三段等长滚动测试补充时间外证据。预先固定的三个测试起点为 {', '.join(project.cfg['rolling']['first_test_starts'])}（source clock）。每个窗口的历法训练、验证、测试长度分别为 365、42、42 天。三段测试互不重叠，均结束于官方测试前。训练期重叠，故窗口比较仍有依赖。每窗口、站点和跨度所有方法对齐 1,008 个目标小时。Ridge、Random Forest、Gradient Boosting 使用既有训练协议，搜索使用预先固定的原候选子集。

## 各窗口的稳定性

下表列出截断方法相对 Seasonal Naive 改善的跨度数。每个站点–方法的跨窗口计数以 12 个窗口–跨度组合为分母，每个组合独立评分。表中的 R1–R3 列为各窗口的 MAE 改善数。

{table(['站点','方法','MAE 改善','RMSE 改善','R1 MAE','R2 MAE','R3 MAE'], diagnostics)}

Caltech 的逐窗口修正诊断见下表。偏差为预测减真实均值；恶化比例计算截断后残差模型绝对误差超过同一小时季节基准的小时数。

{table(['窗口','跨度','修正后偏差','季节偏差','误差恶化比例'], corrections)}

各窗口最低误差模型如下。不同窗口的负载水平不同，原始绝对误差应在同一窗口内解释。

{table(['站点','窗口','跨度','最低 MAE','最低 RMSE'], ranks)}

## 固定窗成对区块重采样

按每一对预测的共同目标键做线性 moving-block bootstrap，每次重采样成对连续块并截取到原长度。每组 2,500 次，主分析为 168h，敏感性为 24h。RMSE 先在每次样本内算均方误差再开方。CSV 记录观测差、bootstrap 差均值、95% 百分位区间及获胜比例。下表仅计完全低于零的区间数。结果依赖当前测试窗口、块长和近似平稳性，多个比较为描述性。

{table(['设定','方法','指标','168h 区间改善','24h 区间改善'], intervals)}

## MAE 导向的 Hurdle

发生概率沿用相应窗口选出的 Experiment 07 发生分类器。只使用已释放的正需求训练行拟合多类别随机森林，在已观察到的整数支持上估计条件正值概率质量函数。若 p≤0.5，最小中位数为 0。否则使用 q=1−0.5/p，取条件正值 CDF 首次达到 q 的整数。每行验证无条件 CDF 在中位数前小于 0.5、在中位数处达到 0.5。保存所选位置两侧的 CDF。有限样本、训练支持上限和发生概率的校准误差均影响估计。此变体作为分布诊断独立比较，主混合保持原定三个专家。

{table(['固定窗设定','Median 胜于期望 MAE','Median 胜于期望 RMSE','Median 胜于 Seasonal Naive MAE'], medians)}

{table(['同站','真实需求组','Median 胜于期望 MAE','Median 胜于期望 RMSE'], regimes)}

以下为实际标签严格低于预测中位数及不高于预测中位数的频率。对可校准的离散中位数，0.5 应落在两者之间；分组覆盖只是样本诊断。正预测组样本较少时不宜据此认证条件分布。

{table(['同站','跨度','概率组','小时','实际低于','实际不高于'], empirical)}

为辨别零值比例对 MAE 的影响，加入固定输出 0 的事后诊断对照。该对照无需估计。Caltech 同站有 {caltech_zero}/4 个跨度的中位数 Hurdle MAE 高于恒零。Caltech 的 p>0.5 组有 {conditional_failures}/4 个跨度实际值低于预测中位数的比例超过一半。估计分布的中位数计算已通过验证，经验条件分布在这些子组仍有校准偏差。

{median_diagnostics(prediction)}

两个站点均使用 source clock，时区和实际运营覆盖未知。零值表示零条归档会话。上述结论适用于归档记录及其选择的时间窗口。
"""


def main_text(score, pred, rolling, weights, subgroup, bootstrap, cross, sensitivity, shift, weeks, trials, project, versions):
    selected = score.loc[score.model.isin([*MAIN, BEST, ADAPTED])]
    main = []
    for (case, h), g in selected.groupby(["case", "horizon_hours"], sort=False):
        r = g.set_index("model")
        def report_value(name):
            return f"{r.at[name, 'MAE']:.4f} / {r.at[name, 'RMSE']:.4f}"
        winners = [r[metric].idxmin() for metric in ("MAE", "RMSE")]
        main.append([title(case), h, *[report_value(c) for c in (SN, BEST, STATIC, RULE, ADAPTIVE)], *winners])
    wt = []
    groups = weights.loc[weights.model.eq(ADAPTIVE) & weights.group_type.isin(["overall","observed_regime_posthoc","probability_regime"])]
    for (case, h), g in groups.groupby(["case", "horizon_hours"], sort=False):
        overall = g.loc[g.group_type.eq("overall")].sort_values("mean_weight", ascending=False)
        r = g.set_index(["group_type","group","expert"])
        def val(kind, group, field):
            return r.at[(kind, group, field), "mean_weight"] if (kind, group, field) in r.index else np.nan
        wt.append([title(case), h, overall.expert.iat[0], f"{overall.mean_weight.iat[0]:.2f}",
                   f"{val('observed_regime_posthoc','zero',SN):.2f}", f"{val('observed_regime_posthoc','positive',SN):.2f}",
                   f"{val('probability_regime','p<=0.5',HURDLE):.2f}", f"{val('probability_regime','p>0.5',HURDLE):.2f}",
                   f"{overall.avg_adjacent_total_variation.iat[0]:.3f}", f"{overall.weight_over_95_fraction.max():.1%}"])
    subgroup = subgroup.loc[subgroup.window.eq("official") & subgroup.partition.eq("test")]
    subgroup_rows = []
    for case in SITE + TRANSFER:
        for name in ("zero", "positive"):
            r = subgroup.loc[subgroup.case.eq(case) & subgroup.regime.eq(name)]
            subgroup_rows.append([title(case), name, improved(r, ADAPTIVE, SN, "MAE"), improved(r, ADAPTIVE, SN, "RMSE")])
    abl = []
    for model in [*CORE, EQUAL, STATIC, *[q for q in ABLATIONS if q != ADAPTIVE]]:
        record = [model]
        for case in SITE:
            sample = score.loc[score.case.eq(case)]
            record += [improved(sample, model, ADAPTIVE, metric) for metric in ("MAE", "RMSE")]
        abl.append(record)
    rolling_summary, interval_summary, gaps = [], [], []
    for site in SITE:
        for window in ["R1", "R2", "R3"]:
            sub = rolling.loc[rolling.case.eq(site) & rolling.window.eq(window)]
            rolling_summary.append([site, window, improved(sub, ADAPTIVE, SN, "MAE"), improved(sub, ADAPTIVE, SN, "RMSE"),
                                    improved(sub, ADAPTIVE, STATIC, "MAE")])
    for case in SITE + TRANSFER:
        rows = bootstrap.loc[bootstrap.case.eq(case) & bootstrap.model.eq(ADAPTIVE)]
        for base in (SN, BEST):
            wins = []
            for length in (168, 24):
                for metric in ("MAE", "RMSE"):
                    condition = rows.loc[rows.baseline.eq(base) & rows.metric.eq(metric) & rows.block_hours.eq(length)]
                    wins.append(f"{(condition.ci_high < 0).sum()}/4")
            interval_summary.append([title(case), base, *wins])
    for row in cross.loc[cross.model.isin([ADAPTIVE, ADAPTED])].itertuples(index=False):
        gaps.append([title(row.case), row.horizon_hours, row.evaluation_mode, f"{row.MAE:.5f}", f"{row.RMSE:.5f}",
                     f"{row.MAE_gap:+.5f}", f"{row.RMSE_gap:+.5f}"])
    ss = sensitivity.loc[sensitivity.partition.eq("test")].copy()
    ss["dimension"] = ss.model.str.split("=").str[0].str.removeprefix("Sensitivity ")
    ranges = ss.groupby(["case","horizon_hours","dimension"]).MAE.agg(lambda g: g.max() - g.min()).unstack()
    drift = shift.loc[shift.window.eq("official") & shift.partition.eq("test_refit")]
    by_site = drift.groupby("case").standardized_mean_change_from_oof.agg(lambda series: series.abs().max())
    weekly = weeks.loc[weeks.window.eq("official") & weeks.model.eq(ADAPTIVE)]
    turns = []
    for (site, h), wk in weekly.groupby(["case","horizon_hours"], sort=False):
        total = wk.loss_difference_sum.sum()
        top = wk.loss_difference_sum.min()
        turns.append([title(site), h, f"{total / wk.n.sum():+.5f}",
                      int(wk.loss_difference_sum.lt(0).sum()), len(wk), "是" if total < 0 <= total - top else "否"])
    complement = []
    for (case, h), sample in pred.groupby(["case", "horizon_hours"], sort=False):
        if case not in SITE:
            continue
        errors = sample[["model:" + name for name in CORE]].sub(sample.actual, axis=0).abs()
        minimum = errors.min(axis=1)
        single = errors.mean().min()
        gate = (sample[f"model:{ADAPTIVE}"] - sample.actual).abs().mean()
        unique = errors.eq(minimum, axis=0)
        strict = unique[unique.sum(axis=1).eq(1)].sum(axis=0) / len(sample)
        complement.append([case, h, f"{single:.4f}", f"{gate:.4f}", f"{minimum.mean():.4f}",
                           *[f"{v:.1%}" for v in strict]])
    median_wins = improved(score, "Hurdle median", ADAPTIVE, "MAE")
    cap = trials.loc[trials.optimizer_status.eq("limit_reached")]
    cap_note = (f"迭代上限影响 {len(cap)} 个候选，其中 {cap.selected.sum()} 个被验证 MAE 选中。"
                "这些候选均在滚动窗口，日志记录了延长上限后的选参和误差对照。") if len(cap) else "所有门控候选在迭代上限内结束。"
    return f"""# Experiment 08 自适应混合充电需求预测

## 研究问题与设定

Experiment 07 显示 Seasonal Naive 的 MAE 强，两个站点零值比例差异明显，Hurdle 和季节修正的排名随站点、跨度和需求组变化。本实验检验样本级门控是否能利用这种互补性。专家为 Seasonal Naive y(t+h−24)、期望 Hurdle pμ 的最终截断版、季节值加带符号残差的最终截断版。保留两种专家的原始值；中位数 Hurdle 作为独立诊断。既有 34 个特征、完成小时释放规则、1/3/6/24h 直接预测及边界排除规则均沿用。

混合预测为 Σᵢwᵢŷᵢ，w≥0 且 Σw=1。等权为固定对照。静态权重在 0.05 间隔的单纯形网格上最小化 OOF MAE 加等权收缩惩罚 λ||w−1/3||²，再按官方验证 MAE 选 λ。规则门控按 Hurdle 的预测概率划分高低组，权重只从对应 OOF 行拟合，阈值由验证 MAE 选择。学习门控用线性 logits 或 6 单元 tanh 层，softmax 输出；输入为专家值、发生概率和已有的历史/日历特征。以 sqrt(e²+0.01²) 平滑绝对误差并加 L2，复杂度和惩罚按精确验证 MAE 选择。线性门从零开始，非线性门使用固定种子。最佳单专家同样按验证 MAE 选出。

## 时间顺序与适配

主 OOF 从训练期 45% 暖启动后开始，划分 4 个预测块。在首块前再划分 504h 前缀验证窗口选择 OOF 专家配置，相关标签均先于首块发布。每块用已释放的更早训练标签扩展拟合，移除 h−1 个未释放标签。门控只拟合 OOF 预测和标签。正式验证专家用训练行拟合，最终专家从头用合规训练加验证标签拟合，门控冻结。另用 3/5 个 OOF 块和种子 {project.cfg['gating_seeds']} 检查敏感性，固定主门架构和超参数。

Caltech 使用完整历史，JPL 及双向迁移使用共同历史；迁移损失用相同日历的 Caltech 或 JPL 同站对照。跨站只用源站拟合与选参，目标站已完成的历史构造输入。单列的 target pretest calibration 仅用目标站已释放的验证标签拟合三个正则化 logit 偏移量。生成该校准输入的专家只拟合源站初始训练，配置在源站验证结束后确定，因此该步骤属于测试发布前的回溯校准。目标测试参与事后评分。

## 固定窗结果

数值为 MAE / RMSE，胜者从此表相关的新方法、专家和对照中选择。每设定均有 3,383 个共同目标小时。CSV 另列 R²、平均预测偏差、零/正值分组、验证分数和相对改善百分比；相对改善的正值代表误差下降。

{table(['设定','跨度','季节','验证最佳专家','静态优化','规则门','完整自适应','最低 MAE','最低 RMSE'], main)}

完整门在固定窗 Caltech 的 MAE 优势集中于 1h，在 JPL 同站四个跨度均降低了三个原始专家中的最小 MAE。JPL 的静态优化和规则门在四个跨度的 RMSE 都低于完整门。在固定窗之外，完整门与静态权重的相对结果可随窗口变化，见下文。

**独立中位数对照。** 主门在本实验预定三个专家之间混合。单独训练的中位数 Hurdle 在 {median_wins} 个固定窗设定–跨度中取得比完整门更低的 MAE。Caltech 同站的恒零对照更低，且 Caltech p>0.5 组的经验中位数覆盖存在偏差。因此同时检查 MAE、RMSE、分组覆盖和正需求误差。详见 Experiment 07 补充报告。

{median_diagnostics(pred)}

## 门控行为与消融

下表比较全体、真实零/正需求（仅事后）及预测概率两组的权重。相邻变化是权重向量 L1 距离除以 2。末列是某个专家获得至少 95% 权重的最大小时占比。均值随具体小时、星期及跨度变化的完整分组分位数在权重汇总表。

{table(['设定','跨度','权重最高专家','均值','季节零值','季节正值','Hurdle 低 p','Hurdle 高 p','相邻变化','≥95%占比'], wt)}

{table(['设定','真实组','自适应改善 MAE 跨度','自适应改善 RMSE 跨度'], subgroup_rows)}

消融用各变体自己的验证 MAE 选参。去除一个专家时，相关预测与 Hurdle 概率也从门输入移除。移除显式历史或日历特征后，专家仍由原始 34 特征产生，因此这些消融针对门控输入。

{table(['变体','Caltech MAE 胜过完整门','Caltech RMSE','JPL MAE','JPL RMSE'], abl)}

Caltech 完整历史设定中，去掉显式日历的门在四个跨度均有更低的 MAE 与 RMSE；JPL 的对应改善仅见于部分跨度。移除 Hurdle 使 JPL 四个跨度的 MAE 和 RMSE 均升高。该结果支持专家互补性与门控特征的作用随站点变化。

各专家互补性可由事后逐行选择的理论参照观察。该参照使用已经揭晓的真实值，不能用于部署或选参。末三列统计该专家在同行独自取得最小绝对误差的小时比例，平局不计入。

{table(['站点','跨度','事后最佳单专家 MAE','完整门 MAE','逐行专家最小 MAE','季节独赢','Hurdle 独赢','残差独赢'], complement)}

## 稳健性与误差集中

{table(['站点','滚动窗口','完整门胜过季节 MAE','完整门胜过季节 RMSE','完整门胜过静态 MAE'], rolling_summary)}

下表统计 95% 区间完全低于零的跨度数。每组 {project.cfg['bootstrap']['replicates']} 次成对移动区块重采样，主块长 168h，24h 为敏感性分析。区间描述当前固定窗时间依赖下的差值，无跨时期保证及多重比较校正。

{table(['设定','基准','168h MAE','168h RMSE','24h MAE','24h RMSE'], interval_summary)}

固定窗各设定–跨度测试 MAE 在种子间的最大极差为 **{ranges['seed'].max():.6f}**，3/4/5 个 OOF 块的最大极差为 **{ranges['oof_blocks'].max():.6f}**。OOF 至最终测试输入的最大绝对标准化均值差（用 OOF 标准差）为 {', '.join(f'{title(site)} {v:.2f}' for site, v in by_site.items())}。该诊断结合了时期/站点负载变化、OOF 拟合样本规模变化和选中专家配置变化，不能归因于单一因素。

{cap_note}

按相邻 168h 划分预测增益，末列表示移除改善最大的周后，全窗 MAE 优势是否消失。测试最后一个不足全周的区块按实际小时计。

{table(['设定','跨度','全窗 ΔMAE','改善周','周数','去掉最好周后反转'], turns)}

## 跨站差值

正差表示误差高于匹配的目标站同站门控。同一源站配对共享冻结的专家与门，校准分支仅增加验证窗估计的偏移。

{table(['方向','跨度','模式','MAE','RMSE','ΔMAE','ΔRMSE'], gaps)}

## 适用边界与后续研究

本研究贡献是把原先的三个预测策略置于严格时间 OOF、可追溯标签释放、跨站迁移和配对误差比较下，识别在哪些条件下简单规则、静态权重与实例门控产生可量化的差异。对同一官方验证窗的分层专家和门选择可能在验证排名上偏乐观，解释以冻结模型的测试结果为据。所有时间沿用 source clock，时区和运营覆盖未知。零值表示未记录到会话，预测假设完成小时的计数及时可用。中位数诊断只能分配训练见过的正计数支持。不同滚动窗口训练期重叠，证据仍局限于两站和观察期。

Experiment 09 可评估分位数回归与 quantile gradient boosting，以 pinball loss 评价分位数，并用分离的时间校准窗检验 conformal 覆盖率及区间宽度。建议按跨度、站点和零/正值分别报告，在跨站漂移下单独检验覆盖。可把校准后的不确定性用作后续门控变量。

复现环境为 {', '.join(f'{key} {v}' for key, v in versions.items())}。官方核对入口为 `experiments/08_adaptive_hybrid/verify_experiment.py`。完整干净重跑为 `evaluate_models.py --verify`，评分专用测试标签会被扰动，已完成历史观测仍随发布时点可用。
"""


def artifacts(out, supplement, project, versions, *, inputs=None, draw=True):
    source = out if inputs is None else inputs
    def metric(name, base=source):
        return read(base / METRICS / (name + ".csv"))
    def tab(name, **options):
        return read(source / TABLE / (name + ".csv"), **options)
    official = metric("adaptive_hybrid_comparison")
    tests = official.loc[official.partition.eq("test")]
    scores = metric("rolling_window_comparison")
    rolling07 = metric("rolling_experiment07_comparison", supplement)
    boot, groups = metric("bootstrap_confidence_intervals"), metric("zero_positive_performance")
    cross, weights = metric("cross_site_adaptive_comparison"), metric("expert_weight_summary")
    predictions = tab("adaptive_hybrid_predictions", parse_dates=["target_hour"])
    pred = predictions.loc[predictions.window.eq("official") & predictions.partition.eq("test")]
    weights = weights.loc[weights.window.eq("official")]
    products = []
    if draw:
        w = tab("gating_weight_predictions", parse_dates=["target_hour"])
        products = figures(out, pred, tests, w.loc[w.window.eq("official")], weights, scores, boot, cross, groups)
    reports = {
        "reports/07_robustness_supplement.md": supplement_text(project, rolling07,
            read(supplement / TABLE / "rolling_experiment07_predictions.csv"), pred, boot, metric("median_hurdle_comparison"), groups),
        "reports/08_adaptive_hybrid.md": main_text(tests, pred, scores, weights, groups, boot, cross, metric("gating_sensitivity"),
                                                 tab("expert_distribution_shift"), tab("weekly_error_contributions"),
                                                 tab("adaptive_hybrid_hyperparameters"), project, versions),
    }
    for name, text in reports.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return products + list(map(out.__class__, reports.keys()))


if __name__ == "__main__":
    from argparse import ArgumentParser
    import platform
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import scipy
    import sklearn
    import yaml

    from audit_previous_results import ROOT, Project

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-reports", action="store_true", help="Refresh only the two Experiment 08 reports from saved tables")
    args = parser.parse_args()
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                "PyYAML": yaml.__version__}
    with TemporaryDirectory(prefix=".exp08-report-", dir=ROOT) as tmp:
        out = Path(tmp)
        for rel in artifacts(out, ROOT, Project(), versions, inputs=ROOT, draw=False):
            if args.refresh_reports:
                (out / rel).replace(ROOT / rel)
            else:
                assert (out / rel).read_text("utf-8") == (ROOT / rel).read_text("utf-8"), rel
    print("Experiment 08 reports refreshed from saved tables." if args.refresh_reports
          else "Both Experiment 08 reports reproduce from saved tables.")
