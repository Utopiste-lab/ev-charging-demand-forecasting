"""Analyze saved forecasts, diagnostics, and matched transfer gaps."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.advanced_forecasting.evaluation import demand_metrics, occurrence_metrics


FIGURE_NAMES = (
    "mae_across_horizons.png", "rmse_across_horizons.png", "zero_vs_positive.png",
    "hurdle_classification.png", "hurdle_calibration.png", "seasonal_vs_residual_correction.png",
    "residual_distribution.png", "prediction_bias.png", "same_vs_cross_site.png", "actual_vs_predicted.png",
)
KEY = ["case", "horizon_hours", "partition", "target_hour"]
HORIZONS = [1, 3, 6, 24]
STYLE = {
    "Seasonal Naive": ("Seasonal Naive", "#255F7D"),
    "Ridge Regression": ("Ridge", "#9D562F"),
    "Random Forest": ("Random Forest", "#2D785A"),
    "Gradient Boosting": ("Gradient Boosting", "#69539C"),
    "Hurdle": ("Hurdle", "#B7425E"),
    "Seasonal Residual": ("Seasonal Residual", "#7B7224"),
}
FOCUS = ["Caltech full", "JPL matched", "Caltech->JPL", "JPL->Caltech"]
SCOPE = {**dict(zip(FOCUS, ["Caltech · full", "JPL · matched", "Caltech → JPL", "JPL → Caltech"])),
         "Caltech matched": "Caltech · matched"}


def label(model, variant="raw"):
    name = STYLE[model][0]
    return name + (" 截断" if variant == "nonnegative" else "")


def analyze(saved, hurdle, residual, config, archive, archived_refs):
    metric_rows, groups, classification, magnitudes, corrections, calibration = [], [], [], [], [], []
    for (case, h), new in saved.groupby(["case", "horizon_hours"], sort=False):
        ref = archived_refs(*archive, case, h)
        exemplar = new.iloc[0]
        meta = {k: exemplar[k] for k in ["phase", "case", "source_site", "target_site", "horizon_hours"]}
        exemplar_rows = new.loc[new.model.eq("Hurdle")]
        expected = exemplar_rows.set_index(["partition", "target_hour"]).actual
        for (model, variant), old in ref.groupby(["model", "variant"], sort=False):
            pd.testing.assert_series_equal(old.set_index(["partition", "target_hour"]).actual, expected)
        old_rows = ref[["partition", "target_hour", "actual", "model", "variant", "prediction", "reference_file"]]
        methods = [old_rows]
        for v in ("raw", "nonnegative"):
            methods.append(new[["partition", "target_hour", "actual", "model", f"prediction_{v}"]]
                           .rename(columns={f"prediction_{v}": "prediction"}).assign(variant=v, reference_file="Experiment 07"))
        all_models = pd.concat(methods, ignore_index=True)
        for (name, v), block in all_models.groupby(["model", "variant"], sort=False):
            tag = {**meta, "model": name, "variant": v}
            record = {}
            for part, rows in block.groupby("partition"):
                record.update({f"{part}_{metric}": value for metric, value in
                               demand_metrics(rows.actual, rows.prediction).items()})
                record[f"n_{part}"] = len(rows)
                if part == "test":
                    for group, mask in [("zero", rows.actual.eq(0)), ("positive", rows.actual.gt(0))]:
                        result = demand_metrics(rows.loc[mask, "actual"], rows.loc[mask, "prediction"])
                        groups.append({**tag, "group": group, "n": int(mask.sum()), "percentage": 100 * mask.mean(),
                                       **result})
                        record.update({f"{group}_{k}": v for k, v in result.items() if k in {"MAE", "RMSE"}})
            metric_rows.append({**tag, **record, "prediction_reference": block.reference_file.iat[0]})
        components = exemplar_rows.merge(hurdle.loc[hurdle.case.eq(case) & hurdle.horizon_hours.eq(h)],
                                         on=KEY, validate="one_to_one")
        for partition, rows in components.groupby("partition"):
            z = rows.actual.gt(0)
            kind = {**meta, "partition": partition}
            classification.append({**kind, **occurrence_metrics(rows.actual, rows.probability, config["probability_threshold"])})
            magnitudes.append({**kind, "n_positive": int(z.sum()),
                               **demand_metrics(rows.loc[z, "actual"], rows.loc[z, "conditional_magnitude_raw"]),
                               **{f"final_{k}": v for k, v in demand_metrics(rows.loc[z, "actual"],
                                  rows.loc[z, "prediction_nonnegative"]).items()}})
            if partition == "test":
                bins = config["calibration_bins"]
                labels = np.minimum((rows.probability * bins).astype(int), bins - 1)
                for i, part in rows.assign(bin=labels).groupby("bin"):
                    calibration.append({**meta, "bin": i, "n": len(part),
                                        "mean_probability": part.probability.mean(), "observed_positive_rate": part.actual.gt(0).mean()})
        rrows = new.loc[new.model.eq("Seasonal Residual") & new.partition.eq("test")].merge(
            residual.loc[residual.case.eq(case) & residual.horizon_hours.eq(h) & residual.partition.eq("test")],
            on=KEY, validate="one_to_one")
        base = demand_metrics(rrows.actual, rrows.seasonal_naive)
        for v in ("raw", "nonnegative"):
            values = rrows[f"prediction_{v}"]
            score = demand_metrics(rrows.actual, values)
            corrections.append({**meta, "variant": v, **{f"seasonal_{k}": base[k] for k in ["MAE","RMSE","bias"]},
                                **{f"residual_{k}": score[k] for k in ["MAE","RMSE","bias"]},
                                "MAE_improvement": base["MAE"] - score["MAE"],
                                "RMSE_improvement": base["RMSE"] - score["RMSE"],
                                "absolute_bias_improvement": abs(base["bias"]) - abs(score["bias"]),
                                "mean_correction_raw": rrows.predicted_residual.mean(),
                                "correction_sd_raw": rrows.predicted_residual.std(),
                                "fraction_absolute_error_worse": (abs(values - rrows.actual) >
                                                                  abs(rrows.seasonal_naive - rrows.actual)).mean()})
    metrics = pd.DataFrame(metric_rows)
    gap_keys = ["target_site", "horizon_hours", "model", "variant"]
    cross = metrics.loc[metrics.case.str.contains("->", regex=False)]
    same = metrics.loc[metrics.case.isin(["Caltech matched", "JPL matched"])]
    delta = cross[["case", "source_site", *gap_keys, "test_MAE", "test_RMSE"]].merge(
        same[[*gap_keys, "test_MAE", "test_RMSE"]].rename(columns={"test_MAE": "matched_MAE", "test_RMSE": "matched_RMSE"}),
        on=gap_keys, validate="one_to_one").rename(columns={"test_MAE":"transfer_MAE", "test_RMSE":"transfer_RMSE"})
    for name in ("MAE","RMSE"):
        delta[f"{name}_gap"] = delta[f"transfer_{name}"] - delta[f"matched_{name}"]
        delta[f"relative_{name}_gap_percent"] = 100 * delta[f"{name}_gap"] / delta[f"matched_{name}"].replace(0,np.nan)
    return metrics, pd.DataFrame(classification), pd.DataFrame(magnitudes), pd.DataFrame(corrections), pd.DataFrame(groups), delta, pd.DataFrame(calibration)


def figures(folder, metrics, subgroup, classification, calibration, improvements, saved, residual):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "grid.alpha": .22, "figure.titlesize": 15})
    x = np.arange(4)
    axis_labels = ["1h", "3h", "6h", "24h"]
    chosen = metrics.loc[(metrics.variant.eq("raw") & ~metrics.model.isin(["Hurdle", "Seasonal Residual"])) |
                         (metrics.variant.eq("nonnegative") & metrics.model.isin(["Hurdle", "Seasonal Residual"]))]
    group = subgroup.merge(chosen[["case","horizon_hours","model","variant"]],
                          on=["case","horizon_hours","model","variant"],validate="many_to_one")
    methods = ["Hurdle", "Seasonal Residual"]

    def save(fig, index):
        fig.savefig(folder / FIGURE_NAMES[index], dpi=175, facecolor="white")
        plt.close(fig)

    def curves(ax, data, field, models=STYLE):
        for model in models:
            display, color = STYLE[model]
            values = data.loc[data.model.eq(model)].set_index("horizon_hours").loc[HORIZONS, field]
            ax.plot(x, values, "o-", lw=1.65, ms=4, label=display, color=color)
        ax.set(xticks=x, xticklabels=axis_labels, xlabel="Forecast horizon")
        ax.grid(axis="y")

    def legend(fig, ax, columns=6):
        entries, names = ax.get_legend_handles_labels()
        fig.legend(entries, names, loc="outside lower center", ncols=columns, frameon=False, fontsize=8)

    for index, measure in enumerate(["MAE","RMSE"]):
        fig, panels = plt.subplots(2,2,figsize=(11.2,7.1), layout="constrained")
        fig.suptitle(f"Hourly test {measure} by horizon · Advanced methods clipped")
        for ax, case in zip(panels.flat, FOCUS):
            curves(ax, chosen.loc[chosen.case.eq(case)], "test_" + measure)
            ax.set(title=SCOPE[case], ylabel=f"{measure} (recorded starts/hour)")
        legend(fig, panels[0,0])
        save(fig,index)
    fig, panels = plt.subplots(2,2,figsize=(11.5,7.2),layout="constrained")
    fig.suptitle("Absolute error on zero and positive target hours")
    for row, case in enumerate(["Caltech full","JPL matched"]):
        for col, subset in enumerate(["zero","positive"]):
            ax=panels[row,col]
            curves(ax,group.loc[group.case.eq(case)&group.group.eq(subset)],"MAE",["Seasonal Naive",*methods])
            ax.set(title=f"{SCOPE[case]} · {subset}", ylabel="MAE (starts/hour)")
    legend(fig,panels[0,0],3)
    save(fig,2)
    colors = dict(zip(FOCUS, ["#24657D","#98782D","#8C3D70","#39815F"]))
    fig, panels=plt.subplots(2,3,figsize=(13,7.1),layout="constrained")
    fig.suptitle("Hurdle occurrence scores · Test")
    for ax, metric in zip(panels.flat, ["precision","recall","F1","ROC_AUC","PR_AUC","Brier"]):
        for case in FOCUS:
            v = classification.loc[classification.case.eq(case)&classification.partition.eq("test")].set_index("horizon_hours")
            ax.plot(x,v.loc[HORIZONS,metric],"o-",lw=1.5,ms=4,color=colors[case],label=SCOPE[case])
        ax.set(title=metric.replace("_"," "), xticks=x, xticklabels=axis_labels, xlabel="Forecast horizon", ylim=(0,1.02))
        ax.grid(axis="y")
    legend(fig,panels[0,0],4)
    save(fig,3)
    fig, panels=plt.subplots(1,2,figsize=(10.7,4.9),layout="constrained",sharey=True)
    fig.suptitle("Hurdle reliability · Occupied equal-width probability bins")
    for ax,h in zip(panels,[1,24]):
        ax.plot([0,1],[0,1],":",color="#747B84")
        for case in FOCUS:
            block=calibration.loc[calibration.case.eq(case)&calibration.horizon_hours.eq(h)].sort_values("bin")
            ax.plot(block.mean_probability,block.observed_positive_rate,"o-",color=colors[case],label=SCOPE[case],ms=4)
        ax.set(title=f"{h}h",xlabel="Mean predicted probability",ylabel="Observed positive fraction",xlim=(0,1),ylim=(0,1))
        ax.grid()
    legend(fig,panels[0],4)
    save(fig,4)
    fig, panels=plt.subplots(1,2,figsize=(11.3,4.6),layout="constrained")
    fig.suptitle("Seasonal Naive minus clipped residual error · Positive favors correction")
    for ax,metric in zip(panels,["MAE","RMSE"]):
        for case in FOCUS:
            v=improvements.loc[improvements.case.eq(case)&improvements.variant.eq("nonnegative")].set_index("horizon_hours")
            ax.plot(x,v.loc[HORIZONS,f"{metric}_improvement"],"o-",color=colors[case],label=SCOPE[case])
        ax.axhline(0,color="#6C7280",ls=":")
        ax.set(xticks=x,xticklabels=axis_labels,xlabel="Forecast horizon",ylabel=f"Improvement in {metric} (starts/hour)")
        ax.grid(axis="y")
    legend(fig,panels[0],4)
    save(fig,5)
    test = saved.loc[saved.partition.eq("test")]
    residual_test = test.loc[test.model.eq("Seasonal Residual")].merge(residual,on=KEY,validate="one_to_one")
    fig,panels=plt.subplots(1,2,figsize=(11.7,4.5),layout="constrained")
    fig.suptitle("Distribution of hourly signed errors · Seasonal and residual-corrected")
    for ax,case in zip(panels,["Caltech full","JPL matched"]):
        for h,style in [(1,"-"),(24,"--")]:
            r=residual_test.loc[residual_test.case.eq(case)&residual_test.horizon_hours.eq(h)]
            for field,color,tag in [("seasonal_naive",STYLE["Seasonal Naive"][1],"Seasonal Naive"),
                                    ("prediction_nonnegative",STYLE["Seasonal Residual"][1],"Residual clipped")]:
                values=np.sort((r[field]-r.actual).to_numpy())
                ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",ls=style,color=color,label=f"{tag} {h}h")
        ax.set(title=SCOPE[case],xlabel="Prediction minus actual (starts/hour)",ylabel="Cumulative fraction")
        ax.grid()
    legend(fig,panels[0],4)
    save(fig,6)
    fig, panels=plt.subplots(2,2,figsize=(11.1,7),layout="constrained")
    fig.suptitle("Signed prediction bias · Advanced methods clipped")
    for ax,case in zip(panels.flat,FOCUS):
        curves(ax,chosen.loc[chosen.case.eq(case)],"test_bias",["Seasonal Naive",*methods])
        ax.axhline(0,color="#6C7280",ls=":")
        ax.set(title=SCOPE[case],ylabel="Mean prediction minus actual")
    legend(fig,panels[0,0],3)
    save(fig,7)
    fig, panels=plt.subplots(2,2,figsize=(11.6,7.2),layout="constrained")
    fig.suptitle("Matched same-site versus transfer · Clipped advanced forecasts")
    for row,site in enumerate(["Caltech","JPL"]):
        cross="JPL->Caltech" if site=="Caltech" else "Caltech->JPL"
        for col,metric in enumerate(["MAE","RMSE"]):
            ax=panels[row,col]
            for case,linestyle in [(site+" matched","--"), (cross,"-")]:
                for model in methods:
                    v=chosen.loc[chosen.case.eq(case)&chosen.model.eq(model)].set_index("horizon_hours")
                    ax.plot(x,v.loc[HORIZONS,f"test_{metric}"],"o",ls=linestyle,color=STYLE[model][1],
                            label=f"{model} · {'matched' if linestyle=='--' else 'transfer'}",lw=1.55,ms=4)
            ax.set(title=f"Target {site} · {metric}",xticks=x,xticklabels=axis_labels,
                   xlabel="Forecast horizon",ylabel=f"{metric} (starts/hour)")
            ax.grid(axis="y")
    legend(fig,panels[0,0],4)
    save(fig,8)
    fig, panels=plt.subplots(2,2,figsize=(12,7.1),layout="constrained",sharex=True)
    fig.suptitle("Actual versus forecast · Daily means of all hourly targets")
    for ax,case in zip(panels.flat,FOCUS):
        r=residual_test.loc[residual_test.case.eq(case)&residual_test.horizon_hours.eq(1)].set_index("target_hour")
        h=test.loc[test.case.eq(case)&test.horizon_hours.eq(1)&test.model.eq("Hurdle")].set_index("target_hour")
        for label,series,color in [("Actual",r.actual,"#262A30"),("Seasonal Naive",r.seasonal_naive,STYLE["Seasonal Naive"][1]),
                                   ("Hurdle clipped",h.prediction_nonnegative,STYLE["Hurdle"][1]),
                                   ("Residual clipped",r.prediction_nonnegative,STYLE["Seasonal Residual"][1])]:
            daily=series.resample("D").mean()
            ax.plot(daily.index,daily,lw=1.2,color=color,label=label)
        ax.set(title=f"{SCOPE[case]} · 1h",ylabel="Recorded starts/hour")
        ax.grid(axis="y")
    for ax in panels[-1]:
        ax.xaxis.set_major_locator(dates.MonthLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter("%b"))
        ax.set_xlabel("Target day (2020 source clock)")
    legend(fig,panels[0,0],4)
    save(fig,9)


def report(metrics, classification, positive, corrections, subgroup, gaps, fit, env):
    primary = metrics.loc[metrics.variant.eq("nonnegative") & metrics.model.isin(["Hurdle","Seasonal Residual"])]
    component = classification.loc[classification.partition.eq("test")].merge(
        positive.loc[positive.partition.eq("test")], on=["case","horizon_hours"], suffixes=("","_mag"),validate="one_to_one")
    scores, outcome, classification_rows, transfers, selections, interpretation = [], [], [], [], [], []
    for case, block in metrics.groupby("case",sort=False):
        for h, step in block.groupby("horizon_hours"):
            p = primary.loc[primary.case.eq(case)&primary.horizon_hours.eq(h)]
            naive = step.loc[step.model.eq("Seasonal Naive")].iloc[0]
            for row in step.itertuples():
                scores.append(f"| {SCOPE[case]} | {h} | {label(row.model,row.variant)} | "
                              f"{row.test_MAE:.6f} | {row.test_RMSE:.6f} | {row.test_R2:.6f} | {row.test_bias:+.6f} | "
                              f"{row.zero_MAE:.6f} | {row.positive_MAE:.6f} | {row.zero_RMSE:.6f} | {row.positive_RMSE:.6f} |")
            winner1, winner2 = step.loc[step.test_MAE.idxmin()], step.loc[step.test_RMSE.idxmin()]
            a,b = p.iloc[0], p.iloc[1]
            outcome.append(f"| {SCOPE[case]} | {h} | {naive.test_MAE:.4f} / {naive.test_RMSE:.4f} | "
                           f"{a.test_MAE:.4f} / {a.test_RMSE:.4f} | {b.test_MAE:.4f} / {b.test_RMSE:.4f} | "
                           f"{label(winner1.model,winner1.variant)} ({winner1.test_MAE:.4f}) | "
                           f"{label(winner2.model,winner2.variant)} ({winner2.test_RMSE:.4f}) |")
        common_keys = ["case","horizon_hours"]
        sn = block.loc[block.model.eq("Seasonal Naive"), common_keys+["test_MAE","test_RMSE","zero_MAE","positive_MAE"]]
        for model, selected in primary.loc[primary.case.eq(case)].groupby("model",sort=False):
            delta=selected.merge(sn,on=common_keys,suffixes=("","_sn"),validate="one_to_one")
            counts=[int(delta[k].lt(delta[k+"_sn"]).sum()) for k in ["test_MAE","test_RMSE","zero_MAE","positive_MAE"]]
            interpretation.append(f"| {SCOPE[case]} | {label(model)} | " + " | ".join(f"{x}/4" for x in counts) + " |")
    for r in component.itertuples():
        classification_rows.append(f"| {SCOPE[r.case]} | {r.horizon_hours} | {r.precision:.4f} | {r.recall:.4f} | "
                                   f"{r.F1:.4f} | {r.ROC_AUC:.4f} | {r.PR_AUC:.4f} | {r.Brier:.4f} | "
                                   f"{r.MAE:.4f} / {r.RMSE:.4f} |")
    for r in gaps.loc[gaps.model.isin(["Hurdle","Seasonal Residual"])&gaps.variant.eq("nonnegative")].itertuples():
        transfers.append(f"| {SCOPE[r.case]} | {r.horizon_hours} | {label(r.model)} | {r.MAE_gap:+.6f} "
                         f"({r.relative_MAE_gap_percent:+.2f}%) | {r.RMSE_gap:+.6f} ({r.relative_RMSE_gap_percent:+.2f}%) |")
    for r in fit.itertuples():
        params = r.chosen_parameters.replace("|","\u007c")
        selections.append(f"| {SCOPE[r.case]} | {r.horizon_hours} | {label(r.model)} | {r.n_final:,} | "
                          f"`{params}` | {r.validation_MAE:.6f} |")
    clipped = corrections.loc[corrections.variant.eq("nonnegative")]
    over = clipped.sort_values("fraction_absolute_error_worse",ascending=False).iloc[0]
    indexed = primary.set_index(["case", "horizon_hours", "model"])
    cal1 = indexed.loc[("Caltech full", 1, "Hurdle")]
    jpl3 = indexed.loc[("JPL matched", 3, "Hurdle")]
    forward1 = indexed.loc[("Caltech->JPL", 1, "Hurdle")]
    back1 = indexed.loc[("JPL->Caltech", 1, "Hurdle")]
    forward24 = indexed.loc[("Caltech->JPL", 24, "Seasonal Residual")]
    prior24 = metrics.loc[metrics.case.eq("Caltech->JPL") & metrics.horizon_hours.eq(24)
                          & ~metrics.model.isin(["Hurdle", "Seasonal Residual"]), "test_RMSE"].min()
    example = gaps.loc[gaps.case.eq("Caltech->JPL") & gaps.horizon_hours.eq(1)
                       & gaps.model.eq("Hurdle") & gaps.variant.eq("nonnegative")].iloc[0]
    return f"""# Experiment 07 充电需求的 Hurdle 与季节残差学习

## 问题与方法

此前 Seasonal Naive 的 MAE 强，零记录小时占比高，按零值和正值分组时模型排名不同。Hurdle 分别估计发生概率 p(x)=P(y>0|x) 与正目标上的条件数量 μ(x)，原始预测为 p(x)×μ(x)。发生分类使用全量可用标签，数量回归只使用正值标签。3 个分类器与 3 个数量模型组成 9 对候选，复用候选组件的验证输出，避免逐组合重复拟合。条件数量模型的原始输出供诊断。主 Hurdle 预测对最终乘积取 max(0,预测)。

季节残差学习以已有 Seasonal Naive 的 y(t+h−24) 为基准。训练标签是已可用的真实值减去该样本的季节预测。Ridge、Random Forest 和 Gradient Boosting 各候选估计带符号的残差。推理时将预测残差加回基准，并分别保存原始及最终截断版本。残差本身保持有符号。两个方法均继续使用既有的 34 个特征和直接滚动起点。

## 实验和公平性

07A 使用官方完整历史 Caltech 同站切分并先通过阶段验证。07B 使用 JPL 的官方共同窗口。候选结构和参数网格在 Caltech 前固定，JPL 与跨站按各自训练站验证集重新选参。07C 使用双向迁移，另拟合共同窗口 Caltech 同站对照，以匹配训练历史长度。全部窗口与官方训练、验证、最终拟合和测试哈希核对。各候选只按原始验证 MAE 选择，最后从头拟合已释放的训练与验证行。无类别阈值或分组测试调参。固定阈值 0.5 只用于分类指标。

目标都是原源时钟的 2020-07-15 00:00 至 2020-12-02 22:00 共 3,383 小时。跨站仅从源站标签训练，冻结后使用目标站届时可用的历史。它属于使用目标历史的迁移。官方基准从历史预测文件逐键读取并重算指标，未复制原预测文件或更新旧结果。部分候选的随机森林并行拟合，串行预测确保求和顺序一致。

## 主结果

前三个分数列为 MAE / RMSE。新方法列用最终非负版本，最优列纳入各官方模型原版及已有截断版本。

| 设定 | h | Seasonal Naive | Hurdle | 季节残差 | 最低 MAE | 最低 RMSE |
|---|---:|---|---|---|---|---|
{chr(10).join(outcome)}

下表统计每个方法改善 Seasonal Naive 的跨度数。正值和零值分组是事后诊断。

| 设定 | 新方法 | 整体 MAE | 整体 RMSE | 零值 MAE | 正值 MAE |
|---|---|---:|---:|---:|---:|
{chr(10).join(interpretation)}

Zero 与 positive 排名应分别解释。发生概率与正条件均值相乘是期望预测，零值频繁时，即使发生概率准确，非零的期望也可能增加零值小时绝对误差。季节残差的绝对误差有时变大。截断版的最大恶化小时占比为 {over.fraction_absolute_error_worse:.2%}（{SCOPE[over.case]}，{over.horizon_hours}h），该比例按逐小时与同一季节基准比较。残差表还记录 MAE、RMSE 和绝对偏差改善，数值为正代表改善。

## Hurdle 发生与数量诊断

分类报告与数量回归分开。PR-AUC 为梯形积分，CSV 还记录平均精度 AP。最后一列在真实正需求小时评估原始条件数量 μ，整体 Hurdle 对这些小时的误差另见分组表。校准图显示非空等宽概率箱，箱内观测频率用于描述校准。

| 设定 | h | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | 条件数量 MAE / RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(classification_rows)}

## 两个迁移方向

迁移差为跨站测试误差减同一目标站匹配同站误差。括号为相对差百分比。表内是新方法截断版，CSV 保留同口径的原版和官方对照。比较只在目标相同的条件下解释。

| 方向 | h | 方法 | MAE 差 | RMSE 差 |
|---|---:|---|---:|---:|
{chr(10).join(transfers)}

## 超参数与拟合验证

| 设定 | h | 方法 | 最终全量拟合 | 选定参数 | 验证原始 MAE |
|---|---:|---|---:|---|---:|
{chr(10).join(selections)}

数量模型的训练和最终拟合样本数及时间戳哈希单列保存，残差目标保留符号并保存顺序哈希。每个模型实际预测均通过未来观测扰动测试。独立脚本重建预测公式、指标、候选赢家和官方拟合键。先验保护清单覆盖 Stage 2 和 Stage 3 的已有文件。两份组件表以 case、horizon_hours、partition、target_hour 与主预测表连接，避免复制逐小时实际值。

## 完整指标

| 设定 | h | 模型 | MAE | RMSE | R² | 偏差 | 零 MAE | 正 MAE | 零 RMSE | 正 RMSE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(scores)}

## 局限、结论与可行扩展

Hurdle 的优势集中在特定站点和指标。Caltech 1h 的 RMSE 为 {cal1.test_RMSE:.6f}，但其 MAE 为 {cal1.test_MAE:.6f}，高于 Seasonal Naive。JPL 3h 的 Hurdle 同时达到 {jpl3.test_MAE:.6f} MAE 和 {jpl3.test_RMSE:.6f} RMSE。Caltech→JPL 1h 的 Hurdle 达到 {forward1.test_MAE:.6f}/{forward1.test_RMSE:.6f}，相对 JPL 匹配同站 Hurdle，MAE 差 {example.MAE_gap:+.6f} 而 RMSE 差 {example.RMSE_gap:+.6f}。JPL→Caltech 1h 的 Hurdle RMSE 为 {back1.test_RMSE:.6f}，较长跨度仍由 Seasonal Naive 领先。

Seasonal Residual 在 JPL 同站四个跨度均改善 Seasonal Naive 的 RMSE，但只在 1h 改善整体 MAE。Caltech→JPL 24h 的截断残差 RMSE 为 {forward24.test_RMSE:.6f}，与此前最优的 {prior24:.6f} 相差 {prior24-forward24.test_RMSE:.6f}。该差值很小，缺乏跨窗口稳定性证据。

结论适用于两个归档站点及当前时间窗口。时区和运营覆盖未知，零值表示没有归档的新会话记录。滚动起点假设计数及时可用。两种模型的候选网格有限。分类概率未做二次校准，模型辨别力和概率可靠性仍需分开判断。当前分数没有跨窗口不确定性区间。图表主比较用已有原始基准和新模型截断值，所有原版与官方截断版均在表中。概率校准和残差分布图示例为 1h 和 24h，实际与预测图用 1h 全测试窗的逐日均值。

- 扩展 C 可用时间顺序的样本外预测训练门控或非负集成权重，并约束权重选择使用验证窗口。逐跨度比较稳定性和失效情形。
- 扩展 D 可用分位数回归和 pinball loss，并留出时间上独立的校准段进行 conformal 区间。在存在时序依赖和分布变化时，应报告分跨度覆盖率和区间宽度，谨慎解释覆盖保证。

以上扩展均未实现。运行环境为 Python {env['python']}、scikit-learn {env['scikit_learn']}、pandas {env['pandas']}、NumPy {env['numpy']}、Matplotlib {env['matplotlib']}。

```powershell
python -B experiments/07_advanced_forecasting/verify_experiment.py
python -B experiments/07_advanced_forecasting/evaluate_models.py --verify
```

第二条命令在干净进程重拟合，并扰动只供评估的测试标签缓冲区，保留可用历史。预期输出与正式实验相同，耗时字段除外。
"""
