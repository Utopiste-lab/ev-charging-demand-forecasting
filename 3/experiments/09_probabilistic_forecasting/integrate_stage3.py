"""Build closure scoreboards, figures and research reports from saved results."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from stage3_context import (FINAL_METRICS, FINAL_TABLES, PROB_METRICS, PROB_TABLES, ROOT, Project, publish, read)
from stage3_figures import baseline, probabilistic

SITES = ["Caltech full", "JPL matched"]
PRIMARY = ["Caltech full", "JPL matched", "Caltech->JPL", "JPL->Caltech"]
KEY = ["case", "horizon_hours"]


def md(headers, lines):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|",
                      *["| " + " | ".join(map(str, row)) + " |" for row in lines]])


def site(name):
    return name.replace("->"," → ")


def fmt(x):
    return f"{x:.6f}"


def coverage(x):
    return f"{x:.2%}"


def official(table):
    return table.loc[table.window.eq("official") & table.partition.eq("test")]


def consolidate(points, timing):
    archive = read(ROOT / FINAL_METRICS / "archived_point_comparison.csv")
    mix = archive.loc[archive.experiment.eq("08") & archive.model.isin(
        ["Static equal","Static optimized","Rule gate","Adaptive full","Adaptive without Hurdle",
         "Adaptive without Residual","Adaptive without history","Adaptive without calendar",
         "Adaptive target-calibrated","Hurdle median"])]
    base = archive.loc[archive.experiment.ne("08")]
    final = official(points)
    novel = final.loc[final.model.isin(["P50","Point GB pre-calibration","Point GB pre-calibration raw"])].merge(
        timing.loc[timing.window.eq("official") & timing.partition.eq("test"),
                   KEY + ["source_site","target_site","timestamp_sha256"]], on=KEY, validate="many_to_one"
    ).rename(columns={"timestamp_sha256":"target_sha256"}).drop(columns=["window","partition"])
    novel = novel.assign(experiment="09",variant="primary",fit_protocol="train_plus_early_validation_excludes_calibration")
    merged = pd.concat([base,mix,novel],ignore_index=True)
    assert merged.groupby(KEY).target_sha256.nunique().eq(1).all()
    assert set(merged.n) == {3383}
    official_only = merged.loc[merged.case.isin(SITES)]
    def rank(g, field):
        q = g.sort_values([field,"model","variant","experiment"],kind="stable")
        return q.iloc[0]
    winners = []
    for (case, h), g in official_only.groupby(KEY,sort=False):
        row = {"case":case,"horizon_hours":h,"n":g.n.iat[0],"target_sha256":g.target_sha256.iat[0]}
        for label, sub in [("overall",g),("learned",g.loc[~g.model.isin(["Constant Zero","Seasonal Naive"])])]:
            for field in ["MAE","RMSE"]:
                chosen = rank(sub,field)
                row.update({f"{label}_{field}_{name}":chosen[name] for name in ("model","variant","experiment")})
                row[f"{label}_{field}"] = chosen[field]
                row[f"{label}_{field}_ties"] = " / ".join(
                    sorted(dict.fromkeys(sub.loc[np.isclose(sub[field],chosen[field],rtol=0,atol=1e-12)]
                                         .apply(lambda u:f"{u.model} ({u.variant})",axis=1))))
        winners.append(row)
    matched = merged.loc[merged.case.isin(["JPL matched","Caltech matched"])]
    cross = merged.loc[merged.case.str.contains("->",regex=False)].copy()
    cross["evaluation_mode"] = np.where(cross.model.eq("Adaptive target-calibrated"),"target_pretest_offset","source_only_or_no_fit")
    cross["matched_case"] = cross.target_site + " matched"
    cross["matched_model"] = cross.model.replace({"Adaptive target-calibrated":"Adaptive full"})
    pairs = cross.merge(matched[["case","horizon_hours","experiment","model","variant","target_sha256","MAE","RMSE"]],
                        left_on=["matched_case","horizon_hours","experiment","matched_model","variant"],
                        right_on=["case","horizon_hours","experiment","model","variant"],suffixes=("","_matched"),
                        validate="many_to_one")
    assert pairs.target_sha256.eq(pairs.target_sha256_matched).all()
    for field in ("MAE","RMSE"):
        pairs[f"{field}_gap"] = pairs[field]-pairs[f"{field}_matched"]
        pairs[f"{field}_gap_percent"] = 100*pairs[f"{field}_gap"]/pairs[f"{field}_matched"].replace(0,np.nan)
    adaptive = mix.loc[mix.model.isin(["Static equal","Static optimized","Rule gate","Adaptive full","Hurdle median"])]
    return merged,pd.DataFrame(winners),pairs,adaptive


def range_text(values, style=".2%"):
    return f"{values.min():{style}}–{values.max():{style}}"


def probabilistic_report(interval, quantiles, point, adjustments, crossing, prediction, bootstrap, cfg):
    ix = official(interval).loc[lambda f:f.regime.eq("all")]
    official_primary = ix.loc[ix.case.isin(SITES)]
    raw = official_primary.loc[official_primary.method.eq("raw")].set_index(KEY)
    cqr = official_primary.loc[official_primary.method.eq("CQR")].set_index(KEY)
    pin = official(quantiles).pivot(index=KEY,columns="quantile",values="pinball")
    point = official(point).pivot(index=KEY,columns="model",values=["MAE","RMSE"])
    subs = official(interval).loc[lambda f:f.method.eq("CQR") & f.regime.ne("all")].set_index(KEY + ["regime"])
    by_primary, point_rows, cross_rows = [], [], []
    for (case,h), row in cqr.iterrows():
        r=raw.loc[(case,h)]
        zero, pos = [subs.loc[(case,h,s)] for s in ["zero","positive"]]
        by_primary.append([site(case),h,coverage(r.coverage),coverage(row.coverage),fmt(r.width),fmt(row.width),fmt(row.interval_score),
                           coverage(zero.coverage),coverage(pos.coverage),fmt(zero.width),fmt(pos.width),
                           *(fmt(pin.loc[(case,h),q]) for q in ["P10","P50","P90"])])
        point_rows.append([site(case),h,*[f"{point.loc[(case,h),('MAE',n)]:.6f} / {point.loc[(case,h),('RMSE',n)]:.6f}"
                                         for n in ["Constant Zero","Seasonal Naive","Point GB pre-calibration","P50"]]])
    for r in ix.loc[ix.case.str.contains("->",regex=False) & ix.method.eq("CQR")].itertuples(index=False):
        match = ix.loc[ix.case.eq(("JPL" if r.case.endswith("JPL") else "Caltech")+" matched")
                       & ix.horizon_hours.eq(r.horizon_hours) & ix.method.eq("CQR")].iloc[0]
        cross_rows.append([site(r.case),r.horizon_hours,coverage(r.coverage),coverage(match.coverage),
                           f"{100*(r.coverage-match.coverage):+.2f} pp",fmt(r.width),fmt(match.width)])
    roll=interval.loc[interval.window.ne("official") & interval.regime.eq("all")]
    rolling_rows = []
    for (window,case), g in roll.groupby(["window","case"],sort=False):
        gg=g.pivot(index="horizon_hours",columns=["partition","method"],values=["coverage","width"])
        rolling_rows.append([window,site(case),range_text(gg[("coverage","calibration","CQR")]),
                             range_text(gg[("coverage","test","raw")]),range_text(gg[("coverage","test","CQR")]),
                             range_text(gg[("width","test","CQR")],".3f")])
    adjustments_primary=adjustments.loc[adjustments.window.eq("official") & adjustments.case.isin(SITES)]
    cal_rows = []
    for r in adjustments_primary.itertuples(index=False):
        cal=interval.loc[interval.window.eq("official") & interval.case.eq(r.case)
                         & interval.horizon_hours.eq(r.horizon_hours) & interval.regime.eq("all") & interval.partition.eq("calibration")]
        q=cal.set_index("method")
        cal_rows.append([r.case,r.horizon_hours,r.n_calibration,r.rank_1_based,fmt(r.adjustment),
                         coverage(q.at["raw","coverage"]),coverage(q.at["CQR","coverage"])])
    n_changed=(adjustments_primary.adjustment!=0).sum()
    all_changed=(adjustments.adjustment!=0).sum()
    fixed_test=official(crossing)
    primary_rows=fixed_test.loc[fixed_test.case.isin(SITES)]
    primary_forecast = official(prediction).loc[lambda f:f.case.isin(SITES)]
    zero_point_rows=int(primary_forecast.P50.eq(0).sum())
    boots=official(bootstrap).loc[lambda f:f.case.isin(SITES) & f.horizon_hours.eq(1) & f.statistic.eq("CQR")]
    brep=[]
    for (case,block), g in boots.groupby(["case","block_hours"],sort=False):
        gi=g.set_index("metric")
        brep.append([case,block,f"[{coverage(gi.at['coverage','ci_low'])}, {coverage(gi.at['coverage','ci_high'])}]",
                     f"[{fmt(gi.at['width','ci_low'])}, {fmt(gi.at['width','ci_high'])}]"])
    return f"""# Experiment 09 概率预测与按时间校准

## 目标与拟合

预测源时钟每小时归档会话开始数。为 1、3、6、24h 分别拟合 P10、P50、P90 的 Gradient Boosting 回归器，使用 pinball loss。每个分位数只比较三个预设梯度提升配置，按提前验证标签选择最低对应 pinball。原始分位数逐行截为非负，再重新排序；保存原始值与更正值。平方误差 Gradient Boosting 点模型使用完全相同的训练和较早验证日历，按验证 MAE 选参，点结果保存原始及非负版本。

官方验证末 {cfg['calibration_days']['official']} 天划为校准名义窗口。R1–R3 的验证末 {cfg['calibration_days']['rolling']} 天划为校准。训练、较早验证、校准和测试相邻边界均执行标签释放清除规则。最终模型只拟合已释放的训练与早期验证，在看到校准标签前冻结。跨站使用源站拟合及校准和目标站因果历史，目标站标签仅用于评分。

原始区间为重新排序后的 [P10,P90]。CQR 用 max(P10−y, y−P90) 的第 ceil(0.80(n+1)) 个顺序统计量。使用有符号调整，低端 0 截断。若区间倒置，保存原上端并把空区间修复为下端单点。主同站官方测试修复次数为 {primary_rows.cqr_empty_repairs.sum():,}。主同站官方测试原始交叉 {primary_rows.crossing_raw_n.sum():,}/{primary_rows.n.sum():,}，修正后 {primary_rows.crossing_after_n.sum():,}。

## 官方测试点结果

下表为 MAE / RMSE，完整 CSV 包含 R² 与偏差、需求分组误差和原始点 GB。

{md(["站点","h","恒零","Seasonal Naive","预校准点 GB","P50"],point_rows)}

主同站测试 P50 在 {zero_point_rows:,}/{len(primary_forecast):,} 行为零。需要同时比较恒零和正需求误差，才能解释 P50 的低 MAE。区间上界仍随小时而变化。

## 官方测试区间与分位数

{md(["站点","h","原始覆盖","CQR 覆盖","原始宽度","CQR 宽度","CQR 区间分数","零值覆盖","正值覆盖","零值宽度","正值宽度","P10 pinball","P50 pinball","P90 pinball"],by_primary)}

覆盖以包含端点计算，区间分数为宽度加端点漏覆盖距离的 10 倍。离散计数和大质量零点可能使覆盖超过 80%，尤其当零点总落在区间内时。分组覆盖是事后诊断，训练和校准权重均不使用测试组别。

## 调整量及校准

主固定窗 {n_changed}/{len(adjustments_primary)} 个同站–跨度的校准调整非零。全部 44 个设定中，{all_changed} 个调整非零。调整精确为零时 CQR 与原区间完全重合。即使严格按秩校准，包含离散零点的区间也可能持续过覆盖，无法据此认定正值组校准良好。区分同窗校准和测试表现。

{md(["站点","h","校准小时","秩","调整","原覆盖","CQR 覆盖"],cal_rows)}

## 滚动窗口

{md(["窗口","站点","校准 CQR 覆盖","测试原覆盖","测试 CQR 覆盖","测试 CQR 宽度"],rolling_rows)}

24h 与 168h 块使用 {cfg['bootstrap']['replicates']} 次移动区块平均。区间对同小时、同区块成对抽样，分别估计覆盖、宽度及区间分数和它们的 CQR 减原始差。bootstrap 区间反映窗口内的相关性与采样不确定性，小时依赖与站点/时间漂移限制了交换性。此研究报告经验覆盖。

{md(["官方 1h","块长","CQR 覆盖 95% 区间","CQR 宽度 95% 区间"],brep)}

## 探索性跨站

{md(["方向","h","源校准覆盖","匹配同站覆盖","覆盖差","源校准宽度","匹配宽度"],cross_rows)}

可比较场景共享官方测试时点。源校准跨站结果反映目标历史可用时的区间迁移。同源校准的覆盖无法仅凭名义水平外推到另一站。

## 复现与范围

`experiments/09_probabilistic_forecasting/run_probabilistic_forecasting.py --verify` 在干净进程重训并替换评分专用测试标签。独立 `verify_experiment.py` 核查预测键、分位数约束、选参、校准秩、区间及重采样指标。输入为通过原有数据合约的小时表。时区和实际运营覆盖未知。预测整数计数的连续分位数可理解为端点估计，区间相对颗粒度和样本分布应一并考察。
"""


LIMITATIONS = [
    {"topic":"档案覆盖和零值","evidence":"仅确认归档网格与文件名会话能够逐小时对账","implication":"无法区分未运营、未归档和真实低需求","next_validation":"获取运营和归档覆盖日志"},
    {"topic":"源时钟","evidence":"文件名无时区，历史最小抽查发现遥测延迟与时间差","implication":"跨站日历标签只在源时钟下可比较","next_validation":"用可信 connectionTime 和设备日志对账"},
    {"topic":"站点与时期","evidence":"两个归档站点，官方测试窗口相同，滚动训练窗重叠","implication":"无法外推到所有充电站或把滚动分数视为独立样本","next_validation":"第三站和更晚窗口冻结外部评估"},
    {"topic":"发生诊断","evidence":"整体 MAE 受大量真实零值影响","implication":"恒零的低 MAE 与正值召回为零可以同时成立","next_validation":"预先冻结正值损失与实际告警阈值"},
    {"topic":"窗口和门控","evidence":"滚动窗口排名变化，OOF 至正式测试输入有均值位移","implication":"门控收益和专家权重可能对时期与特征敏感","next_validation":"按更多预设窗口测量漂移并作外部复验"},
    {"topic":"预测区间","evidence":"点质量零值、离散计数、相关小时及分布变化","implication":"校准集内覆盖不能保证测试名义覆盖，整体值可掩盖分组失败","next_validation":"单独监控零/正值和跨站覆盖"},
    {"topic":"建模目标和延迟","evidence":"标签是归档开始次数，使用完成小时即时可得假设","implication":"结果无法直接解释为能量、占用量或上线延迟下的表现","next_validation":"取得业务目标和实际数据到达时间"},
    {"topic":"同键排名与不同拟合窗","evidence":"Experiment 09 保留了单独校准标签，统一点模型使用原协议","implication":"最终排名用于描述冻结预测，在协议间不能单独归因于模型架构","next_validation":"在同一拟合窗与完整共同测试集下另行设计控制比较"},
]


def closure_report(project, merged, winners, pairs, adaptive, interval, point, forecasts, replay_audit):
    zero = merged.loc[merged.model.eq("Constant Zero")].set_index(KEY)
    season = merged.loc[merged.model.eq("Seasonal Naive")].set_index(KEY)
    by_site = []
    for case in SITES:
        z,s=zero.loc[(case,1)],season.loc[(case,1)]
        by_site.append([case,int(z.n),coverage(z.zero_proportion),fmt(z.MAE),fmt(s.MAE),
                        fmt(z.RMSE),fmt(s.RMSE),fmt(z.positive_MAE),fmt(s.positive_MAE),coverage(z.occurrence_recall)])
    one = read(ROOT / "results/metrics/tree_models_1h_comparison.csv")
    h1 = [[row.model, row.prediction_variant, fmt(row.test_MAE),fmt(row.test_RMSE),fmt(row.test_R2)]
          for row in one.itertuples(index=False)]
    prev04 = merged.loc[merged.experiment.eq("04")]
    horizons = []
    for h, g in prev04.groupby("horizon_hours"):
        a,b=g.loc[g.MAE.idxmin()],g.loc[g.RMSE.idxmin()]
        horizons.append([h,f"{a.model} ({a.variant})",fmt(a.MAE),f"{b.model} ({b.variant})",fmt(b.RMSE)])
    classic = pairs.loc[pairs.experiment.eq("06") & pairs.variant.eq("raw")
                       & ~pairs.model.eq("Seasonal Naive")]
    transfer=[]
    for case,g in classic.groupby("case"):
        transfer.append([site(case), len(g), range_text(g.MAE_gap,"+.4f"),range_text(g.RMSE_gap,"+.4f"),
                         range_text(g.bias,"+.4f"), int((g.MAE_gap>0).sum()),int((g.RMSE_gap>0).sum())])
    prev7=merged.loc[merged.experiment.eq("07") & merged.variant.eq("nonnegative")]
    e7=[]
    for case in SITES:
        for model in ["Hurdle","Seasonal Residual"]:
            a=prev7.loc[prev7.case.eq(case)&prev7.model.eq(model)].set_index("horizon_hours")
            b=season.loc[case]
            e7.append([case,model,f"{(a.MAE<b.MAE).sum()}/4",f"{(a.RMSE<b.RMSE).sum()}/4",
                       f"{(a.zero_MAE<b.zero_MAE).sum()}/4",f"{(a.positive_MAE<b.positive_MAE).sum()}/4"])
    e8=[]
    for case in SITES:
        q=adaptive.loc[adaptive.case.eq(case)].pivot(index="horizon_hours",columns="model",values=["MAE","RMSE"])
        for metric in ["MAE","RMSE"]:
            g=q[metric]
            e8.append([case,metric,*[f"{(g['Adaptive full']<g[name]).sum()}/4" for name in
                                   ["Static optimized","Rule gate","Hurdle median"]]])
    raw_vs_cqr=official(interval).loc[lambda f:f.case.isin(SITES)&f.regime.eq("all")]
    iv=[]
    for case,g in raw_vs_cqr.groupby("case"):
        raw=g.loc[g.method.eq("raw")]
        cqr=g.loc[g.method.eq("CQR")]
        masked=official(interval).loc[lambda f:f.case.eq(case)&f.method.eq("CQR")]
        z,p=[masked.loc[masked.regime.eq(s)] for s in ("zero","positive")]
        iv.append([case,range_text(raw.coverage),range_text(cqr.coverage),range_text(cqr.width,".4f"),
                   range_text(z.coverage),range_text(p.coverage)])
    prev_roll=read(ROOT / "results/metrics/adaptive_hybrid/rolling_window_comparison.csv")
    roll_new=interval.loc[interval.window.ne("official") & interval.method.eq("CQR")
                          & interval.partition.eq("test") & interval.regime.eq("all")]
    rollrows=[]
    for (w,case),g in prev_roll.groupby(["window","case"]):
        pm=g.pivot(index="horizon_hours",columns="model",values="MAE")
        pi=roll_new.loc[roll_new.window.eq(w)&roll_new.case.eq(case)]
        rollrows.append([w,case,*[f"{(pm[model]<pm['Seasonal Naive']).sum()}/4" for model in
                                ["Hurdle clipped","Residual clipped","Adaptive full"]],range_text(pi.coverage)])
    wrank=[[r.case,r.horizon_hours,f"{r.overall_MAE_model} ({r.overall_MAE_variant})",fmt(r.overall_MAE),
            f"{r.overall_RMSE_model} ({r.overall_RMSE_variant})",fmt(r.overall_RMSE),r.learned_MAE_model,fmt(r.learned_MAE)]
           for r in winners.itertuples(index=False)]
    cov_table=md(["站点","原始总体覆盖","CQR 总体覆盖","CQR 宽度","零值 CQR 覆盖","正值 CQR 覆盖"],iv)
    hour_counts = project.full.groupby("site").size()
    tests = project.keys
    aud = [[row.step,row.method,row.status,row.scope] for row in replay_audit.itertuples(index=False)]
    p50 = official(point).loc[lambda f:f.case.isin(SITES)].pivot(index=KEY,columns="model",values=["MAE","RMSE"])
    vz = p50["MAE"]["P50"].lt(p50["MAE"]["Constant Zero"]).sum()
    vp = p50["MAE"]["P50"].lt(p50["MAE"]["Point GB pre-calibration"]).sum()
    central=official(forecasts).loc[lambda f:f.case.isin(SITES)]
    central_zeros=central.P50.eq(0).mean()
    return f"""# Stage 3 Final Research Report

## 1. 执行摘要

本阶段冻结两站四跨度的小时归档会话开始数预测。模型的总体绝对误差、正值识别和区间覆盖可以给出不同排序。Caltech 恒零基准的低 MAE 与正值召回为零同时成立。JPL 的自适应门控有固定窗 MAE 收益，简单混合在 RMSE 或其他时期也有优势。概率预测显示区间的总体覆盖与正值小时覆盖差异，测试覆盖需按站点、时期和需求组说明。

## 2. 背景与动机

EV 充电站有日历规律、短时观测依赖和大量零计数。对多个预测跨度和站点分别评分，能检验这些规律是地点特定还是可迁移。单一误差平均值无法表达对少见正计数的处理，也无法描述预测不确定性。

## 3. 研究目标

RQ1 比较直接预测的四个跨度。RQ2 比较简单基准和模型复杂度。RQ3 量化零值与正值的误差。RQ4 在同一目标时点比较双向迁移与匹配同站控制。RQ5 检验自适应组合与简化方案。RQ6 检验分位数和校准区间覆盖、宽度及其稳健性。

## 4. 数据与处理

Stage 3 读取已对账的 `hourly_counts_v1.csv`。Caltech {hour_counts['Caltech']:,} 小时，JPL {hour_counts['JPL']:,} 小时。一条档案源会话由文件名开始时间计入一个小时。保留零值、峰值及首尾边界标记，按原数据合约核对 SHA-256 与站点范围。正式模型排除可能不完整的边界目标和历史。此次收官只枚举原始归档元数据，未开启归档内容。

## 5. 问题定义

第 t 个历法小时的目标 y(t) 是归档开始数。以最近已完成的小时 o 为原点，直接预测 y(o+h)，h∈{{1,3,6,24}}。发布时钟为 o+1h，允许使用届时已完成的历史。统计量的单位是记录的 starts/hour。

## 6. 特征与可用性

34 列输入为目标时点的 30 个日历指标，以及原点处的 4 个历史需求特征 y(o)、y(o−23)、y(o−167) 与截至 o 的过去 24 小时平均。Seasonal Naive 使用 y(o+h−24)，对全部跨度都在发布时可用。从独立目标和已完成观测索引构造，保留标签发布时间与预测发布时间。

## 7. 实验日历与协议

所有固定窗比较对齐源时钟 {tests[0]} 至 {tests[-1]}，各有 {len(tests):,} 个测试目标。校验时钟依据记录时间，标签在目标小时结束时释放。已统一的点模型用训练选参候选、验证 MAE 选择和可释放训练加验证最终拟合。Experiment 09 单独保留校准期，P50 与点 GB 共享更早的拟合窗口。跨站 matched 组使用重叠历史；Caltech full 保留完整历史。所有团队的原始比较保持原样。

## 8. Experiments 01–03 传统基准

研究问题是可复现的一小时性能和训练集公平性。现行分数使用统一最终重拟合，原先 train-only Ridge 与不同长度历史协议保留在历史审查。MAE 优胜和 RMSE 优胜分别报告。早期历史包的 3,384 样本口径仅用于回归检查，正式窗口排除末端小时。

{md(["方法","版本","MAE","RMSE","R²"],h1)}

## 9. Experiment 04 多跨度

复用 34 列、MAE 选参和最终重拟合协议，模型按跨度分别训练。记录误差随跨度的变化，并用同键避免目标样本构成成为解释。下表是原实验候选的最低分，恒零和晚期模型在第 14 节。

{md(["跨度","最低 MAE 模型","MAE","最低 RMSE 模型","RMSE"],horizons)}

## 10. Experiments 05–06 双向迁移

Experiment 05 独立核查共同源时钟、释放约束和站点隔离。Experiment 06 将源站训练的 Ridge、Random Forest 和 Gradient Boosting 沿用到另一站，目标站滚动已完成历史可作输入，匹配同站使用同历法，以隔离目标分布与比较时窗。下表对原始未截断学习模型及四个跨度分别算差值范围，正差表示跨站较差。两站均值、零比例与设备归档差异提供转移困难的线索，档案无法确定成因。

{md(["方向","比较数","MAE 差范围","RMSE 差范围","跨站偏差范围","MAE 恶化数","RMSE 恶化数"],transfer)}

## 11. Experiment 07 两类混合专家

Hurdle 通过发生概率乘条件正值数量预测期望，季节残差法用机器学习估计带符号的同季节差。下表统计最终截断模型比 Seasonal Naive 误差更低的跨度数。正值组的收益可能被零值组损失抵消。类别概率和正值数量分别评分，发生分类的 0.5 阈值与第 14 节的点预测正值规则独立。

{md(["站点","模型","整体 MAE","整体 RMSE","零值 MAE","正值 MAE"],e7)}

## 12. Experiment 08 自适应组合

归档预测先用扩展式时间 OOF 得到门控训练输入，再由正式验证选择静态权重、概率阈值或线性/小型 tanh softmax 门。最终拟合覆盖已释放训练与验证标签，冻结门权重函数。固定窗源站迁移与目标站测试前校准单列。下表较低误差计数的比较方法不使用目标测试调参。

{md(["站点","指标","完整门胜优化静态","完整门胜规则","完整门胜中位数"],e8)}

## 13. Experiment 09 概率预测

对 P10、P50、P90 按 pinball 独立选参、非负重排，使用末 28 天的独立校准集和有限样本顺序秩拟合 CQR。滚动测试用末 14 天校准，保留更早验证供选参。P50 在主固定窗八个设置中有 {vp} 个 MAE 低于同拟合期点 GB、有 {vz} 个低于恒零。主同站测试行的 {central_zeros:.2%} 输出 P50=0。低 MAE 与恒零对照间的关系可直接核对。P50、点 GB、零值和季节基准逐键对齐。

{cov_table}

[官方测试区间](../results/figures/probabilistic_forecasting/02_prediction_intervals.png) · [分组覆盖与宽度](../results/figures/probabilistic_forecasting/07_demand_regime_intervals.png) · [源校准跨站区间](../results/figures/probabilistic_forecasting/10_cross_site_coverage.png)

## 14. 最终同键模型比较

固定窗每站的回顾性最低分如下，含恒零、已出版的模型变体和 Experiment 09 点预测。截止同键比较并未改变各实验拟合窗口。并列及对应协议见 CSV 原精度。最低训练模型一列仅排除两项免拟合基准。

{md(["站点","跨度","最低 MAE","MAE","最低 RMSE","RMSE","最低 MAE 训练模型","训练模型 MAE"],wrank)}

恒零和 Seasonal Naive 的比较在所有跨度与迁移目标上均按测试键保存。两站 1h 示例给出正值识别与计数误差之间的区别。

{md(["站点","n","零值比例","恒零 MAE","季节 MAE","恒零 RMSE","季节 RMSE","恒零正值 MAE","季节正值 MAE","恒零正值召回"],by_site)}

恒零每小时输出零，均值偏差等于负的平均归档需求。它在有正值标签时召回为 0。点预测发生诊断使用严格大于 0 的统一阈值，未进行测试阈值优化。恒零无预测正类时精度采用 0 的记录惯例，同时标明未定义。低总体 MAE 无法支持把恒零视为能识别活动的运营预测。

[零值和正值基准图](../results/figures/stage3_final/constant_zero_demand_regimes.png) · [完整模型比较 CSV](../results/metrics/stage3_final/consolidated_point_comparison.csv)

## 15. 消融与时间稳健性

Experiment 08 的门控输入消融在 Caltech 观察到去日历的改进，JPL 去掉 Hurdle 则削弱表现。三段预设滚动窗确认结果随时期变化。下表点模型列按每站四个跨度计相对 Seasonal Naive 的 MAE 获胜数，CQR 列给出同窗四跨度的覆盖范围。两类分析训练时窗有重叠，区块重采样的 24h 及 168h 区间仅反映所观察时段。

{md(["窗口","站点","Hurdle","季节残差","完整门","CQR 总体覆盖范围"],rollrows)}

## 16. 研究问题总结

- **RQ1** 四个跨度的最优模型和误差排序会变化，跨度大小不能单独决定误差。相同测试键保留了跨跨度可比性。
- **RQ2** 均值类、分位数类、门控与季节基准各自优化不同目标。复杂模型的优势集中在特定站点、跨度或损失。
- **RQ3** 归档零非常常见，压低总体 MAE 的方法可能牺牲活动识别。应同时看正值误差、召回、预测偏差和分组区间。
- **RQ4** 源站和目标站分布不同，跨站偏差与损失方向不一致。使用目标历史能检验有限的参数迁移，时区未知限制对日历可迁移性的解释。
- **RQ5** 动态权重能利用某些需求分组中的专家互补，但简单权重常取得更好的 RMSE，且第三个窗口无 MAE 优势。中位数诊断进一步表明组件目标影响排序。
- **RQ6** 分位数给出了逐小时区间，CQR 把校准分数映射到明确的调整量。离散零值、时间依赖和迁移使经验覆盖随窗口与分组变化。宽度、区间分数和正值覆盖应联合判断。

## 17. 实际研究贡献

1. 以可核查的归档目标与发布时点构造两站、四跨度的直接滚动预测流程。
2. 通过共同目标键、匹配对照和同源拟合隔离，系统比较模型和跨站迁移。
3. 将恒零、需求分组、Hurdle、季节修正和时间 OOF 组合置于可复查的实验框架，展示收益与失败时期。
4. 以时间隔离的 CQR、同拟合期点模型、区块区间及分组覆盖记录预测不确定性。

## 18. 局限与审查

{md(["主题","证据","影响","后续核验"],[[r[k] for k in ["topic","evidence","implication","next_validation"]] for r in LIMITATIONS])}

{md(["实验","核验方式","状态","范围"],aud)}

## 19. 结论

项目完成了从小时档案输入、点预测、跨站、需求分组到概率区间的闭环。模型选择依赖业务关注的损失；应用区间前应确认关心分组、时间段和站点的经验覆盖。最终排行榜描述当前两站窗口，仍保留各研究的实验差异。

## 20. 后续研究

优先取得可信会话时间及运营元数据，并对更多站点和时期作冻结评估。后续会话级阶段先验证原始遥测和关联键，再比较短前缀的简单模型与小型序列编码器。具体依赖与首个试验建议见 [Stage 4 就绪评估](STAGE4_READINESS_ASSESSMENT.md)。
"""


INDEX = """# Stage 3 Research Index

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
"""

READINESS = """# Stage 4 就绪评估

- **可复用。** Stage 2 会话路径和源时钟起点清单、Stage 3 小时对账与来源 SHA-256、标签发布时间约束、按时间和站点分割的验证框架，以及点和区间的比较流程。小时结果可作为外部基线。
- **首轮需核验的原始字段。** 归档文件名中的站点、设备/端口、会话标识候选与开始时间，记录内带偏移的时间戳，充电状态，电流、pilot current、电压、功率和累计能量（如存在）。逐档检验列名、单位、设备标识是否稳定、起止时间、采样率、重复/逆序时间、缺测、设备断连、长间隔、能量重置及会话之间重叠。通过可信 `sessionID` 或 `connectionTime` 做关联对账，外部时间证据优先。
- **充分性。** 现有会话索引支持定位源文件、构建分组和检查归档覆盖。只有索引与小时计数无法提供 CNN、LSTM 或 TCN 的输入张量。历史最小抽查已经发现极短会话、同秒多记录及空字段，未知完整样本的可用比例和标签质量。全量序列适用性尚待验证。
- **先冻结独立验证条件。** 分层抽样覆盖两个站点、设备、长短会话和零值附近区间。定义去重及补齐策略、最短前缀、能量标签的观测完整性、右删失和时点可用约束。会话及设备重复应按时间隔离并保留同站及跨站测试，变换参数只用训练样本。
- **推荐首个实验。** 在完整且可对账的已开始会话中，使用首个可用 5 或 10 分钟预测会话剩余时长。先固定前缀和 MAE/RMSE，与站点/时段中位数及简单特征回归做同样本比较。通过数据验证后再加入小型 CNN 或 TCN 检验增益。输入不得使用标签时点之后的会话片段。

本文件记录后续可行性及前置条件。此阶段只做归档元数据一致性检查。
"""


def integration(out):
    def prob(name, **kwargs):
        return read(ROOT / PROB_METRICS / (name+".csv"), **kwargs)
    def tables(name, **kwargs):
        return read(ROOT / PROB_TABLES / (name+".csv"), **kwargs)
    project = Project()
    interval, quantiles, point = prob("interval_metrics"), prob("quantile_metrics"), prob("point_metrics")
    pred=tables("probabilistic_predictions",parse_dates=["target_hour"])
    timing, changes, crossing=tables("temporal_verification"),tables("calibration_adjustments"),prob("quantile_constraint_summary")
    merged, winner, paired, adapt=consolidate(point,timing)
    rolling_point=read(ROOT / "results/metrics/adaptive_hybrid/rolling_window_comparison.csv")
    audit=read(ROOT / FINAL_TABLES / "experiment_replay_audit.csv")
    assert audit.status.eq("passed").all()
    const=read(ROOT / FINAL_METRICS / "constant_zero_vs_seasonal_naive.csv")
    for row in const.itertuples(index=False):
        moments=merged.loc[merged.case.eq(row.case) & merged.horizon_hours.eq(row.horizon_hours) & merged.model.eq(row.model)]
        assert len(moments) and np.allclose(moments.MAE,row.MAE,atol=1e-13,rtol=0)
    for w in winner.itertuples(index=False):
        g=merged.loc[merged.case.eq(w.case)&merged.horizon_hours.eq(w.horizon_hours)]
        assert w.overall_MAE==min(g.MAE) and w.overall_RMSE==min(g.RMSE)
    prior = read(ROOT / FINAL_TABLES / "previous_stage2_stage3_sha256.csv")
    raw = read(ROOT / FINAL_TABLES / "raw_archive_metadata.csv",usecols=["path"])
    outputs = {
        FINAL_METRICS / "consolidated_point_comparison.csv":merged,
        FINAL_METRICS / "best_mae_rmse_by_site_and_horizon.csv":winner,
        FINAL_METRICS / "matched_same_site_vs_cross_site.csv":paired,
        FINAL_METRICS / "adaptive_vs_nonadaptive.csv":adapt,
        FINAL_METRICS / "probabilistic_interval_comparison.csv":interval,
        FINAL_METRICS / "rolling_probabilistic_robustness.csv":interval.loc[interval.window.ne("official")],
        FINAL_METRICS / "rolling_point_robustness.csv":rolling_point,
        FINAL_METRICS / "major_limitations.csv":pd.DataFrame(LIMITATIONS),
    }
    reports={
        Path("reports/09_probabilistic_forecasting.md"):probabilistic_report(interval,quantiles,point,changes,crossing,pred,
                                                                            prob("interval_block_bootstrap"),
                                                                            project_cfg()),
        Path("reports/STAGE3_FINAL_REPORT.md"):closure_report(project,merged,winner,paired,adapt,interval,point,pred,audit),
        Path("reports/STAGE3_RESEARCH_INDEX.md"):INDEX,
        Path("reports/STAGE4_READINESS_ASSESSMENT.md"):READINESS,
        Path("reports/STAGE3_COMPLETION_CHECKLIST.md"):f"""# Stage 3 完成清单

- [x] 恒零基准覆盖两站、四跨度及同站与跨站目标，保存零/正值分组与发生指标。
- [x] Experiment 09 分位数与点对照覆盖主固定窗口并使用相同目标时点。
- [x] 单独划分校准，严格清除未释放标签，冻结模型和有限样本调整。
- [x] 点、分位数、区间和零/正值指标均可从保存的预测重算。
- [x] 三个有效历史滚动测试、24h 和 168h 区块统计。
- [x] 复核 Experiments 01–09，包含历史点预测逐键评分回读及重要训练重跑。
- [x] 对保存的概率结果进行独立计算验证和干净进程重放。
- [x] 完成最终研究报告、研究索引和 Stage 4 就绪评估。
- [x] 复核 {len(prior):,} 个历史 Stage 2/3 文件 SHA-256，旧指标与预测均保留。
- [x] 复核 {len(raw):,} 个原始树文件的路径、大小与纳秒修改时间，归档内容未打开。
- [x] 保存配置、版本、可复查的分割哈希及冻结清单。
- [x] 记录时区、覆盖、分组覆盖、数据范围及时间依赖限制。

复现审查状态与浮点容差见 `results/tables/stage3_final/experiment_replay_audit.csv`。最终清单为 `results/tables/stage3_final/final_artifact_manifest.csv`。原始归档的核查为元数据级。冻结为文件清单和 SHA-256 校验，不改变文件权限。
""",
    }
    for path,text in reports.items():
        target=out/path
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("x",encoding="utf-8",newline="\n") as stream:
            stream.write(text)
    for path,frame in outputs.items():
        publish(frame,out/path)
    figures=probabilistic(out,pred,point,interval,quantiles)+baseline(out,const)
    return [*outputs,*reports,*figures]


def project_cfg():
    from stage3_context import config
    return config("config/probabilistic_forecasting.yaml")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    from shutil import copyfileobj
    with TemporaryDirectory(prefix=".stage3-report-",dir=ROOT) as tmp:
        draft=Path(tmp)
        paths=integration(draft)
        for path in paths:
            final=ROOT/path
            if args.verify:
                assert (draft/path).read_bytes()==final.read_bytes(),str(path)
            else:
                final.parent.mkdir(parents=True,exist_ok=True)
                with (draft/path).open("rb") as a,final.open("xb") as b:
                    copyfileobj(a,b)
        print(f"{'Reproduced' if args.verify else 'Published'} {len(paths)} consolidated tables, reports and plots.",flush=True)
