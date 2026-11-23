"""Validate feature causality and common chronological splits without expanding experiments."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import re
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

from src.feature_engineering import HOUR, feature_catalog
from src.forecasting_dataset import build_dataset, load_hourly
from src.temporal_split import PARTITIONS, calendar_windows, common_view, split_dataset
from test_feature_validation import (test_a_future_perturbation, test_b_feature_availability,
                                     test_c_temporal_split, test_d_target_leakage,
                                     test_e_historical_replay)


def summarize(ledger, windows, context):
    for partition, (start, stop) in windows.items():
        part = ledger.loc[ledger.partition.eq(partition)]
        used = part.loc[part.status.eq("eligible")]
        counts = part.status.value_counts()
        yield {**context, "partition": partition, "policy_window_start": start,
               "policy_window_end_inclusive": stop - HOUR, "candidate_targets": len(part),
               "warmup": counts.get("warmup", 0), "excluded_partial_target": counts.get("boundary_target", 0),
               "excluded_partial_history": counts.get("boundary_history", 0),
               "excluded_label_release": counts.get("label_release_purge", 0),
               "eligible_n": len(used), "eligible_target_start": used.index.min(),
               "eligible_target_end_inclusive": used.index.max(),
               "first_issue": used.forecast_issued_at.min(), "last_issue": used.forecast_issued_at.max(),
               "last_label_available_at": used.target_end.max()}


def render_report(summary, windows, first, last, n_overlap, replay, results, accepted, checked_rows, configs):
    main = summary.query("boundary_policy == 'exclude_partial'")
    lines = []
    for key, group in main.groupby(["scope", "direction", "horizon_hours"], sort=False):
        scope, direction, h = key
        n = group.set_index("partition").eligible_n
        lines.append(f"| {scope} | {direction} | {h} | {n.train:,} | {n.validation:,} | {n.test:,} |")
    removed = []
    for key, group in main.groupby(["scope", "direction"], sort=False):
        counts = group[["excluded_partial_target", "excluded_partial_history"]].sum()
        removed.append(f"| {key[0]} | {key[1]} | {counts.iloc[0]:,} | {counts.iloc[1]:,} |")
    score_rows = []
    for model, (full, subset) in results.items():
        for note, s, count in [("历史完整切分", full, configs["split"]["test"]["n"]),
                               ("历史边界敏感性", subset, accepted)]:
            score_rows.append(f"| {model} | {note} | {count:,} | {s['MAE']:.6f} | {s['RMSE']:.6f} | {s['R2']:.6f} |")
    train, validation, test = (windows[name] for name in PARTITIONS)
    return f"""# Stage 3 特征与时间切分验证

## 预测时点与信息边界

沿用源时钟，时区和运营覆盖仍未知。目标是 Hourly Recorded Session-Start Counts。定义 `y(s)` 为 `[s,s+1h)` 内记录到的会话开始数。令 `t` 为**最后一个已完成小时的起点**，实际发布时间 `o=t+1h`。假定无上报延迟，`y(t)` 在 `o` 可用。对每个 `h∈{{1,3,6,24}}` 单独构造标签 `y(t+h)`，其目标小时从 `t+h` 开始，结束于 `o+h`。记录中 `forecast_origin=t`、`forecast_issued_at=o`。例如最后观测为 09:00–10:00，10:00 发布，四个目标起点依次为 10:00、12:00、15:00 和次日 09:00。

Rolling-Origin 每小时更新发布时点，可使用此时已结束并已观测的历史，包含先前测试小时。Fixed-Origin 在一个发布时间固定历史，在该批预测中不获得随时间揭晓的真实值。本框架采用 Rolling-Origin。

## 34 列输入和 Seasonal Naive

复核了 Stage 2 的 `make_features` 实现。Stage 2 的行索引是目标区间起点 `s`。其 `shift(1/24/168)` 分别取 `y(s−1)`、`y(s−24)`、`y(s−168)`，滚动列是 `shift(1).rolling(24, min_periods=24)`。以一小时的 `s=t+1` 改写后，四个历史列如下。

| 输入列 | 四个跨度共享的历史定义 | 按目标索引的读取偏移 |
|---|---|---|
| `lag_1` | `y(t)` | `shift(h)` |
| `lag_24` | `y(t−23)` | `shift(h+23)` |
| `lag_168` | `y(t−167)` | `shift(h+167)` |
| `rolling_mean_24` | `y(t−23)…y(t)` 的 24 小时均值 | `shift(h).rolling(24)` |

保留历史命名以便复现。按 `t` 锚定后，`lag_24` 实际引用 `t−23`，含义是距真实发布时间 24 小时的区间起点。该组保持历史协议的四个需求特征。其余 30 列是已知目标小时的日历信息，含 23 个小时指示列、6 个星期指示列与周末标记，均读取 `t+h` 的源时钟。标签、时间元数据与独立基准预测单独保存，供模型使用的 `X` 只有 34 列。

Seasonal Naive 独立取 `y(t+h−24)`。1/3/6/24 小时分别引用 `t−23`、`t−21`、`t−18`、`t`。这些区间均在发布时点或更早结束。24 小时基准的参考小时恰好在发布时间结束。

## 时间切分与首尾小时

目标时间统一按 Stage 2 Caltech 日历边界分配。候选训练期 `[{train[0]}, {train[1]})`，验证期 `[{validation[0]}, {validation[1]})`，测试期 `[{test[0]}, {test[1]})`。各站的实际输入覆盖及预热可缩短可用范围。训练标签须在首个验证发布时间前或同刻已完整结束，验证标签须在首个测试发布时间前或同刻结束。按下一分区起点减 `h−1` 小时校验，每处分界为 1/3/6/24 小时分别移除 0/2/5/23 个前一分区尾部目标。测试内较早完成的小时可作后续滚动历史输入。

主策略 `exclude_partial` 排除被标记的目标小时，以及历史特征或 Seasonal Naive 引用任一边界小时的候选。首尾行保留于原视图。没有足够历史的样本记为预热；超出指定目标日历的小时可用于历史，但不进入任一目标分区。敏感性策略 `retain_sensitivity` 允许使用带标记的边界小时，其他规则一致。下表排除统计按四个跨度与相应分区累计，计数是候选样本次数。

| 范围 | 站点或方向 | 排除的边界目标 | 排除的边界历史 |
|---|---|---:|---:|
{chr(10).join(removed)}

## 跨站共同窗口

共同小时范围为 **{first} 至 {last}（含末小时，共 {n_overlap:,} 小时）**。双向方案的候选训练、验证、测试使用上述共同日历。训练与验证归源站，测试归目标站。历史限制在共同范围内。任一站原始边界落在共同范围时，两站共同标记该小时，保证主策略下两个方向使用相同目标日期。主方案共同目标测试窗口为 **{test[0]} 至 {last-HOUR}**，保留边界的敏感性窗口延伸至 {last}。逐跨度的实际训练、验证和测试日期、发布时间和标签可用时间见汇总 CSV。跨站使用目标历史的接口可沿用此框架，严格 zero-shot 未来应限制为日历输入。本阶段只验证切分。

| 范围 | 站点或方向 | h | 训练 | 验证 | 测试 |
|---|---|---:|---:|---:|---:|
{chr(10).join(lines)}

## 自动化测试与一小时回归

测试 A 扰动各分区代表起点之后的全部真实值，特征与基准不变。测试 B 对 {checked_rows:,} 个纳入样本行核对每项输入的来源和可用时间，并按时间戳独立重算数值。测试 C 全量核对目标分区、边界与标签发布时间。测试 D 单独扰动目标，核对标签变动与预测输入隔离。A–D 覆盖两站、四个时域、两种边界策略以及同站和共同窗口计划。测试 E 全量核对一小时 34 列与已验证的 Stage 2 公式，复用历史超参数重放 Caltech Ridge，未重新调参。逐时原始预测与保存基准的最大差为 Seasonal Naive {replay['Seasonal Naive']:.3g}、Ridge {replay['Ridge Regression']:.3g}。

| 预测版本 | 测试范围 | n | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
{chr(10).join(score_rows)}

此表是 Stage 2 历史回归审计。敏感性行在冻结的历史预测上排除测试末小时。现行一小时指标见 [统一训练实验](03_tree_models_1h.md)，统一实验会重新拟合所有学习模型。

在 Stage 3 根目录运行以下命令复核。依赖沿用 `experiments/01_baseline_and_ridge/requirements.txt`。

```powershell
python -B experiments/02_feature_validation/run_validation.py --check-only
```

`config/feature_validation.yaml` 引用既有小时数据和一小时配置。新增脚本按需构造内存数据集，输出两个小型审查表。没有生成重复的特征文件或修改原小时视图。
"""


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--check-only", action="store_true", help="Run all tests without writing output")
    args = cli.parse_args()
    output_paths = [
        ROOT / "results/tables/feature_availability.csv", ROOT / "results/tables/temporal_split_summary.csv",
        ROOT / "reports/02_feature_and_split_validation.md", ROOT / "logs/phase3_feature_validation.log",
    ]
    if not args.check_only and any(path.exists() for path in output_paths):
        raise FileExistsError("Phase 3 output already exists. Use --check-only to rerun validations.")
    cfg = yaml.safe_load((ROOT / "config/feature_validation.yaml").read_text("utf-8"))
    historical = yaml.safe_load((ROOT / cfg["legacy_config"]).read_text("utf-8"))
    view_path = ROOT / cfg["data_view"]
    hourly, provenance = load_hourly(view_path, ROOT / cfg["provenance"])
    horizons = cfg["horizons_hours"]
    features = historical["features"]
    period = historical["models"]["seasonal_naive"]["period_hours"]
    windows = calendar_windows(historical["split"])
    common, first, last, partial = common_view(hourly)
    stats = []
    checked_rows, plans = 0, 0
    baseline = {}
    for scope, frame in [("same_site", hourly), ("cross_site", common)]:
        for policy in cfg["boundary_policies"]:
            for horizon in horizons:
                ledgers = {}
                for site, rows in frame.groupby("site", sort=True):
                    data = build_dataset(rows, features, horizon, period, policy=policy,
                                         shared_partial_hours=partial if scope == "cross_site" else None)
                    ledger = split_dataset(data, windows)
                    counts = rows.set_index("timestamp").session_start_count.sort_index()
                    test_a_future_perturbation(counts, data, ledger, features, period)
                    test_b_feature_availability(counts, data, ledger, features, period)
                    test_c_temporal_split(data, ledger, windows, policy)
                    test_d_target_leakage(counts, data, ledger, features, period)
                    checked_rows += ledger.status.eq("eligible").sum()
                    plans += 1
                    ledgers[site] = ledger
                    if scope == "same_site" and site == historical["site"] and horizon == 1:
                        if policy == "retain_sensitivity":
                            baseline.update(data=data, ledger=ledger, counts=counts)
                        else:
                            baseline["main_ledger"] = ledger
                if scope == "same_site":
                    for site, ledger in ledgers.items():
                        stats.extend(summarize(ledger, windows, dict(scope=scope, direction=site,
                            source_site=site, target_site=site, horizon_hours=horizon, boundary_policy=policy)))
                else:
                    one, two = sorted(ledgers)
                    for src, tgt in [(one, two), (two, one)]:
                        source = ledgers[src]
                        target = ledgers[tgt]
                        pd.testing.assert_frame_equal(source, target)
                        selected = pd.concat([source.loc[~source.partition.eq("test")],
                                              target.loc[target.partition.eq("test")]]).sort_index()
                        rows = summarize(selected, windows, dict(scope=scope, direction=f"{src}->{tgt}",
                            source_site=src, target_site=tgt, horizon_hours=horizon, boundary_policy=policy))
                        for row in rows:
                            row["partition_site"] = tgt if row["partition"] == "test" else src
                            stats.append(row)
    replay, metrics, accepted = test_e_historical_replay(
        ROOT, historical, baseline["counts"],
        baseline["data"], baseline["ledger"], baseline["main_ledger"])
    summary = pd.DataFrame(stats)
    summary["partition_site"] = summary.partition_site.fillna(summary.source_site)
    counts = summary[["warmup", "excluded_partial_target", "excluded_partial_history",
                      "excluded_label_release", "eligible_n"]].sum(axis=1)
    assert counts.eq(summary.candidate_targets).all()
    availability = pd.DataFrame([
        {"horizon_hours": h, "field": name, "kind": kind, "included_in_model_X": kind in {"history", "calendar"},
         "first_hour_offset_from_t": first_offset, "last_hour_offset_from_t": last_offset,
         "earliest_available_offset_from_t": (last_offset + 1) if kind != "calendar" else None,
         "available_at_issue": kind != "label", "validation": "pass"}
        for h in horizons
        for name, kind, first_offset, last_offset in feature_catalog(features, h, period)
    ])
    availability["earliest_available_offset_from_t"] = availability.earliest_available_offset_from_t.astype("Int64")
    with view_path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == provenance["view"]["sha256"]
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    log = "\n".join([
        f"{timestamp} Stage 3 Phase 3 feature and split validation",
        f"Hourly view SHA-256 verified before and after: {provenance['view']['sha256']}",
        f"Source-clock hours: {hourly.groupby('site').size().to_dict()}",
        f"Cross-site overlap: {first} to {last}, {common.timestamp.nunique()} hours",
        f"PASS A Future perturbation and D direct target perturbation across {plans} plans",
        f"PASS B Timestamp and numerical feature availability across {checked_rows} eligible rows",
        "PASS C Exact target windows, per-horizon release purge and shared cross-site calendar",
        f"PASS E All historical 34 Caltech columns and 3,384 aligned predictions: {replay}",
        f"Published {len(availability)} feature rows and {len(summary)} split rows.",
        "Raw content reads: 0. Stage 2 archived forecasting source and predictions read only for test E.",
        "Replayed one historical Ridge fit with frozen settings. No new model fits or transfer training.",
        "",
    ])
    report = render_report(summary, windows, first, last, common.timestamp.nunique(),
                           replay, metrics, accepted, checked_rows, historical)
    if not args.check_only:
        for path, text in zip(output_paths, [availability.to_csv(index=False), summary.to_csv(index=False), report, log]):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as destination:
                destination.write(text)
    print(log, end="")


if __name__ == "__main__":
    main()
