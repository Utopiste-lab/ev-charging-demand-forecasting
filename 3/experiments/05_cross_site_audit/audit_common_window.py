"""Cross-site preparation: inspect, validate, and report. No estimator fit."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "experiments/02_feature_validation")]

from compare_site_distributions import FIGURE_NAMES, describe_demand, describe_features, plot_distributions
from src.feature_engineering import HOUR, feature_catalog
from src.forecasting_dataset import load_hourly
from src.temporal_split import calendar_windows, common_view
from validate_cross_site_features import validate_transfer
from validate_cross_site_splits import site_ledger, summarize_direction, timestamp_hash


COMMON = "results/tables/cross_site_common_window.csv"
SPLIT = "results/tables/cross_site_split_summary.csv"
FEATURES = "results/tables/cross_site_feature_compatibility.csv"
DISTRIBUTION = "results/tables/cross_site_distribution_comparison.csv"
CHECKS = "results/tables/cross_site_audit_checks.csv"
REPORT = "reports/05_cross_site_experimental_design.md"
LOG = "logs/phase6_cross_site_audit.log"
CHART_DIR = "results/figures/cross_site"
FIGURES = tuple(f"{CHART_DIR}/{name}" for name in FIGURE_NAMES)
TABLES = (COMMON, SPLIT, FEATURES, DISTRIBUTION, CHECKS)
OUTPUT = (*TABLES, *FIGURES, REPORT, LOG)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_yaml(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def independent_window(hourly, expected_common, expected_test):
    by_site = {site: group.sort_values("timestamp") for site, group in hourly.groupby("site")}
    assert set(by_site) == {"Caltech", "JPL"}
    indices = {site: pd.DatetimeIndex(frame.timestamp) for site, frame in by_site.items()}
    first = max(keys.min() for keys in indices.values())
    last = min(keys.max() for keys in indices.values())
    common_keys = pd.date_range(first, last, freq=HOUR)
    test_keys = pd.date_range(expected_test["start"], expected_test["end"], freq=HOUR, name="timestamp")
    common, start, end, shared_partial = common_view(hourly)
    assert (first, last) == (start, end)
    assert (str(first), str(last), len(common_keys)) == (
        expected_common["start"], expected_common["end"], expected_common["hours"])
    assert len(test_keys) == expected_test["hours"]
    out = []
    for site, frame in by_site.items():
        index = indices[site]
        restrict = index[(index >= first) & (index <= last)]
        assert index.is_unique and index.equals(pd.date_range(index.min(), index.max(), freq=HOUR))
        assert restrict.equals(common_keys) and test_keys.difference(restrict).empty
        assert test_keys.intersection(shared_partial).empty
        counts = frame.session_start_count
        assert np.isfinite(counts).all() and counts.ge(0).all() and counts.mod(1).eq(0).all()
        assert frame.is_first_hour.eq(frame.timestamp.eq(index.min())).all()
        assert frame.is_last_hour.eq(frame.timestamp.eq(index.max())).all()
        flags = frame.set_index("timestamp")
        out.append({
            "site": site, "full_hours": len(index), "full_start": index.min(), "full_end_inclusive": index.max(),
            "full_missing_hours": 0, "full_duplicate_hours": 0, "common_start": first, "common_end_inclusive": last,
            "common_hours": len(restrict), "common_missing_hours": 0, "common_duplicate_hours": 0,
            "counts_valid": True, "common_start_site_flagged": bool(flags.at[first, "is_first_hour"]),
            "common_end_site_flagged": bool(flags.at[last, "is_last_hour"]),
            "shared_partial_hours": "|".join(map(str, sorted(shared_partial))),
            "test_start": test_keys.min(), "test_end_inclusive": test_keys.max(), "test_hours": len(test_keys),
            "test_missing_hours": len(test_keys.difference(index)),
            "common_timestamps_sha256": timestamp_hash(restrict), "test_timestamps_sha256": timestamp_hash(test_keys),
            "clock_basis": frame.clock_basis.iloc[0], "timezone_status": frame.timezone_status.iloc[0],
            "coverage_status": frame.coverage_status.iloc[0],
        })
    assert indices["Caltech"].intersection(indices["JPL"]).equals(common_keys)
    return common, first, last, shared_partial, test_keys, pd.DataFrame(out)


def check_existing_summary(splits, prior, single_site, training, test_keys):
    assert len(prior) == 24 and len(single_site) == len(training) == 12
    cut = splits.loc[splits.partition.ne("final_refit")]
    keyed = cut.merge(prior, on=["direction", "horizon_hours", "partition"], suffixes=("", "_previous"),
                      validate="one_to_one")
    assert len(keyed) == 24
    assert keyed.eligible_n.eq(keyed.eligible_n_previous).all()
    assert pd.to_datetime(keyed.target_start).eq(pd.to_datetime(keyed.eligible_target_start)).all()
    assert pd.to_datetime(keyed.target_end_inclusive).eq(pd.to_datetime(keyed.eligible_target_end_inclusive)).all()
    assert single_site.loc[single_site.partition.eq("test"), "eligible_n"].eq(len(test_keys)).all()
    assert single_site.loc[single_site.partition.eq("test"), "target_timestamps_sha256"].eq(timestamp_hash(test_keys)).all()
    full = training.groupby("horizon_hours").n_final_refit.agg(["min", "max"])
    assert full["min"].eq(full["max"]).all()
    for h, rows in splits.loc[splits.partition.eq("final_refit")].groupby("horizon_hours"):
        assert rows.eligible_n.nunique() == 1 and rows.eligible_n.iat[0] < full.at[h, "min"]


def research_report(common, splits, distribution, checks, refit_reference):
    both = common.iloc[0]
    boundaries, refits, data, rhythms, historical = [], [], [], [], []
    for (direction, h), block in splits.groupby(["direction", "horizon_hours"], sort=False):
        parts = block.set_index("partition")
        train, val, test, fit = (parts.loc[key] for key in ("train", "validation", "test", "final_refit"))
        formatted = [f"{r.target_start} 至 {r.target_end_inclusive}（{r.eligible_n:,}）" for r in (train, val, test)]
        boundaries.append(f"| {direction} | {h} | " + " | ".join(formatted) + " |")
        refits.append(f"| {direction} | {h} | {fit.eligible_n:,} | {train.excluded_label_release} / "
                      f"{val.excluded_label_release} | {fit.last_label_available_at} | {test.first_issue} | "
                      f"{fit.target_timestamps_sha256[:12]} |")
    overview = distribution.loc[distribution.scope.isin(["full_archive", "common_archive", "common_refit_1h",
                                                        "test_targets"]) & distribution.group_type.eq("overall")]
    for row in overview.itertuples():
        data.append(f"| {row.scope} | {row.site} | {row.n:,} | {row.mean:.4f} | {row.median:g} | "
                    f"{row.std_ddof1:.4f} | {row.max:g} | {row.zero_fraction:.2%} | "
                    f"{row.positive_fraction:.2%} | {row.p95:g} / {row.p99:g} |")
    for (scope, site), block in distribution.loc[distribution.scope.isin(["common_archive", "test_targets"])].groupby(["scope","site"]):
        hour = block.loc[block.group_type.eq("hour")].sort_values("mean", ascending=False).iloc[0]
        weekday = block.loc[block.group_type.eq("weekday")].sort_values("mean", ascending=False).iloc[0]
        groups = block.loc[block.group_type.eq("day_type")].set_index("group")
        rhythms.append(f"| {scope} | {site} | {int(hour.group):02d}:00 ({hour['mean']:.3f}) | "
                       f"{'一二三四五六日'[int(weekday.group)]} ({weekday['mean']:.3f}) | "
                       f"{groups.at['weekday','mean']:.3f} | {groups.at['weekend','mean']:.3f} |")
    examples = distribution.loc[distribution.horizon_hours.eq(1)
        & distribution.variable.isin(["lag_1", "rolling_mean_24"])]
    for r in examples.itertuples():
        historical.append(f"| {r.direction} | {r.scope} | {r.site} | {r.variable} | "
                          f"{r.mean:.4f} | {r.std_ddof1:.4f} | {r.zero_fraction:.2%} |")
    lookup = overview.set_index(["scope", "site"])
    ca, jpl = [lookup.loc[("common_archive", s)] for s in ["Caltech", "JPL"]]
    ca_test, jpl_test = [lookup.loc[("test_targets", s)] for s in ["Caltech", "JPL"]]
    labels = {
        "A": "扰动目标站所有当前测试标签，来源站三套拟合输入逐行不变。",
        "B": "逐行核对目标站滞后和滚动窗的取值及完成时间，并对代表起点执行未来扰动及标签扰动。",
        "C": "核实训练、验证标签的发布时间约束和各跨度剔除数量。",
        "D": "两方向和正式同站 1/3/6/24h 测试日历逐键相同。",
        "E": "拟合数据结构仅接收源站的监督输入，逐行核对标签和特征的来源。尚未调用估计器。",
        "F": "逐目标核对基准为目标站前一日记录，并验证两站参考实际存在差异。",
        "G": "两站完全一致的 34 列 schema、类型和日历定义，无站点 ID 或非有限输入。",
    }
    validation = "\n".join(f"| {key} | {description} | {checks.loc[checks.test.eq(key),'result'].eq('PASS').sum()}/8 |"
                           for key, description in labels.items())
    previous = refit_reference.groupby("horizon_hours").n_final_refit.first().tolist()
    return f"""# 跨站点泛化实验设计与验证

**建议进入训练阶段。** Caltech→JPL 和 JPL→Caltech 均具备同一套按小时配对的目标、足够预热历史与经时间剔除的源站样本。本阶段完成数据和接口验证，未拟合模型，也未计算跨站预测指标。

## 共同时间轴与边界

独立从处理后数据的两个站点索引求得共同范围 **{both.common_start} 至 {both.common_end_inclusive}**，含首尾共 **{both.common_hours:,} 小时**。每站全范围与共同范围均无缺失或重复时刻，小时网格连续，需求均为非负整数。Caltech 共 {common.loc[common.site.eq('Caltech'),'full_hours'].iat[0]:,} 小时，JPL 共 {common.loc[common.site.eq('JPL'),'full_hours'].iat[0]:,} 小时。共同起点只被 JPL 标记为原始首小时，两站都标记共同终点为原始末小时。两个共同边界按联合标记处理。正式样本排除这些边界为目标或必要历史输入的情形。原视图保留全部值及标记。

正式测试目标为 **{both.test_start} 至 {both.test_end_inclusive}**，每站、每方向、每跨度均为同一 **{both.test_hours:,} 小时**，与已发布 Caltech 同站实验逐键相符。共同历史足以构造三个滞后、24 小时窗口和季节基准。第一个可用训练目标随跨度移动，最早用于构造历史的共同小时不跨越共同范围。已识别的首尾标记表示可能仅部分覆盖，站点其余小时的运营覆盖仍未知。

## 双向时间切分

下表时间均为目标小时起点，末小时包含在区间内。训练和验证来自箭头左侧，测试来自右侧。继承验证起点 2020-02-24 23:00 与测试起点 2020-07-15 00:00，各方向使用相同日历。发布时点按目标小时减 h−1 小时。源站训练标签须在第一个验证发布时点前或同刻可用，源站选参和最终拟合标签须在第一个目标测试发布时点前或同刻可用。

| 方向 | h | 源站训练 | 源站验证 | 目标站测试 |
|---|---:|---|---|---|
{chr(10).join(boundaries)}

| 方向 | h | 源站最终拟合 | 分界剔除 训练 / 验证 | 最后拟合标签可用时点 | 首次目标测试发布 | 最终时间戳哈希前缀 |
|---|---:|---:|---:|---|---|---|
{chr(10).join(refits)}

每个计划只构造一份源站最终拟合样本，供 Ridge、Random Forest 和 Gradient Boosting 共用。完整哈希按有序时间戳每行一个 ISO 字符串计算，保存在切分表的 `final_refit` 行。Caltech 既有同站实验的最终拟合样本为 {"、".join(f"{n:,}" for n in previous)}，因拥有更早历史而比跨站共同期长。若后续量化迁移代价，应在同一共同训练窗口另拟合目标站自身参照，再按相同目标比较。原 Caltech 已发布分数可提供背景。

## 特征、隔离和模型方案

- 使用已有 34 列和既有缺失处理。4 列需求历史分别取 y(t)、y(t−23)、y(t−167) 与截至 t 的过去 24 小时均值。30 列目标小时的源时钟日历由已知时间索引生成。两个站点的字段及顺序、类型和定义一致，可用行无缺失，不使用站点 ID。
- 源站单独提供训练、验证与最终拟合的 X 和 y。目标站预测入口只接收特征、发布时间和独立季节基准。预测当前目标时，只有在发布时已完成且已到达的目标站需求参与历史特征。滚动起点允许较早测试观测用于之后的预测；固定起点会在同一个截止点冻结历史。本方案为使用目标历史的跨站迁移。无目标历史的严格 zero-shot 需要另设仅含可提前获知输入的协议。
- 目标站 Seasonal Naive 使用 y_target(t+h−24)。1/3/6/24h 的参考分别为目标站 t−23、t−21、t−18、t。方向改变时同步更换基准历史来源。
- 下一阶段可复用正式统一流水线中的候选选择和重新拟合组件，使用既有参数候选和相同种子。所有学习模型仅以源站验证 MAE 选参，最终只在已释放的源站训练加验证上重拟合。Ridge 的 StandardScaler 从相应源站拟合样本学习，并冻结后用于转换目标特征。树模型沿用原始单位。当前不增加新归一化，任何目标站测试统计均不得用于缩放、选特征、调参或校准。
- 候选选定和最终拟合后，冻结模型、目标键和预测，再连接目标真值，使用 `src/evaluation.py` 计算 MAE、RMSE、R²。交付 2 方向 × 4 提前量 × 4 模型的 32 个主组合，负预测的非负截断作次要变体。逐模型记录实际拟合键并比对本审查的 SHA-256。对目标历史基准使用相同目标键。

## 两站记录分布

标准差采用样本定义。分位数使用线性插值，完整 CSV 另含 P10/P25/P50/P75/P90、每小时和每星期分组、周末分组及计数区间频率。共同归档描述含边界原始行，正式测试和 1h 拟合统计按可用键。描述性测试统计仅用于记录分布，未反馈至模型设计或归一化。

| 口径 | 站点 | 小时 | 均值 | 中位数 | 标准差 | 最大值 | 零值 | 正值 | P95 / P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(data)}

共同归档内，JPL 均值比 Caltech 高（{jpl['mean']:.4f} 对 {ca['mean']:.4f}），零值比例也更高（{jpl.zero_fraction:.2%} 对 {ca.zero_fraction:.2%}）。这表明只看平均值会遗漏分布差异。共同测试期的均值为 JPL {jpl_test['mean']:.4f}、Caltech {ca_test['mean']:.4f}，零值比例分别为 {jpl_test.zero_fraction:.2%} 和 {ca_test.zero_fraction:.2%}。最大值、分位数和标准差提供不同的量级信息。高峰和长零时段均保留，来源与真实运营状态无法由这些统计确认。

| 范围 | 站点 | 源时钟日内均值最高 | 星期均值最高 | 工作日均值 | 周末均值 |
|---|---|---|---|---:|---:|
{chr(10).join(rhythms)}

历史输入也可能偏移。下表展示 1h 的需求滞后和滚动均值示例，对应各方向的源站最终拟合输入和目标站测试输入。完整表覆盖四个历史特征、四个跨度、双向和两种样本角色。此对照同时改变站点与时间，不能把全部差异归因于站点。上面的共同归档和匹配测试窗口可辅助分辨。

| 方向 | 角色 | 站点 | 特征 | 均值 | 标准差 | 零值 |
|---|---|---|---|---:|---:|---:|
{chr(10).join(historical)}

## 自动化验证

| 测试 | 核验内容 | 通过计划 |
|---|---|---:|
{validation}

验证共 {len(checks)} 项。A 和 E 证明准备阶段的数据路由与隔离，实际训练阶段仍须记录估计器收到的样本并重新执行约束。逐计划证据、特征偏移和切分哈希随 CSV 交付。运行方法如下。

```powershell
python -B experiments/05_cross_site_audit/audit_common_window.py --verify
```

## 未解决事项

两站仍使用 `source_clock`，时区和运营覆盖未知。源时钟标签相同无法证明处于同一当地时间，日历特征迁移可能受未识别的时钟差异影响。每小时标签表示归档记录的会话开始数。滚动预测假设上一完成小时的计数已在发布时到达；若发现上报延迟，需重新验证两个站点的特征与标签可用时间，包括 24h Seasonal Naive 恰在发布时可用的参考。分布差异可能影响迁移表现，实际泛化性能需要训练和留出评估确认。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Recheck without replacing saved output")
    args = parser.parse_args()
    for name in OUTPUT:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or (path.exists() and not args.verify):
            raise FileExistsError(f"Output outside Stage 3 or already present: {path}. Use --verify.")
    cfg_path = "config/cross_site.yaml"
    cfg = load_yaml(cfg_path)
    framework = load_yaml(cfg["framework"])
    multi = load_yaml(cfg["multi_horizon_reference"])
    baseline = load_yaml(framework["legacy_config"])
    contract_path = f"config/{baseline['data_contract']}"
    references = cfg["references"]
    inputs = [cfg_path, cfg["framework"], cfg["multi_horizon_reference"], framework["legacy_config"], contract_path,
              framework["data_view"], framework["provenance"], *references.values(),
              "src/feature_engineering.py", "src/forecasting_dataset.py", "src/temporal_split.py", "src/evaluation.py"]
    before = {item: sha(ROOT / item) for item in inputs}
    policy, horizons = cfg["boundary_policy"], cfg["horizons_hours"]
    assert policy == multi["boundary_policy"] == "exclude_partial"
    assert horizons == multi["horizons_hours"] == framework["horizons_hours"] == [1, 3, 6, 24]
    assert [(p["source"], p["target"]) for p in cfg["directions"]] == [("Caltech", "JPL"), ("JPL", "Caltech")]
    hourly, provenance = load_hourly(ROOT / framework["data_view"], ROOT / framework["provenance"])
    contract = load_yaml(contract_path)
    assert all(provenance[key] == contract[key] for key in ("target", "clock", "coverage"))
    overlap, first, last, partial, expected_test, common = independent_window(
        hourly, cfg["expected_common"], cfg["expected_test"])
    settings = baseline["features"]
    period = baseline["models"]["seasonal_naive"]["period_hours"]
    windows = calendar_windows(baseline["split"])
    fit_key_1h = None
    split_rows, columns, checks, historical = [], [], [], []
    for h in horizons:
        datasets, ledgers = {}, {}
        for site, rows in overlap.groupby("site", sort=True):
            datasets[site], ledgers[site] = site_ledger(rows, settings, h, period, policy, partial, windows)
        history = [name for name, kind, *_ in feature_catalog(settings, h, period) if kind == "history"]
        for direction in cfg["directions"]:
            source, target = direction["source"], direction["target"]
            tag = f"{source}->{target}"
            fit, target_X, schema, results = validate_transfer(overlap, direction, h, datasets, ledgers, settings,
                                                                period, policy, partial, windows, expected_test)
            split_rows.extend(summarize_direction(h, tag, fit, ledgers[source], ledgers[target], windows, first, last))
            columns.extend(schema)
            checks.extend(results)
            historical.extend(describe_features(h, tag, fit, target_X, history))
            if h == 1:
                if fit_key_1h is not None:
                    assert fit_key_1h.equals(fit.final_refit.X.index)
                fit_key_1h = fit.final_refit.X.index
            print(f"{tag} {h}h: A–G passed", flush=True)
    splits, compatibility, tests = map(pd.DataFrame, (split_rows, columns, checks))
    assert len(splits) == 32 and len(compatibility) == 272 and len(tests) == 56 and tests.result.eq("PASS").all()
    prior = pd.read_csv(ROOT / references["prior_cross_split"]).query(
        "scope == 'cross_site' and boundary_policy == @policy")
    assert len(prior) == 24
    same = pd.read_csv(ROOT / references["same_site_split"])
    refits = pd.read_csv(ROOT / references["same_site_refit"])
    check_existing_summary(splits, prior, same, refits, expected_test)
    distribution = pd.DataFrame(describe_demand(hourly, overlap, expected_test, fit_key_1h) + historical)
    assert {key: sha(ROOT / key) for key in inputs} == before
    with TemporaryDirectory(prefix=".cross_site_audit_", dir=ROOT) as directory:
        temp = Path(directory)
        for name in OUTPUT:
            (temp / name).parent.mkdir(parents=True, exist_ok=True)
        for path, frame in zip(TABLES, (common, splits, compatibility, distribution, tests), strict=True):
            frame.to_csv(temp / path, index=False)
        plot_distributions(pd.read_csv(temp / DISTRIBUTION, float_precision="round_trip"), temp / CHART_DIR)
        (temp / REPORT).write_text(research_report(common, splits, distribution, tests, refits), encoding="utf-8")
        now = datetime.now(timezone(timedelta(hours=8))).isoformat()
        (temp / LOG).write_text("\n".join([
            f"{now} Phase 6 cross-site preparation; zero model fits and zero forecast metrics.",
            f"Checked Stage 3 inputs before/after SHA-256: {json.dumps(before, sort_keys=True)}",
            f"Common index: {first} to {last}; {common.common_hours.iat[0]} complete matched keys.",
            f"Common test: {expected_test[0]} to {expected_test[-1]}, {len(expected_test)} exact keys.",
            "The Phase 5 fit reference is stored under results/metrics, as configured.",
            "All 8 direction/horizon plans passed A–G. No Stage 2 or raw content reads in this audit runner.",
            tests.to_csv(index=False).rstrip(), "",
        ]), encoding="utf-8")
        if args.verify:
            for name in (*TABLES, *FIGURES, REPORT):
                assert sha(ROOT / name) == sha(temp / name), name
            print("Reproduced all five CSV tables, four figure hashes and the design report.")
        else:
            for name in OUTPUT:
                path = ROOT / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as out, (temp / name).open("rb") as source:
                    shutil.copyfileobj(source, out)


if __name__ == "__main__":
    main()
