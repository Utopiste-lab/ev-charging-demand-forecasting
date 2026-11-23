"""Render the official one-hour figures and reports from saved predictions."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np


FIGURES = (
    "results/figures/model_comparison_1h_mae.png",
    "results/figures/model_comparison_1h_rmse.png",
    "results/figures/actual_vs_predicted_1h.png",
    "results/figures/residual_comparison_1h.png",
    "results/figures/validation_vs_test_1h.png",
)
STYLE = {
    ("Seasonal Naive", "raw"): ("Seasonal Naive", "#25667C"),
    ("Ridge Regression", "raw"): ("Ridge", "#A55228"),
    ("Ridge Regression", "nonnegative"): ("Ridge Clipped", "#B77C25"),
    ("Random Forest", "raw"): ("Random Forest", "#287557"),
    ("Gradient Boosting", "raw"): ("Gradient Boosting", "#5B4494"),
    ("Gradient Boosting", "nonnegative"): ("Gradient Boosting Clipped", "#A25DA2"),
}


def save(fig, path):
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def create_figures(metrics, predictions, paths):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True, "grid.alpha": .23})
    summary = metrics.set_index(["model", "prediction_variant"])
    count = int(metrics.n_test.iloc[0])
    labels, colors = zip(*STYLE.values())
    for metric, path in zip(("MAE", "RMSE"), FIGURES[:2]):
        fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
        errors = [summary.at[k, f"test_{metric}"] for k in STYLE]
        bars = ax.barh(range(len(STYLE)), errors, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set_yticks(range(len(STYLE)), labels)
        ax.invert_yaxis()
        ax.set(title=f"Caltech one-hour test {metric} · Boundary-aware, {count:,} hours",
               xlabel=f"{metric} (recorded starts per hour)", xlim=(0, max(errors) * 1.17))
        ax.grid(axis="x")
        save(fig, paths[path])

    test = predictions.loc[predictions.partition.eq("test")]
    blocks = {(model, version): block for (model, version), block in test.groupby(["model", "prediction_variant"])}
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 7.3), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Actual and predicted session starts · Daily means of hourly test forecasts", fontsize=15)
    for ax, (key, (label, color)) in zip(axes.flat, STYLE.items()):
        daily = blocks[key].set_index("target_hour")[["actual", "prediction"]].resample("D").mean()
        ax.plot(daily.index, daily.actual, c="#252B33", lw=1.25, label="Actual")
        ax.plot(daily.index, daily.prediction, c=color, lw=1.15, label="Predicted")
        ax.set_title(label, loc="left")
        ax.grid(axis="y")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recorded starts/hour")
    for ax in axes[-1]:
        ax.xaxis.set_major_locator(dates.MonthLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter("%b"))
        ax.set_xlabel("Target day (2020 source clock)")
    axes[0, 0].legend(loc="upper left", ncols=2, frameon=False)
    save(fig, paths[FIGURES[2]])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True, layout="constrained")
    fig.suptitle("Caltech one-hour test residuals", fontsize=15)
    for ax, title in zip(axes, ("Full distribution", "Detail near zero error")):
        for key, (label, color) in STYLE.items():
            residual = np.sort(blocks[key].residual_prediction_minus_actual.to_numpy())
            ax.step(residual, np.arange(1, len(residual) + 1) / len(residual), where="post",
                    c=color, lw=1.45, label=label)
        ax.axvline(0, lw=1, c="#545960", ls=":")
        ax.set(title=title, xlabel="Prediction minus actual (recorded starts per hour)")
        ax.grid()
    axes[0].set_ylabel("Cumulative fraction of test hours")
    axes[1].set_xlim(-1, 1)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, paths[FIGURES[3]])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
    fig.suptitle("Validation and test errors · Caltech one-hour forecasts", fontsize=15)
    for ax, metric in zip(axes, ("MAE", "RMSE")):
        lo = min(metrics[f"validation_{metric}"].min(), metrics[f"test_{metric}"].min()) * .87
        hi = max(metrics[f"validation_{metric}"].max(), metrics[f"test_{metric}"].max()) * 1.1
        ax.plot([lo, hi], [lo, hi], ls=":", color="#757980", label="Equal errors")
        for key, (label, color) in STYLE.items():
            ax.scatter(summary.at[key, f"validation_{metric}"], summary.at[key, f"test_{metric}"],
                       label=label, color=color, s=52, edgecolors="white", lw=.55,
                       marker="D" if key[1] == "nonnegative" else "o")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), title=metric,
               xlabel=f"Validation {metric} (starts/hour)", ylabel=f"Test {metric} (starts/hour)")
        ax.grid()
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, paths[FIGURES[4]])


def render_reports(metrics, predictions, training, provenance, environment):
    rows, baseline_rows, configurations, diagnostics = [], [], [], []
    summary = metrics.set_index(["model", "prediction_variant"])
    test = predictions.loc[predictions.partition.eq("test")]
    for key, (label, _) in STYLE.items():
        r = summary.loc[key]
        line = (f"| {label} | {r.validation_MAE:.6f} | {r.validation_RMSE:.6f} | {r.test_MAE:.6f} | "
                f"{r.test_RMSE:.6f} | {r.test_R2:.6f} | {r.test_negative_predictions:,} |")
        rows.append(line)
        if key[0] in ["Seasonal Naive", "Ridge Regression"]:
            baseline_rows.append(line)
        subset = test.loc[test.model.eq(key[0]) & test.prediction_variant.eq(key[1])]
        err = subset.residual_prediction_minus_actual.abs()
        diagnostics.append(f"| {label} | {err[subset.actual.eq(0)].mean():.6f} | {err[subset.actual.gt(0)].mean():.6f} |")
    for r in training.itertuples():
        parameters = ", ".join(f"{key}={value}" for key, value in json.loads(r.selected_hyperparameters).items())
        t = summary.loc[(r.model, "raw")]
        configurations.append(f"| {r.model} | {parameters} | {t.final_fit_seconds:.3f} | {t.test_prediction_seconds:.4f} |")
    fit = training.iloc[0]
    seasonal = summary.loc[("Seasonal Naive", "raw")]
    actual = test.loc[test.model.eq("Seasonal Naive"), "actual"]
    winners = {metric: STYLE[summary[metric].idxmin()][0] for metric in ("test_MAE", "test_RMSE")}
    score_header = "| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE | Test R² | Negative test predictions |\n|---|---:|---:|---:|---:|---:|---:|"
    run = "python -B experiments/03_tree_models_1h/run_comparison.py --verify"
    report = f"""# Caltech One-Hour Unified Model Comparison

**{winners['test_MAE']}** had the lowest test MAE and **{winners['test_RMSE']}** had the lowest test RMSE on the official boundary-aware test set. The six variants share all {fit.n_test:,} target timestamps from {seasonal.test_start} to {seasonal.test_end} in the source clock.

## Selection and refit

Eight prespecified candidates per learned model were fitted on {fit.n_train:,} training rows and ranked solely by raw MAE on {fit.n_validation:,} chronological validation rows. The earliest configured candidate resolves exact ties. Ridge includes alpha 1,000 in its search. The existing RF and GB candidate lists were reused.

After selection, every learned model was reinitialized and fitted on the same **{fit.n_final_refit:,}** ordered target timestamps, from {fit.final_training_start} through {fit.final_training_end}. Ridge's StandardScaler was fitted inside the pipeline on the appropriate fit rows. Trees used unscaled inputs. Random Forest fitted in parallel and summed tree predictions in a fixed order for exact replay. The last refit label became available at {fit.last_refit_label_available_at}, the first test issue. All choices and refits finished before test prediction. Seasonal Naive uses the count 24 hours earlier and has no fit.

The shared timestamp SHA-256 is `{fit.final_training_timestamps_sha256}`. The [training verification table](../results/tables/training_sample_verification_1h.csv) records counts, endpoints, parameters and the hash for each model.

| Model | Selected parameters | Final fit seconds | Test prediction seconds |
|---|---|---:|---:|
{chr(10).join(configurations)}

## Official results

{score_header}
{chr(10).join(rows)}

Validation predictions come from the selected training-only candidate. Test predictions come from its fresh refit. MAE and RMSE are measured in recorded starts per hour. R² is supplementary and uses 1 − SSE/SST, with NA when the target variance is zero. Raw and clipped Ridge and Gradient Boosting predictions are preserved separately. Random Forest had no negative validation or test predictions, so clipping it leaves every prediction unchanged.

## Error by observed target

Zero counts made up {actual.eq(0).mean():.1%} of test targets. These groups use observed test outcomes for retrospective diagnosis. The held-out errors estimate performance on this Caltech window, with operating coverage still unknown.

| Model | MAE on zero-count hours | MAE on positive-count hours |
|---|---:|---:|
{chr(10).join(diagnostics)}

## Reproduce

Runtime versions were Python {environment['python']}, scikit-learn {environment['sklearn']}, pandas {environment['pandas']}, NumPy {environment['numpy']} and Matplotlib {environment['matplotlib']}. Use the existing `experiments/03_tree_models_1h/requirements.txt` and run from Stage 3:

```powershell
{run}
```

The official runner checks the existing data SHA-256 and split ledger. Verification retrains every candidate and refit, compares the six generated tables apart from timings, and checks the hashes of the five regenerated charts. Results are published only for Caltech and the one-hour horizon.
"""
    site_rows = [
        f"| {site} | {site_stats['hours_in_series']:,} | {site_stats['first_hour']} | {site_stats['last_hour']} | "
        f"{site_stats['zero_hours']:,} | {site_stats['max_starts_in_one_hour']} |"
        for site, site_stats in provenance["sites"].items()
    ]
    baseline = f"""# Stage 3 数据口径与统一一小时基准

## 数据口径

目标为 **Hourly Recorded Session-Start Counts**。按归档文件名的会话开始时间，统计每个站点在左闭右开小时区间内的记录数。每个归档 CSV 包含一个源会话的多条遥测。运营覆盖和物理车辆到达需要独立证据。

`data/processed/hourly_counts_v1.csv` 保持原始站点、时间戳、计数和行序。字段为 `site`、`timestamp`、`session_start_count`、`clock_basis`、`timezone_status`、`coverage_status`、`is_first_hour` 和 `is_last_hour`。时间为 `source_clock`，时区和运营覆盖均为 `unknown`。所有零值及峰值保留于原视图。首尾小时可能只覆盖部分时段，经验证的边界策略仅限制实验样本。

| 站点 | 小时 | 起点 | 末小时 | 零小时 | 峰值 |
|---|---:|---|---|---:|---:|
{chr(10).join(site_rows)}

`data/processed/data_provenance.json` 记录 v1 创建时的输入和配置指纹。当前实验使用统一协议的配置，小时视图及其原始溯源文件保持原样。

## 正式一小时基准

Caltech 使用与完整比较相同的 {fit.n_test:,} 个测试目标，范围为 {seasonal.test_start} 至 {seasonal.test_end}。Ridge 先在 {fit.n_train:,} 个训练样本上拟合每个候选，在 {fit.n_validation:,} 个验证样本上按原始 MAE 选优，再将 StandardScaler 与 Ridge 从头拟合于同一组 {fit.n_final_refit:,} 个符合时序的最终样本。选中 alpha 为 **{json.loads(summary.at[('Ridge Regression', 'raw'), 'parameters'])['alpha']:g}**。Seasonal Naive 取前 24 小时对应计数。截断版保留为独立预测版本。

{score_header}
{chr(10).join(baseline_rows)}

正式基准文件为 [baseline_1h_unified.csv](../results/metrics/baseline_1h_unified.csv) 和 [baseline_predictions.csv](../results/tables/baseline_predictions.csv)，均由[完整统一实验](03_tree_models_1h.md)的结果筛出。旧 Stage 2 指标仅属于历史审查。基准入口调用统一运行器：

```powershell
python -B experiments/01_baseline_and_ridge/run_baseline.py --verify
```
"""
    return report, baseline
