"""Describe archived demand and feature shift, without using statistics to fit models."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_NAMES = (
    "demand_distribution_comparison.png", "zero_demand_comparison.png",
    "hourly_pattern_comparison.png", "weekday_pattern_comparison.png",
)
SITES = {"Caltech": "#2E6C91", "JPL": "#B65B2D"}
BIN_LABELS = ["0", "1", "2", "3", "4", "5–9", "10–19", "20+"]
LABELS = {"common_archive": "Common archive", "test_targets": "Matched test hours"}


def statistics(values):
    q = values.quantile([.10, .25, .50, .75, .90, .95, .99])
    return {
        "n": len(values), "mean": values.mean(), "median": q.loc[.50], "std_ddof1": values.std(ddof=1),
        "max": values.max(), "zero_n": int(values.eq(0).sum()), "positive_n": int(values.gt(0).sum()),
        "zero_fraction": values.eq(0).mean(), "positive_fraction": values.gt(0).mean(),
        **{f"p{round(100*k):02d}": v for k, v in q.items()},
    }


def describe_demand(hourly, common, test_index, refit_index):
    out = []
    for scope, frame in [
        ("full_archive", hourly), ("common_archive", common),
        ("common_refit_1h", common.loc[common.timestamp.isin(refit_index)]),
        ("test_targets", common.loc[common.timestamp.isin(test_index)]),
    ]:
        for site, hours in frame.groupby("site", sort=True):
            values = hours.session_start_count
            base = {"scope": scope, "site": site, "variable": "session_start_count"}
            out.append({**base, "group_type": "overall", "group": "all",
                        **statistics(values)})
            if scope in LABELS:
                for grouping, key in [
                    ("hour", hours.timestamp.dt.hour),
                    ("weekday", hours.timestamp.dt.dayofweek),
                    ("day_type", hours.timestamp.dt.dayofweek.ge(5).map({True: "weekend", False: "weekday"})),
                ]:
                    for group, data in values.groupby(key, sort=True):
                        out.append({**base, "group_type": grouping, "group": str(group), **statistics(data)})
                bins = pd.cut(values, [-.5, .5, 1.5, 2.5, 3.5, 4.5, 9.5, 19.5, np.inf], labels=BIN_LABELS)
                frequencies = bins.value_counts(sort=False)
                assert frequencies.sum() == len(values)
                for group, n in frequencies.items():
                    out.append({**base, "group_type": "count_bin", "group": group,
                                "n": n, "fraction_of_scope": n / len(values)})
    return out


def describe_features(h, direction, source, target, history):
    rows = []
    for scope, payload, X in [
        ("historical_source_refit", source, source.final_refit.X),
        ("historical_target_test", target, target.X),
    ]:
        for feature in history:
            rows.append({"scope": scope, "site": payload.site, "direction": direction, "horizon_hours": h,
                         "variable": feature, "group_type": "overall", "group": "all", **statistics(X[feature])})
    return rows


def plot_distributions(table, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.axisbelow": True, "grid.alpha": .22,
                         "figure.titlesize": 15})

    def draw(figure, filename):
        figure.savefig(directory / filename, dpi=170, facecolor="white")
        plt.close(figure)

    def selected(scope, kind, site):
        return table.loc[table.scope.eq(scope) & table.group_type.eq(kind) & table.site.eq(site)]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True, layout="constrained")
    fig.suptitle("Recorded hourly session starts · Count distribution")
    x = np.arange(len(BIN_LABELS))
    for ax, (scope, title) in zip(axes, LABELS.items()):
        for i, (site, color) in enumerate(SITES.items()):
            p = selected(scope, "count_bin", site).set_index("group").loc[BIN_LABELS].fraction_of_scope
            ax.bar(x + (i - .5) * .35, p * 100, width=.35, color=color, label=site)
        ax.set(title=title, xlabel="Recorded starts per hour", xticks=x, xticklabels=BIN_LABELS)
        ax.grid(axis="y")
    axes[0].set_ylabel("Hours (%)")
    axes[1].legend(frameon=False)
    draw(fig, FIGURE_NAMES[0])

    fig, ax = plt.subplots(figsize=(9.5, 4.6), layout="constrained")
    fig.suptitle("Share of hours with zero recorded session starts")
    scopes = {"full_archive": "Full archive", "common_archive": "Common archive",
              "common_refit_1h": "Common 1h refit window", "test_targets": "Matched test hours"}
    x = np.arange(len(scopes))
    for i, (site, color) in enumerate(SITES.items()):
        v = [selected(scope, "overall", site).zero_fraction.iat[0] * 100 for scope in scopes]
        bars = ax.bar(x + (i - .5) * .34, v, width=.34, color=color, label=site)
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(xticks=x, xticklabels=list(scopes.values()), ylabel="Zero-count hours (%)", ylim=(0, 110))
    ax.grid(axis="y")
    ax.legend(frameon=False, ncols=2, loc="upper right")
    draw(fig, FIGURE_NAMES[1])

    for kind, tick_labels, xlabel, filename in [
        ("hour", [str(i) for i in range(24)], "Hour of day (source clock)", FIGURE_NAMES[2]),
        ("weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
         "Day of week (source clock)", FIGURE_NAMES[3]),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True, layout="constrained")
        fig.suptitle(f"Mean recorded session starts by {kind}")
        keys = np.arange(len(tick_labels))
        step = 3 if kind == "hour" else 1
        for ax, (scope, title) in zip(axes, LABELS.items()):
            for site, color in SITES.items():
                values = selected(scope, kind, site).set_index("group").loc[[str(i) for i in keys], "mean"]
                ax.plot(keys, values, marker="o", ms=3, lw=1.45, color=color, label=site)
            ax.set(title=title, xlabel=xlabel, xticks=keys[::step], xticklabels=tick_labels[::step])
            ax.grid(axis="y")
        axes[0].set_ylabel("Mean recorded starts per hour")
        axes[0].legend(frameon=False)
        draw(fig, filename)
