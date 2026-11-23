"""Static scientific plots made only from saved forecasts and scores."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np

from stage3_context import FINAL_FIGURES, PROB_FIGURES

SITES = ["Caltech full", "JPL matched"]
H = [1, 3, 6, 24]
BLUE, TEAL, GOLD, GRAY = "#2461A5", "#168579", "#BB6B30", "#434A56"


def finish(fig, axs, title, out, folder, name):
    fig.suptitle(title, weight="semibold")
    for ax in axs:
        ax.grid(axis="y", alpha=.21, linewidth=.7)
        ax.set_axisbelow(True)
    relative = folder / (name + ".png")
    (out / folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out / relative, dpi=182, facecolor="white")
    plt.close(fig)
    return relative


def pair():
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.6), layout="constrained")
    return fig, axs


def lines(ax, data, y, labels, names):
    p = data.pivot(index="horizon_hours", columns=labels, values=y).loc[H]
    for i, name in enumerate(names):
        ax.plot(range(4), p[name], label=name, marker=["o","s","D"][i%3], color=[GRAY,BLUE,TEAL][i%3], linewidth=1.7)
    ax.set_xticks(range(4), [f"{h}h" for h in H])
    ax.set_xlabel("Forecast horizon")


def probabilistic(out, forecasts, pscore, intervals, quantiles):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"semibold", "legend.frameon":False})
    products = []
    test = forecasts.loc[forecasts.window.eq("official") & forecasts.partition.eq("test")]
    for chart in ("quantiles", "interval"):
        fig, axs = plt.subplots(2, 1, figsize=(11.5, 6.1), layout="constrained")
        for ax, case in zip(axs, SITES, strict=True):
            rows = test.loc[test.case.eq(case) & test.horizon_hours.eq(1)].sort_values("target_hour").iloc[:168]
            t = rows.target_hour
            if chart == "quantiles":
                for name, color in [("P10", TEAL), ("P50", BLUE), ("P90", GOLD)]:
                    ax.plot(t, rows[name], color=color, lw=1.1, label=name)
            else:
                ax.fill_between(t, rows.CQR_lower, rows.CQR_upper, color=TEAL, alpha=.25, label="CQR 80%")
                ax.plot(t, rows.P90, color=BLUE, lw=.9, ls="--", label="Raw upper")
                ax.plot(t, rows.P10, color=BLUE, lw=.9, ls="--")
                ax.plot(t, rows.P50, color=BLUE, lw=1.2, label="P50")
            ax.scatter(t, rows.actual, s=8, color=GRAY, alpha=.62, label="Observed")
            ax.xaxis.set_major_formatter(dates.DateFormatter("%b %d"))
            ax.set(title=case, ylabel="Recorded starts/hour")
        axs[0].legend(ncol=4, loc="upper right")
        name = "01_quantiles_vs_actual" if chart=="quantiles" else "02_prediction_intervals"
        products.append(finish(fig, axs, "Official test · first chronological week · 1h ahead",out,PROB_FIGURES,name))
    core = intervals.loc[intervals.case.isin(SITES) & intervals.window.eq("official")]
    overall = core.loc[core.partition.eq("test") & core.regime.eq("all")]
    fig, ax = plt.subplots(figsize=(9.5,4.7), layout="constrained")
    a = overall.groupby(["case","method"]).coverage.mean().unstack()
    for i, (m, color) in enumerate([("raw", BLUE), ("CQR", TEAL)]):
        ax.bar(np.arange(2)+(i-.5)*.32, a.loc[SITES,m]*100,.32,color=color,label=m)
    ax.axhline(80, color=GRAY, ls="--", label="Nominal 80%")
    ax.set_xticks(range(2), SITES)
    ax.set(ylabel="Mean coverage across horizons (%)",ylim=(0,104))
    ax.legend(ncol=3)
    products.append(finish(fig,[ax],"Interval coverage by site · equal weight across four horizons",out,PROB_FIGURES,"03_coverage_by_site"))
    for field, heading, name in [("coverage","Coverage by forecasting horizon","04_coverage_by_horizon"),
                                 ("width","Mean interval width by site and horizon","05_interval_width")]:
        fig, axs = pair()
        for ax, case in zip(axs,SITES):
            d = overall.loc[overall.case.eq(case)].copy()
            if field=="coverage":
                d[field] *= 100
                ax.axhline(80, color=GOLD, ls="--", label="Nominal 80%")
            lines(ax,d,field,"method",["raw","CQR"])
            ax.set(title=case,ylabel="Coverage (%)" if field=="coverage" else "Width (starts/hour)")
        axs[-1].legend()
        products.append(finish(fig,axs,heading,out,PROB_FIGURES,name))
    fig, axs = pair()
    for ax, case in zip(axs,SITES):
        d=quantiles.loc[quantiles.window.eq("official") & quantiles.case.eq(case) & quantiles.partition.eq("test")]
        lines(ax,d,"pinball","quantile",["P10","P50","P90"])
        ax.set(title=case,ylabel="Mean pinball loss")
    axs[-1].legend()
    products.append(finish(fig,axs,"Quantile scores on the official test window",out,PROB_FIGURES,"06_pinball_loss"))
    fig, axs = plt.subplots(2,2,figsize=(11.4,7),layout="constrained")
    regimes = core.loc[core.partition.eq("test") & core.regime.ne("all") & core.method.eq("CQR")].copy()
    regimes["coverage"] *=100
    for j, case in enumerate(SITES):
        for i, field in enumerate(["coverage","width"]):
            ax=axs[i,j]
            lines(ax,regimes.loc[regimes.case.eq(case)],field,"regime",["zero","positive"])
            if field=="coverage":
                ax.axhline(80,color=GOLD,ls="--", label="Nominal 80%")
            ax.set(title=case if i==0 else "",ylabel="Coverage (%)" if i==0 else "Width (starts/hour)")
    axs[0,1].legend()
    products.append(finish(fig,axs.ravel(),"CQR performance by observed demand · post-hoc groups",out,PROB_FIGURES,"07_demand_regime_intervals"))
    fig, axs = pair()
    for ax, case in zip(axs,SITES):
        d=core.loc[core.case.eq(case) & core.regime.eq("all") & core.method.eq("CQR")].copy()
        d["coverage"]*=100
        ax.axhline(80,color=GOLD,ls="--", label="Nominal 80%")
        lines(ax,d,"coverage","partition",["calibration","test"])
        ax.set(title=case,ylabel="Empirical coverage (%)")
    axs[-1].legend()
    products.append(finish(fig,axs,"Calibration fit versus coverage on held-out test hours",out,PROB_FIGURES,"08_calibration_and_test_coverage"))
    fig, axs=plt.subplots(2,2,figsize=(11.4,7.1),layout="constrained")
    rolling=intervals.loc[intervals.window.ne("official") & intervals.regime.eq("all")]
    for i, case in enumerate(SITES):
        for j, part in enumerate(["calibration","test"]):
            ax=axs[i,j]
            d=rolling.loc[rolling.case.eq(case) & rolling.partition.eq(part)]
            for m,color in [("raw",BLUE),("CQR",TEAL)]:
                v=d.loc[d.method.eq(m)].sort_values(["window","horizon_hours"])
                ax.plot(range(12),100*v.coverage,marker="o",ms=3,lw=1.25,color=color,label=m)
            ax.axhline(80,color=GOLD,ls="--")
            ax.set_xticks([1.5,5.5,9.5],["R1","R2","R3"])
            ax.set(title=f"{case} · {part}",ylabel="Coverage (%)")
    axs[0,0].legend()
    products.append(finish(fig,axs.ravel(),"Rolling windows · each group shows 1h, 3h, 6h and 24h",out,PROB_FIGURES,"09_rolling_calibration_coverage"))
    cross_cases = [("Caltech->JPL", "JPL matched"), ("JPL->Caltech", "Caltech matched")]
    if test.case.str.contains("->", regex=False).any():
        fig, axs=pair()
        for ax,(transfer, match) in zip(axs,cross_cases):
            d=intervals.loc[intervals.window.eq("official") & intervals.regime.eq("all") & intervals.partition.eq("test")
                            & intervals.method.eq("CQR") & intervals.case.isin([transfer,match])].copy()
            d["coverage"]*=100
            d["case"]=d.case.replace({transfer:"Source calibration only",match:"Matched same-site"})
            lines(ax,d,"coverage","case",["Matched same-site","Source calibration only"])
            ax.axhline(80,color=GOLD,ls="--", label="Nominal 80%")
            ax.set(title=transfer.replace("->"," → "),ylabel="Coverage (%)")
        axs[-1].legend()
        products.append(finish(fig,axs,"Exploratory transfer · source-calibrated versus same-site intervals",out,PROB_FIGURES,"10_cross_site_coverage"))
    return products


def baseline(out, compar):
    fig, axs=pair()
    use = compar.loc[compar.case.isin(SITES) & compar.horizon_hours.eq(1)]
    for ax, case in zip(axs,SITES):
        d=use.loc[use.case.eq(case)].set_index("model")
        fields=["MAE","zero_MAE","positive_MAE"]
        for j,(name,color) in enumerate([("Constant Zero",GRAY),("Seasonal Naive",TEAL)]):
            ax.bar(np.arange(3)+(j-.5)*.32,d.loc[name,fields],.32,color=color,label=name)
        ax.set_xticks(range(3),["All hours","Zero actual","Positive actual"])
        ax.set(title=case,ylabel="MAE (starts/hour)")
    axs[1].legend()
    return [finish(fig,axs,"Constant zero and seasonal baselines · identical 1h test targets",out,FINAL_FIGURES,"constant_zero_demand_regimes")]
