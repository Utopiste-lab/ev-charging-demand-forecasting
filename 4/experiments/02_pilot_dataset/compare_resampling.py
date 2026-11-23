"""Pilot cohort summaries and publication-ready empirical figures."""
import numpy as np
import pandas as pd
from src.preprocessing.resampling import grid,prefix,future_target,SECOND


def summarize(r,p,y,f,ex):
    raw=[]
    for (site,cohort),g in r.groupby(["site","cohort"]):
        row=dict(site=site,cohort=cohort,selected=len(g),parsed=g.processing_status.eq("parsed").sum(),
                 available_months=g.filename_month.nunique(),original_rows=g.original_rows.sum(),
                 missing_current_rows=g.missing_current_rows.sum(),invalid_current_rows=g.invalid_current_rows.sum(),
                 malformed_rows=g.malformed_rows.sum(),invalid_timestamp_rows=g.invalid_timestamp_rows.sum(),
                 duplicates_within_rows=g.duplicate_timestamp_rows.sum(),conflict_files=g.conflicting_current_times.gt(0).sum(),
                 raw_current_availability=g.finite_current_timestamp_rows.sum()/g.parsed_rows.sum(),
                 negative_current_fraction=g.negative_current_rows.sum()/g.finite_current_timestamp_rows.sum(),
                 zero_current_fraction=g.zero_current_rows.sum()/g.finite_current_timestamp_rows.sum(),
                 gap_median_s=g.median_gap_s.median(),gap_max_s=g.max_gap_s.max(),
                 long_gap_records=g.long_observation_gaps.gt(0).sum(),median_duration_min=g.observed_duration_s.median()/60,
                 min_a=g.min_current_a.min(),max_a=g.max_current_a.max(),original_rows_median=g.original_rows.median())
        raw.append(row)
    summary=[]
    for (site,c,g),v in f.groupby(["site","cohort","grid_s"]):
        complete=v.loc[v.full_status.ne("full_grid_cap")]
        n=complete.bins.sum()
        for policy in ("no_interpolation","limited_interpolation"):
            limit=policy=="limited_interpolation"
            summary.append(dict(site=site,cohort=c,grid_s=g,policy=policy,n_records=len(v),
                                full_grid_cap_records=v.full_status.eq("full_grid_cap").sum(),
                                full_eligible_records=v.full_status.eq("eligible").sum(),total_bins=n,
                                direct_fraction=complete.direct_bins.sum()/n,interpolated_fraction=complete.interpolated_bins.sum()/n if limit else 0,
                                missing_fraction=complete["missing_limited_bins" if limit else "missing_unfilled_bins"].sum()/n,
                                maximum_consecutive_missing_bins=complete["max_missing_limited_bins" if limit else "max_missing_unfilled_bins"].max(),
                                pointwise_mean_MAE_record_median_a=complete.mean_pointwise_MAE_a.median(),
                                pointwise_median_MAE_record_median_a=complete.median_pointwise_MAE_a.median(),
                                fluctuation_range_retention_record_median=complete.fluctuation_range_retained.median()))
    res=pd.DataFrame(summary)
    p_report=p.assign(strict_stored=p.strict&p.stored).groupby(["site","cohort","grid_s","prefix_min","quality_category"],as_index=False).agg(
        records=("record_id","size"),inclusive_pass=("inclusive","sum"),strict_pass=("strict","sum"),
        stored=("stored","sum"),stored_strict=("strict_stored","sum"),mean_direct_fraction=("observed_fraction","mean"))
    horizon=y.groupby(["site","cohort","grid_s","prefix_min","future_min","quality_category"],as_index=False).agg(
        records=("record_id","size"),prefix_linked_stored=("stored","sum"),usable_inclusive=("usable_inclusive","sum"),
        usable_strict=("usable_strict","sum"),fully_observed=("fully_observed","sum"),mean_direct_fraction=("observed_fraction","mean"))
    compatibility=pd.DataFrame(raw).merge(
        res.loc[res.policy.eq("limited_interpolation"),["site","cohort","grid_s","direct_fraction","interpolated_fraction","missing_fraction",
                                                       "full_eligible_records"]],on=["site","cohort"],validate="one_to_many")
    quality_counts=p.groupby(["site","cohort","grid_s"],as_index=False).agg(
        inclusive_prefix_cells=("inclusive","sum"),strict_prefix_cells=("strict","sum"),prefix_cells=("record_id","size"))
    future_counts=y.groupby(["site","cohort","grid_s"],as_index=False).agg(
        downstream_strict_cells=("usable_strict","sum"),downstream_inclusive_cells=("usable_inclusive","sum"))
    compatibility=compatibility.merge(quality_counts,on=["site","cohort","grid_s"]).merge(future_counts,on=["site","cohort","grid_s"])
    return {"pilot_sample_summary.csv":pd.DataFrame(raw),"resampling_comparison.csv":res,
            "gap_quality_summary.csv":r.groupby(["site","cohort"],as_index=False).agg(
                records=("record_id","size"),records_with_long_gap=("long_observation_gaps",lambda s:s.gt(0).sum()),
                gap_count=("long_observation_gaps","sum"),max_raw_gap_s=("max_gap_s","max"),median_record_gap_s=("median_gap_s","median")),
            "phase42/prefix_eligibility.csv":p_report,"future_current_target_coverage.csv":horizon,
            "phase42/site_compatibility.csv":compatibility,
            "split_summary.csv":r.groupby(["site","cohort","partition"],as_index=False).agg(
                records=("record_id","size"),group_n=("duplicate_group_id","nunique"),
                first_anchor_us=("first_timestamp_us","min"),last_record_end_us=("last_timestamp_us","max")),
            "dataset_exclusion_summary.csv":ex.groupby(["site","scope","reason"],as_index=False).size().rename(columns={"size":"records_or_cases"})}


def figures(out,records,parsed,p,y,f,primary,cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    from datetime import datetime,timezone
    root=out/"results/figures/phase42"
    root.mkdir(parents=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
    blue,sea,orange,gray="#205f8e","#378876","#ad633c","#586775"
    colors={"Caltech":blue,"JPL":orange}
    records=records.set_index("record_id")
    scope=p.loc[p.grid_s.eq(primary)&p.prefix_min.eq(10)&p.stored&p.strict]
    qualified=y.loc[y.grid_s.eq(primary)&y.prefix_min.eq(10)&y.future_min.eq(30)&y.usable_strict]
    examples={site:subset.sort_values(["observed_fraction","record_id"],ascending=[False,True]).record_id.iat[0]
              for site,subset in qualified.groupby("site",sort=True)}
    for site,part in scope.groupby("site",sort=True):
        examples.setdefault(site,part.record_id.iat[0])
    def emit(fig,name,heading):
        fig.suptitle(heading,fontsize=12,fontweight="semibold")
        fig.savefig(root/(name+".png"),dpi=165,facecolor="white")
        plt.close(fig)
    def short(rid):
        return records.at[rid,"source_file"].split("/")[-1]
    def scatter(ax,g,seconds,offset=0,labels=True,kind="mean"):
        xx=offset+(np.arange(len(g.values))+.5)*seconds
        ax.scatter(xx[g.observed]/60,g.values[g.observed],s=12,color=blue,label="Direct bin mean" if labels else None)
        if g.interpolated.any():
            ax.scatter(xx[g.interpolated]/60,g.values[g.interpolated],s=18,color=orange,label="Short-hole interpolation" if labels else None)
        for j in np.flatnonzero(~g.observed&~g.interpolated):
            ax.axvspan((offset+j*seconds)/60,(offset+(j+1)*seconds)/60,color="#dce2e8",alpha=.55)
        ax.grid(alpha=.15)
    fig,ax=plt.subplots(2,1,figsize=(11,6),layout="constrained")
    for pane,(site,rid) in zip(ax,examples.items(),strict=True):
        packet=parsed[rid]
        t=packet.elapsed_us; sel=t<600*SECOND
        pane.scatter(t[sel]/(60*SECOND),packet.values[sel],s=5,c=gray,label="Original current")
        g,_=prefix(packet,10,primary,cfg)
        scatter(pane,g,primary)
        pane.set(title=f"{site} | {short(rid)}",ylabel="Current (A)",xlabel="Minutes since first telemetry")
    ax[0].legend(frameon=False,ncol=3)
    emit(fig,"01_irregular_vs_grid","Original observations and primary fixed grid")
    fig,ax=plt.subplots(2,1,figsize=(11,6),layout="constrained")
    for pane,(site,rid) in zip(ax,examples.items(),strict=True):
        packet=parsed[rid]
        for k,(s,color) in enumerate(zip([5,10,30],[blue,sea,orange])):
            g,_=prefix(packet,5,s,cfg)
            x=(np.arange(len(g.values))+.5)*s
            pane.plot(x/60,np.where(g.observed,g.values,np.nan),marker=[".","o","s"][k],ms=3,label=f"{s}s · observed bin mean",color=color)
        pane.set(title=f"{site} | {short(rid)}",ylabel="Current (A)",xlabel="Minutes since first telemetry")
    ax[0].legend(ncol=3,frameon=False)
    emit(fig,"02_grid_comparison","5s, 10s and 30s grids on identical prefixes")
    fig,ax=plt.subplots(2,1,figsize=(11,5.6),layout="constrained")
    for pane,(site,rid0) in zip(ax,examples.items(),strict=True):
        options=p.loc[p.site.eq(site)&p.grid_s.eq(primary)&p.prefix_min.eq(10)&p.cohort.eq("representative")]
        options=options.assign(both=options.interpolated_bins.gt(0)&options.missing_bins.gt(0))
        rid=options.sort_values(["both","interpolated_bins","observed_fraction","record_id"],
                                ascending=[False,False,False,True]).record_id.iat[0]
        g,_=prefix(parsed[rid],10,primary,cfg)
        x=(np.arange(len(g.values))+.5)*primary/60
        pane.scatter(x[g.observed],np.full(g.observed.sum(),2),s=13,label="direct",color=blue)
        pane.scatter(x[g.interpolated],np.full(g.interpolated.sum(),1),s=18,label="interpolated",color=orange)
        missing=~g.observed&~g.interpolated
        pane.scatter(x[missing],np.zeros(missing.sum()),s=14,label="missing",color=gray)
        pane.set(yticks=[0,1,2],yticklabels=["missing","interpolated","direct"],ylim=(-.5,2.5),title=f"{site} | {short(rid)}",
                 xlabel="Minutes since first telemetry")
    ax[0].legend(frameon=False,ncol=3)
    emit(fig,"03_observation_and_interpolation_masks",f"Three distinct mask states · {primary}s")
    for duration,filename in [(5,"04_prefix_5min"),(10,"05_prefix_10min")]:
        fig,ax=plt.subplots(2,1,figsize=(11,6),layout="constrained")
        for pane,(site,rid) in zip(ax,examples.items(),strict=True):
            g,_=prefix(parsed[rid],duration,primary,cfg)
            scatter(pane,g,primary)
            pane.axvline(duration,color=gray,ls="--",label="Prefix cutoff")
            pane.set(title=f"{site} | {short(rid)}",ylabel="Current (A)",xlabel="Minutes since first telemetry")
        ax[0].legend(ncol=3,frameon=False)
        emit(fig,filename,f"{duration}-minute prefix · {primary}s")
    fig,ax=plt.subplots(2,1,figsize=(11,6),layout="constrained")
    for pane,(site,rid) in zip(ax,examples.items(),strict=True):
        g,_=prefix(parsed[rid],10,primary,cfg)
        scatter(pane,g,primary,labels=False)
        for h,color in [(30,sea),(10,orange)]:
            z,_=future_target(parsed[rid],10,h,primary,cfg)
            x=10+(np.arange(len(z.values))+.5)*primary/60
            pane.plot(x,np.where(z.observed,z.values,np.nan),".",ms=3,label=f"Observed future {h}min",color=color)
        pane.axvline(10,ls="--",lw=1,color=gray,label="Prefix cutoff")
        pane.set(title=f"{site} | {short(rid)}",ylabel="Current (A)",xlabel="Minutes since first telemetry")
    ax[0].legend(frameon=False,ncol=3)
    emit(fig,"06_future_current","Observed downstream targets and corresponding prefix")
    fig,ax=plt.subplots(figsize=(10.5,4.4),layout="constrained")
    for site,v in records.loc[records.cohort.eq("representative")].groupby("site"):
        a=np.concatenate([parsed[rid].values for rid in v.index if rid in parsed])
        x=np.arange(-1,131,1)
        h,_=np.histogram(a,bins=x)
        ax.stairs(np.maximum(h/len(a),1e-7),x,label=f"{site} (n={len(a):,} points)",color=colors[site])
    ax.set(yscale="log",xlabel="Charging current (A), 1 A bins",ylabel="Proportion per bin (log; floor 1e−7)")
    ax.legend(frameon=False)
    emit(fig,"07_site_current_distributions","Current distribution · all eligible timestamp/current points in representative draw")
    fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    for ax,(site,v) in zip(axs,p.loc[p.cohort.eq("representative")&p.prefix_min.eq(10)].groupby("site"),strict=True):
        a=v.groupby("grid_s").agg(strict=("strict","mean"),inclusive=("inclusive","mean"),observed=("observed_fraction","mean"))
        for i,(metric,color) in enumerate([("strict",blue),("inclusive",sea),("observed",orange)]):
            ax.bar(np.arange(3)+(i-1)*.22,100*a[metric],.22,label=metric,color=color)
        ax.set(xticks=range(3),xticklabels=["5s","10s","30s"],ylim=(0,108),ylabel="Percent",title=site,xlabel="Grid")
    axs[0].legend(frameon=False,ncol=3)
    emit(fig,"08_sequence_quality","10-minute quality and average direct-bin coverage · representative cohorts")
    fig,ax=plt.subplots(figsize=(11,4.2),layout="constrained")
    for i,(site,v) in enumerate(records.loc[records.cohort.eq("representative")].groupby("site")):
        for name,color in [("train",blue),("validation",sea),("test",orange),("excluded",gray)]:
            points=pd.to_datetime(v.loc[v.partition.eq(name),"first_timestamp_us"],unit="us",utc=True)
            ax.scatter(points,np.repeat(i,len(points)),s=8,color=color,label=name if i==0 else None,alpha=.65)
    ax.set(yticks=[0,1],yticklabels=sorted(colors),ylim=(-.5,1.5),xlabel="First observation, absolute calendar (+00:00 display)")
    ax.xaxis.set_major_formatter(md.DateFormatter("%Y-%m"))
    ax.legend(ncol=4,frameon=False)
    emit(fig,"09_split_timeline","Representative record assignment · common chronological split")

