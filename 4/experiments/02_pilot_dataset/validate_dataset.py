"""Independent verification from serialized pilot artifacts and raw audit tokens."""
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd

D=ROOT/"data/processed/pilot_v1"
M=ROOT/"data/manifests"
SPLITS={"train","validation","test"}


def independently_bin(pairs,start,end,g,future,fill,cap):
    n=(end-start)//g
    bins=[[] for _ in range(n)]
    for t,v in pairs:
        allowed=(start<t<=end) if future else (start<=t<end)
        if allowed:
            k=int(np.ceil((t-start)/g)-1 if future else np.floor((t-start)/g))
            bins[k].append((t,v))
    observed=np.array([bool(p) for p in bins],dtype=bool)
    mask=np.zeros(n,dtype=bool)
    values=np.array([sum(x[1] for x in part)/len(part) if part else 0. for part in bins])
    for j in range(1,n-1):
        if fill and not observed[j] and observed[j-1] and observed[j+1]:
            if bins[j+1][0][0]-bins[j-1][-1][0]<=cap:
                mask[j]=True
                values[j]=(values[j-1]+values[j+1])/2
    return values,observed,mask


def verify(root=ROOT):
    d=root/"data/processed/pilot_v1"
    m=root/"data/manifests"
    n=pd.read_csv(m/"record_manifest.csv",dtype={"record_id":str})
    s=pd.read_csv(m/"dataset_split.csv",dtype={"record_id":str})
    p=pd.read_csv(d/"metadata/prefix_by_record.csv",dtype={"record_id":str})
    f=pd.read_csv(d/"metadata/future_by_record.csv",dtype={"record_id":str})
    full=pd.read_csv(d/"metadata/resampling_by_record.csv",dtype={"record_id":str})
    cfg=json.loads((d/"metadata/dataset_metadata.json").read_text(encoding="utf8"))
    assert cfg["scaler_fitted"] is False and len(n)==608 and n.record_id.is_unique
    assert s.record_id.is_unique and set(s.record_id)==set(n.record_id)
    assert n.groupby(["site","cohort"]).size().tolist()==[300,4,300,4]
    sel=n.loc[n.partition.isin(SPLITS)]
    assert sel.cohort.eq("representative").all()
    assert sel.groupby("duplicate_group_id").partition.nunique().le(1).all()
    for a,b,c in zip(("train","validation"),("validation","test"),cfg["cutoffs_us"],strict=True):
        before,after=[sel.loc[sel.partition.eq(x)] for x in (a,b)]
        assert len(before) and len(after) and before.last_timestamp_us.max()<c<=after.first_timestamp_us.min()
    n=n.set_index("record_id")
    subset=set()
    for (site,category),q in p.loc[p.cohort.eq("representative") & p.stored].groupby(["site","quality_category"]):
        q=q.sort_values(["interpolated_bins","record_id"],ascending=[False,True])
        subset.update(q.record_id.drop_duplicates().head(4))
    raw={k:{} for k in subset}
    timestamps={}
    seen_rows=0
    per_id={}
    with gzip.open(d/"metadata/raw_current_audit.csv.gz","rt",encoding="utf8",newline="") as stream:
        for row in csv.DictReader(stream):
            rid=row["record_id"]
            per_id[rid]=per_id.get(rid,0)+1
            seen_rows+=1
            if rid not in raw or row["timestamp_status"]!="parsed":
                continue
            ts=datetime.fromisoformat(row["timestamp_original"])
            timestamps[rid]=min(timestamps.get(rid,ts),ts)
            if row["current_status"]=="finite" and row["same_time_conflict"]=="False":
                raw[rid][ts]=float(row["current_original"])
    assert per_id==n.loc[n.processing_status.ne("failed"),"original_rows"].astype(int).to_dict()
    pairs={rid:sorted(((t-timestamps[rid]).total_seconds(),a) for t,a in pts.items()) for rid,pts in raw.items()}
    arrays_checked=0
    regenerated=0
    for stage in sorted(d.rglob("*.npz")):
        site=stage.relative_to(d).parts[0].capitalize() if stage.relative_to(d).parts[0]=="caltech" else "JPL"
        with np.load(stage,allow_pickle=False) as data:
            ids=data["record_id"]
            val,obs,imp=(data[k] for k in ("current_values","observed_mask","interpolated_mask"))
            assert len(set(ids))==len(ids) and all(rid in sel.record_id.values for rid in ids)
            assert val.shape==obs.shape==imp.shape and obs.dtype==imp.dtype==np.bool_ and val.dtype==np.float32
            assert not (obs&imp).any() and np.isfinite(val).all() and (val[~obs&~imp]==0).all()
            if stage.name=="observed_record_sequences.npz":
                offsets=data["offsets"]
                expected=full.loc[full.site.eq(site)&full.stored_full].set_index("record_id")
                assert set(ids)==set(expected.index) and offsets[-1]==len(val) and len(offsets)==len(ids)+1
                for k,rid in enumerate(ids):
                    sl=slice(offsets[k],offsets[k+1])
                    row=expected.loc[rid]
                    assert (len(val[sl]),obs[sl].sum(),imp[sl].sum())==(row.bins,row.direct_bins,row.interpolated_bins)
                    if rid in pairs:
                        step=cfg["primary_grid_seconds"]; end=int(row.bins)*step
                        z=independently_bin(pairs[rid],0,end,step,False,True,cfg["config"]["max_interpolation_span_seconds"])
                        np.testing.assert_allclose(val[sl],z[0],rtol=1e-6,atol=1e-6)
                        np.testing.assert_array_equal(obs[sl],z[1]); np.testing.assert_array_equal(imp[sl],z[2])
                        regenerated+=1
            else:
                step=int(stage.parent.name[:-1]); is_future=stage.name.startswith("future")
                if is_future:
                    duration,_,pref=stage.stem.split("_")[1:]
                    prefix_min=int(pref.removesuffix("min")); duration=int(duration.removesuffix("min"))
                    expected=f.loc[f.site.eq(site)&f.grid_s.eq(step)&f.prefix_min.eq(prefix_min)&f.future_min.eq(duration)&f.stored].set_index("record_id")
                    assert not imp.any()
                    np.testing.assert_array_equal(data["strict_eligible"],expected.loc[ids,"usable_strict"])
                    assert not expected.loc[~expected.span_reaches_window,"usable_strict"].any()
                else:
                    duration=int(stage.stem.split("_")[1].removesuffix("min"))
                    prefix_min=0
                    expected=p.loc[p.site.eq(site)&p.grid_s.eq(step)&p.prefix_min.eq(duration)&p.stored].set_index("record_id")
                    np.testing.assert_array_equal(data["strict_eligible"],expected.loc[ids,"strict"])
                assert val.shape==(len(ids),duration*60//step) and set(ids)==set(expected.index)
                for k,rid in enumerate(ids):
                    row=expected.loc[rid]
                    assert (obs[k].sum(),imp[k].sum())==(row.observed_bins,row.interpolated_bins)
                    if rid in pairs:
                        cutoff=prefix_min*60
                        z=independently_bin(pairs[rid],cutoff,cutoff+duration*60,step,is_future,not is_future,cfg["config"]["max_interpolation_span_seconds"])
                        np.testing.assert_allclose(val[k],z[0],rtol=1e-6,atol=1e-6)
                        np.testing.assert_array_equal(obs[k],z[1]); np.testing.assert_array_equal(imp[k],z[2])
                        regenerated+=1
        arrays_checked+=1
    print(f"Verified all {arrays_checked} tensor files, chronology/group keys and masks. Independently regenerated "
          f"{regenerated} windows from raw timestamp/current tokens across {len(subset)} records. Audited {seen_rows} raw rows.")
    return arrays_checked


if __name__=="__main__":
    verify()

