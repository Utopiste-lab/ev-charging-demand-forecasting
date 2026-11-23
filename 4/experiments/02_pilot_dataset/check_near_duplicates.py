"""Cross-check candidate overlap using mutual-nearest ±1s telemetry timestamps."""
import argparse
import csv
from datetime import datetime,timezone
import gzip
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def nearest(origin,destination):
    right=np.clip(np.searchsorted(destination,origin),0,len(destination)-1)
    left=np.maximum(0,right-1)
    return np.where(abs(origin-destination[left])<=abs(origin-destination[right]),left,right)


def score(root):
    at=root/"results/tables/duplicate_candidates.csv"
    cand=pd.read_csv(at)
    needed=set(cand.record_a)|set(cand.record_b)
    series={k:{} for k in needed}
    epoch=datetime(1970,1,1,tzinfo=timezone.utc)
    with gzip.open(root/"data/processed/pilot_v1/metadata/raw_current_audit.csv.gz","rt",encoding="utf8",newline="") as handle:
        for row in csv.DictReader(handle):
            rid=row["record_id"]
            if rid not in series or row["timestamp_status"]!="parsed" or row["current_status"]!="finite" or row["same_time_conflict"]!="False":
                continue
            ts=datetime.fromisoformat(row["timestamp_original"])-epoch
            t=(ts.days*86400+ts.seconds)*1000000+ts.microseconds
            series[rid][t]=float(row["current_original"])
    arrays={}
    for rid,seq in series.items():
        order=sorted(seq)
        arrays[rid]=(np.asarray(order,dtype=np.int64),np.asarray([seq[t] for t in order],dtype=float))
    output=[]
    for row in cand.itertuples(index=False):
        ta,ya=arrays[row.record_a]
        tb,yb=arrays[row.record_b]
        rate,mae,spread,count=0.,np.nan,0.,0
        if len(ta) and len(tb) and row.overlap_s>0:
            lower=max(ta[0],tb[0]); upper=min(ta[-1],tb[-1])
            a=(ta>=lower)&(ta<=upper); b=(tb>=lower)&(tb<=upper)
            ta,ya,tb,yb=ta[a],ya[a],tb[b],yb[b]
            if len(ta) and len(tb):
                j=nearest(ta,tb); i=nearest(tb,ta)
                good=(i[j]==np.arange(len(ta)))&(abs(ta-tb[j])<=1000000)
                count=int(good.sum()); rate=count/max(len(ta),len(tb))
                if count:
                    mae=abs(ya[good]-yb[j[good]]).mean()
                    spread=min(np.ptp(ya[good]),np.ptp(yb[j[good]]))
        strong=(row.overlap_fraction_of_shorter>=.8 and count>=20 and rate>=.8 and mae<=.1
                and (row.same_token or spread>.5))
        output.append(dict(record_a=row.record_a,record_b=row.record_b,original_evidence=row.evidence,tolerance_seconds=1,
                           mutual_matches=count,matched_timestamp_fraction=rate,matched_current_MAE_a=mae,shared_range_min_a=spread,
                           strong_tolerance_candidate=bool(strong)))
    result=pd.DataFrame(output)
    added=result.strong_tolerance_candidate & ~result.original_evidence.isin(["confirmed_duplicate","strong_candidate"])
    assert not added.any(),"New strong evidence requires re-assessing the persisted chronological split"
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    names=parser.parse_args()
    result=score(ROOT)
    name=ROOT/"results/tables/phase42/duplicate_timestamp_tolerance.csv"
    content=result.to_csv(index=False,lineterminator="\n").encode()
    if names.verify:
        assert name.read_bytes()==content
    else:
        with name.open("xb") as stream:
            stream.write(content)
    print(f"Checked {len(result)} pairs with ±1 second mutual nearest alignment; no new strong relationship.")

