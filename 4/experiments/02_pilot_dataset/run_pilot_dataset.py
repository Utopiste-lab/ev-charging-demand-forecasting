"""Generate or replay Phase 4.2, reading only the bounded pilot archives."""
import argparse
from collections import defaultdict
import csv
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import platform
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True

import numpy as np
import pandas as pd
import yaml
from src.preprocessing.sampling import select
from src.preprocessing.parser import load
from src.preprocessing.duplicate_detection import candidates,groups
from src.preprocessing.dataset_split import split
from src.preprocessing.resampling import prefix,future_target,fit_comparison
from src.preprocessing.serialization import pack,equal_npz
from compare_resampling import summarize,figures

PROCESSED=Path("data/processed/pilot_v1")
META=PROCESSED/"metadata"
TABLE=Path("results/tables")
MANIFEST=Path("data/manifests")
SPLITS={"train","validation","test"}


def write(out, file, frame):
    file=out/file
    file.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(file,index=False,encoding="utf8",lineterminator="\n")


def read_archives(out,sample,cfg):
    meta_dir=out/META
    meta_dir.mkdir(parents=True,exist_ok=True)
    parsed,results=[],{}
    path=meta_dir/"raw_current_audit.csv.gz"
    with path.open("wb") as handle, gzip.GzipFile(fileobj=handle,mode="wb",mtime=0,filename="") as stream, \
            io.TextIOWrapper(stream,encoding="utf8",newline="") as txt:
        writer=csv.writer(txt,lineterminator="\n")
        writer.writerow(["record_id","source_line","timestamp_original","current_original","timestamp_status",
                         "current_status","same_time_conflict","malformed_fields_json"])
        for i,row in enumerate(sample.itertuples(index=False),1):
            try:
                parsed_record=load(ROOT.parent,row,cfg,writer)
            except (OSError,UnicodeError,ValueError,csv.Error) as exc:
                parsed.append(dict(record_id=row.record_id,processing_status="failed",parse_error=str(exc),
                                   clock_admissible=False))
                continue
            results[row.record_id]=parsed_record
            p=parsed_record.info
            parsed.append(dict(record_id=row.record_id,processing_status="parsed" if p["clock_admissible"] else "unverified_clock",
                               long_observation_gaps=int((p["gaps_seconds"]>cfg["long_gap_seconds"]).sum()),
                               **{key:val for key,val in p.items() if key!="gaps_seconds"}))
            if i%100==0 or i==len(sample):
                print(f"Decoded {i}/{len(sample)} sampled records",flush=True)
    return pd.DataFrame(parsed),results


def build(grids,records,parsed,cfg):
    p_rows,y_rows,f_rows=[],[],[]
    inputs,targets=defaultdict(list),defaultdict(list)
    for row in records.itertuples(index=False):
        packet=parsed.get(row.record_id)
        if packet is None or not packet.info["clock_admissible"]:
            continue
        meta=dict(record_id=row.record_id,site=row.site,cohort=row.cohort,partition=row.partition,
                  duplicate_group_id=row.duplicate_group_id,original_rows=row.original_rows,observed_duration_s=row.observed_duration_s)
        assigned=row.partition in SPLITS
        for g in grids:
            for p in cfg["prefix_minutes"]:
                x,quality=prefix(packet,p,g,cfg)
                stored=assigned and quality["inclusive"]
                p_rows.append(dict(**meta,grid_s=g,prefix_min=p,stored=stored,**quality))
                if stored:
                    inputs[row.site,g,p].append((row.record_id,x,quality["strict"]))
                for h in cfg["future_minutes"]:
                    y,label=future_target(packet,p,h,g,cfg)
                    y_rows.append(dict(**meta,grid_s=g,prefix_min=p,future_min=h,prefix_inclusive=quality["inclusive"],
                                       prefix_strict=quality["strict"],stored=stored,
                                       usable_inclusive=bool(quality["inclusive"] and label["inclusive"] and assigned),
                                       usable_strict=bool(quality["strict"] and label["strict"] and assigned),
                                       fully_observed=bool(label["span_reaches_window"] and y.observed.all()),**label))
                    if stored:
                        targets[row.site,g,p,h].append((row.record_id,y,label["strict"] and quality["strict"]))
            _,_,f=fit_comparison(packet,g,cfg)
            f_rows.append(dict(**meta,grid_s=g,**f))
    return pd.DataFrame(p_rows),pd.DataFrame(y_rows),pd.DataFrame(f_rows),inputs,targets


def grid_choice(prefix_rows):
    # Inspect only the training representative records for this pilot decision.
    p=prefix_rows.query('partition=="train" and cohort=="representative" and prefix_min==10')
    x=p.groupby(["grid_s","site"]).observed_fraction.mean().unstack()
    worst=x.min(axis=1)
    if worst[5]>=.88 and (x.loc[10]-x.loc[5]).max()<=.025:
        chosen=5
    elif worst[10]>=.88 and (x.loc[30]-x.loc[10]).max()<=.04:
        chosen=10
    else:
        chosen=30
    detail={str(int(k)):{s:float(v) for s,v in row.items()} for k,row in x.iterrows()}
    return chosen,detail


def generate(out,cfg):
    sample, strata=select(ROOT,cfg)
    meta,parsed=read_archives(out,sample,cfg)
    records=sample.merge(meta,on="record_id",validate="one_to_one")
    pairs=candidates(records,parsed,cfg)
    grouped=groups(records,pairs,parsed)
    records=records.merge(grouped,on="record_id",validate="one_to_one")
    split_table,cut=split(records,cfg)
    records=records.merge(split_table[["record_id","partition","nominal_partition","split_exclusion_reason"]],
                          on="record_id",validate="one_to_one")
    p_rows,y_rows,f_rows,inputs,targets=build(cfg["grids_seconds"],records,parsed,cfg)
    selected,train_coverage=grid_choice(p_rows)
    f_rows["stored_full"]=f_rows.grid_s.eq(selected)&f_rows.full_status.eq("eligible")&f_rows.partition.isin(SPLITS)
    for site,site_rows in records.groupby("site",sort=True):
        target_root=out/PROCESSED/site.lower()
        for g in cfg["grids_seconds"]:
            for p in cfg["prefix_minutes"]:
                pack(target_root/f"{g}s/current_{p}min.npz",inputs[site,g,p],width=p*60//g)
                for h in cfg["future_minutes"]:
                    pack(target_root/f"{g}s/future_{h}min_after_{p}min.npz",targets[site,g,p,h],width=h*60//g)
        full=[]
        for row in f_rows.loc[f_rows.site.eq(site)&f_rows.stored_full].itertuples(index=False):
            mean,_,_=fit_comparison(parsed[row.record_id],selected,cfg)
            full.append((row.record_id,mean))
        pack(target_root/"observed_record_sequences.npz",full)
    exclusions=[]
    for r in records.loc[records.split_exclusion_reason.ne("")].itertuples(index=False):
        exclusions.append(dict(record_id=r.record_id,site=r.site,scope="record_split",reason=r.split_exclusion_reason,
                               grid_s=None,prefix_min=None,future_min=None))
    for data,scope,key,fixed in [(p_rows,"prefix","quality_exclusion",False),(y_rows,"future","quality_exclusion",True)]:
        for r in data.loc[data[key].ne("")].itertuples(index=False):
            exclusions.append(dict(record_id=r.record_id,site=r.site,scope=scope,reason=getattr(r,key),
                                   grid_s=r.grid_s,prefix_min=r.prefix_min,future_min=r.future_min if fixed else None))
    for r in f_rows.loc[f_rows.full_status.ne("eligible")].itertuples(index=False):
        exclusions.append(dict(record_id=r.record_id,site=r.site,scope="observed_record",reason=r.full_status,
                               grid_s=r.grid_s,prefix_min=None,future_min=None))
    ex=pd.DataFrame(exclusions)
    write(out,MANIFEST/"pilot_sample_manifest.csv",records[list(sample.columns)+["processing_status"]])
    write(out,MANIFEST/"record_manifest.csv",records)
    write(out,MANIFEST/"dataset_split.csv",split_table)
    write(out,MANIFEST/"duplicate_groups.csv",grouped)
    write(out,MANIFEST/"exclusion_manifest.csv",ex)
    write(out,TABLE/"duplicate_candidates.csv",pairs)
    write(out,META/"sampling_strata.csv",strata)
    for name,frame in [("prefix_by_record",p_rows),("future_by_record",y_rows),("resampling_by_record",f_rows)]:
        write(out,META/(name+".csv"),frame)
    for name,frame in summarize(records,p_rows,y_rows,f_rows,ex).items():
        write(out,TABLE/name,frame)
    metadata=dict(dataset_version=cfg["dataset_version"],seed=cfg["seed"],primary_grid_seconds=selected,
                  training_only_grid_diagnostics=train_coverage, **cut, config=cfg,
                  input_sha256={p:sha256((ROOT/p).read_bytes()).hexdigest() for p in
                                ("data/manifests/raw_file_inventory.csv","data/manifests/sample_manifest.csv","config/preprocessing.yaml")},
                  representative_n=int(records.cohort.eq("representative").sum()),stress_n=int(records.cohort.eq("stress").sum()),
                  python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,pyyaml=yaml.__version__,scaler_fitted=False)
    schema=dict(value_unit="A",source="Charging Current (A)",anchor="first valid offset-aware telemetry timestamp",
                original_time_offset_preserved=True,prefix="[0, cutoff)",target="(cutoff, cutoff + horizon], right-closed bins",
                exact_cutoff="belongs to neither prefix nor future target",stored_current_dtype="float32",raw_serialized_precision="original token",
                observed_mask="direct current measurement in bin after same-time conflict masking",
                interpolated_mask="short single hole using two direct bins inside the same input window; false for targets",
                missing="both masks false, current_values=0 placeholder",
                observed_record_sequences="primary grid only; ragged arrays with offsets, offline-use only",
                future_strict="matching strict prefix, telemetry spans requested end, >=70% direct observations and <=120s blank",
                future_targets="unfilled and masked, stored for inclusive prefix cohort even when incomplete; check masks and eligibility",
                stress="diagnostics only; never used as tensors in split cohorts")
    for name,payload in [("dataset_metadata",metadata),("feature_schema",schema)]:
        (out/META/(name+".json")).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf8")
    figures(out,records,parsed,p_rows,y_rows,f_rows,selected,cfg)
    print(f"Generated bounded pilot: {len(records)} records, primary grid={selected}s, split={cut}",flush=True)


def compare(root, staging):
    names=[p.relative_to(staging) for p in staging.rglob("*") if p.is_file()]
    for name in names:
        a,b=root/name,staging/name
        if name.suffix==".npz":
            equal_npz(a,b)
        elif name.suffix==".gz":
            with gzip.open(a,"rb") as old,gzip.open(b,"rb") as new:
                while True:
                    left,right=old.read(1<<20),new.read(1<<20)
                    assert left==right,name
                    if not left:
                        break
        else:
            assert a.read_bytes()==b.read_bytes(),name
    print(f"Replay matched {len(names)} artifacts, including semantic content of {sum(n.suffix=='.npz' for n in names)} NPZ files.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true",help="Run again and compare without replacing results")
    args=parser.parse_args()
    cfg=yaml.safe_load((ROOT/"config/preprocessing.yaml").read_text(encoding="utf8"))
    with TemporaryDirectory(dir=ROOT,prefix=".phase42-") as td:
        out=Path(td)
        generate(out,cfg)
        if args.verify:
            compare(ROOT,out)
        else:
            files=[p.relative_to(out) for p in out.rglob("*") if p.is_file()]
            for name in files:
                assert not (ROOT/name).exists(),name
            for name in files:
                final=ROOT/name
                final.parent.mkdir(parents=True,exist_ok=True)
                with final.open("xb") as dst,(out/name).open("rb") as src:
                    shutil.copyfileobj(src,dst)
            print(f"Published {len(files)} Phase 4.2 artifacts.",flush=True)

