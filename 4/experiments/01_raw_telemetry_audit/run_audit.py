"""Phase 4.1 bounded read-only telemetry audit, with an independent output replay."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data_audit.inventory import inventory, sample
from src.data_audit.summaries import summarize, duplicates, plot
from src.data_audit.telemetry import inspect

plt.rcParams["font.family"]="DejaVu Sans"

MANIFEST="data/manifests"
TABLE="results/tables"
FIG="results/figures"


def generate(out):
    cfg=json.loads((ROOT/"config/phase41.json").read_text(encoding="utf8"))
    project=ROOT.parent
    for where in (MANIFEST,TABLE,FIG,"data/samples"):
        (out/where).mkdir(parents=True,exist_ok=True)
    inv=inventory(project,cfg)
    chosen=sample(inv,cfg)
    quality, schema, states, early, views=[],[],[],[],{}
    for entry in chosen.to_dict("records"):
        meta={k:entry[k] for k in ["site","path","sample_group","filename_source_time","identifier_token_unverified"]}
        p=project/entry["path"]
        assert p.resolve().is_relative_to((project/"raw_data").resolve())
        assert (p.stat().st_size,p.stat().st_mtime_ns)==(entry["bytes"],entry["mtime_ns"])
        try:
            quality_row, fields, classes, feasibility, view=inspect(p,meta,cfg)
        except (OSError,UnicodeError,ValueError) as e:
            quality.append(dict(**meta,read_status="failed",read_error=repr(e)))
            continue
        quality.append(quality_row)
        schema.extend(fields)
        states.extend(classes)
        early.extend(feasibility)
        views[entry["path"]]=view
    q,fields,states,early=map(pd.DataFrame,(quality,schema,states,early))
    manifest=chosen.merge(q[["path","read_status","compressed_sha256","payload_sha256", *(["read_error"] if "read_error" in q else [])]],
                           on="path",validate="one_to_one")
    summary=summarize(inv,q,fields,states,early)
    for r in q.loc[q.read_status.ne("failed")].itertuples(index=False):
        # Only sampled archives are content-hashed. All other files use frozen metadata.
        with (project/r.path).open("rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest()==r.compressed_sha256
    assert len(q)==len(manifest) and len(early)==2*q.read_status.ne("failed").sum()
    assert summary["early_prediction_feasibility"].input_screen_pass.le(summary["early_prediction_feasibility"].n_read_ok).all()
    for label in ("observed_remaining_duration_proxy","observed_energy_proxy","future_power_10m","future_power_30m"):
        assert early[label].le(early.input_screen_pass).all()
    frames={f"{MANIFEST}/raw_file_inventory.csv":inv, f"{MANIFEST}/sample_manifest.csv":manifest,
            f"{TABLE}/sample_quality_summary.csv":q,f"{TABLE}/raw_data_schema.csv":fields,f"{TABLE}/charging_states.csv":states,
            f"{TABLE}/early_prediction_by_sample.csv":early,f"{TABLE}/sample_duplicate_screen.csv":duplicates(q),
            "data/samples/figure_sources.csv":plot(out/FIG,q,views,cfg["max_gap_seconds"]),
            **{f"{TABLE}/{name}.csv":table for name,table in summary.items()}}
    for relative, data in frames.items():
        data.to_csv(out/relative,index=False,encoding="utf-8",lineterminator="\n")
    print(f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
          f"matplotlib {matplotlib.__version__}, seed {cfg['seed']}")
    print(f"{len(inv)} metadata entries, {len(chosen)} archives attempted; {q.read_status.value_counts().to_dict()}.")
    print(f"{q.uncompressed_bytes.sum():,.0f} sampled uncompressed bytes in memory only.")
    print(f"Verified {q.read_status.ne('failed').sum()} sampled compressed SHA-256 values; prefix outcome checks passed.")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify",action="store_true",help="Replay bounded audit and compare artifacts without replacing published tables.")
    verify=cli.parse_args().verify
    if verify:
        with TemporaryDirectory(dir=ROOT,prefix=".phase41-replay-") as temp:
            target=Path(temp)
            generate(target)
            files=[p.relative_to(target) for p in target.rglob("*") if p.is_file()]
            for rel in files:
                assert (target/rel).read_bytes()==(ROOT/rel).read_bytes(),rel
            print(f"Exact replay matched {len(files)} tables and plots.")
    else:
        generate(ROOT)

