"""Verify historical contents and raw metadata, then record closure artifacts."""

import argparse
from pathlib import Path

import pandas as pd

from stage3_context import FINAL_TABLES, ROOT, digest, publish, read

MANIFEST = FINAL_TABLES / "final_artifact_manifest.csv"
NEW = {
    "config/probabilistic_forecasting.yaml",
    "reports/09_probabilistic_forecasting.md",
    "reports/STAGE3_FINAL_REPORT.md",
    "reports/STAGE3_RESEARCH_INDEX.md",
    "reports/STAGE3_COMPLETION_CHECKLIST.md",
    "reports/STAGE4_READINESS_ASSESSMENT.md",
    "logs/phase10_probabilistic_forecasting.log",
}
DIRECTORIES = ("experiments/09_probabilistic_forecasting/", "src/probabilistic_forecasting/",
               *[f"results/{kind}/{view}/" for kind in ["metrics","tables","figures"]
                 for view in ("probabilistic_forecasting","stage3_final")])


def verify_preserved():
    root=ROOT.parent
    prior=read(ROOT / FINAL_TABLES / "previous_stage2_stage3_sha256.csv")
    raw=read(ROOT / FINAL_TABLES / "raw_archive_metadata.csv",usecols=["path","bytes","mtime_ns"])
    for row in prior.itertuples(index=False):
        p=root/row.path
        stat=p.stat()
        assert stat.st_size==row.bytes and stat.st_mtime_ns==row.mtime_ns and digest(p)==row.sha256,row.path
    actual2={p.relative_to(root).as_posix() for p in (root/"2").rglob("*") if p.is_file()}
    assert actual2==set(prior.loc[prior.path.str.startswith("2/"),"path"])
    actual_raw=[]
    for p in sorted((root/"raw_data").rglob("*")):
        if p.is_file():
            st=p.stat()
            actual_raw.append({"path":p.relative_to(root).as_posix(),"bytes":st.st_size,"mtime_ns":st.st_mtime_ns})
    pd.testing.assert_frame_equal(pd.DataFrame(actual_raw),raw,check_exact=True)
    original3 = {p.removeprefix("3/") for p in prior.loc[prior.path.str.startswith("3/"),"path"]}
    actual3={p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
    introduced=actual3-original3
    assert original3.issubset(actual3)
    assert all(p in NEW or p.startswith(DIRECTORIES) for p in introduced),sorted(introduced)
    print(f"Preserved {len(prior)} Stage 2/3 files byte-for-byte and {len(raw):,} raw paths/sizes/mtime values.",flush=True)
    return introduced


def artifact_manifest(paths):
    rows=[]
    for relative in sorted(paths - {MANIFEST.as_posix()}):
        p=ROOT/relative
        rows.append({"path_relative_to_stage3":relative,"bytes":p.stat().st_size,"sha256":digest(p)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    args=parser.add_mutually_exclusive_group()
    args.add_argument("--freeze",action="store_true")
    args.add_argument("--verify-manifest",action="store_true")
    flags=parser.parse_args()
    introduced=verify_preserved()
    if flags.freeze or flags.verify_manifest:
        manifest=artifact_manifest(introduced)
        if flags.freeze:
            publish(manifest,ROOT/MANIFEST)
        else:
            pd.testing.assert_frame_equal(manifest,read(ROOT/MANIFEST),check_exact=True)
        print(f"Verified {len(manifest)} closure artifacts. The manifest excludes its own hash.")
