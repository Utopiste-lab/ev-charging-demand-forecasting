"""Publish content hashes for the data files, code, figures and reports."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd

from src.full_dataset.common import DATA, LOG, ROOT, MAN, save_csv, sha

if __name__ == "__main__":
    protected=json.loads((LOG/"protected_before.json").read_text(encoding="utf-8"))
    verified=json.loads((LOG/"protected_after.json").read_text(encoding="utf-8"))
    assert {k for k in protected if protected.get(k)!=verified.get(k)}=={"4/README.md"}
    out=LOG/"full_dataset_artifact_manifest.csv"
    def included(path):
        rel=path.relative_to(ROOT).as_posix()
        return (path.is_file() and path!=out and not rel.startswith("logs/phase43/matplotlib/")
                and (f"4/{rel}" not in protected or rel=="README.md"))
    paths=sorted(path for path in ROOT.rglob("*") if included(path))
    with ThreadPoolExecutor(max_workers=6) as pool:
        save_csv(out,pd.DataFrame([dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=h)
                                  for path,h in zip(paths,pool.map(sha,paths))]))
    print(f"Manifested {len(paths):,} artifacts.")

