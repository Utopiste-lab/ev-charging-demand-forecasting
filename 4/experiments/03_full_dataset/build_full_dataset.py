"""Run the pilot gate then build independent, resumable archive batches."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pandas as pd

from src.full_dataset.common import CONFIG, LOG, MAN, ROOT, config, sha
from src.full_dataset.compatibility import run as check_pilot
from src.full_dataset.processing import build_batch, hash_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--workers", type=int)
    opts = parser.parse_args()
    check_pilot()
    if opts.pilot_only:
        return
    cfg, pc = config()
    files = [CONFIG, ROOT / cfg["pilot_config"], ROOT / "src/full_dataset/processing.py",
             ROOT / "src/preprocessing/parser.py", ROOT / "src/preprocessing/resampling.py"]
    signature = hash_bytes("".join(sha(p) for p in files).encode())
    inv = pd.read_csv(MAN / "source_inventory.csv", keep_default_na=False)
    query = inv.loc[inv.is_archive].sort_values(["site","filename_timestamp","source_file"],ignore_index=True)
    size = cfg["batch_records"]
    start = time.perf_counter()
    with (LOG/"batch_execution.log").open("a", encoding="utf-8", buffering=1) as log:
        with ProcessPoolExecutor(max_workers=opts.workers or cfg["workers"]) as pool:
            tasks = [pool.submit(build_batch,f"batch_{k//size:04d}",query.iloc[k:k+size].to_dict("records"),
                                 signature,cfg,pc) for k in range(0,len(query),size)]
            for count, task in enumerate(as_completed(tasks), 1):
                name, n, status, seconds = task.result()
                line = f"{count}/{len(tasks)} {name} {status} {n} records {seconds:.2f}s elapsed={time.perf_counter()-start:.1f}s"
                print(line, file=log)
                if count % 10 == 0 or count == len(tasks):
                    print(line, flush=True)


if __name__ == "__main__":
    main()

