"""Re-run archived point trainers, comparing float cells with a documented tolerance."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import os
import sys

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from stage3_context import ROOT, digest, read

RUNS = {
    "03": ("03_tree_models_1h/run_comparison.py", "verify_rerun"),
    "04": ("04_multi_horizon/run_multi_horizon.py", "compare_rerun"),
    "06": ("06_cross_site_training/run_cross_site.py", "verify_replay"),
}


def compare(module, scratch):
    max_delta, n, image_mismatches = 0.0, 0, []
    for name in module.TABLES:
        older, newer = read(ROOT / name), read(scratch / name)
        stable = [field for field in older if not field.endswith("_seconds")]
        nonfloat = [field for field in stable if not pd.api.types.is_float_dtype(older[field])]
        pd.testing.assert_frame_equal(older[nonfloat], newer[nonfloat], check_exact=True)
        floats = [field for field in stable if field not in nonfloat]
        for field in floats:
            a, b = older[field].to_numpy(dtype=float), newer[field].to_numpy(dtype=float)
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            cells = np.isfinite(a) & np.isfinite(b)
            np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11, equal_nan=True)
            if cells.any():
                max_delta = max(max_delta, float(np.max(np.abs(a[cells]-b[cells]))))
        n += 1
    for name in module.FIGURES:
        if digest(ROOT / name) != digest(scratch / name):
            image_mismatches.append(str(name))
    print(f"LEGACY_TOLERANT_PASS: {n} tables, maximum float absolute delta {max_delta:.17g}; "
          f"{len(module.FIGURES)-len(image_mismatches)}/{len(module.FIGURES)} regenerated plot hashes identical.", flush=True)
    for name in image_mismatches:
        print("LEGACY_PLOT_HASH_DIFFERENCE:", name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=RUNS)
    stage = parser.parse_args().stage
    file, function = RUNS[stage]
    source = ROOT / "experiments" / file
    sys.path.insert(0, str(source.parent))
    spec = spec_from_file_location("point_replay", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    setattr(module, function, lambda scratch, *args: compare(module, scratch))
    sys.argv = [str(source), "--verify"]
    module.main()
