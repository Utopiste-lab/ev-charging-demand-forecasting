"""Reuse the frozen Experiment 08 source-clock, case and calendar definitions."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "experiments/08_adaptive_hybrid"))

from audit_previous_results import CASES, HOUR, Project, config, digest, publish, read, timestamp_hash

PROB_METRICS = Path("results/metrics/probabilistic_forecasting")
PROB_TABLES = Path("results/tables/probabilistic_forecasting")
PROB_FIGURES = Path("results/figures/probabilistic_forecasting")
FINAL_METRICS = Path("results/metrics/stage3_final")
FINAL_TABLES = Path("results/tables/stage3_final")
FINAL_FIGURES = Path("results/figures/stage3_final")
CASE_SITES = {case: (source, target) for case, source, target in CASES}
MATCHED = {"Caltech->Caltech": "Caltech matched", "JPL->JPL": "JPL matched"}
