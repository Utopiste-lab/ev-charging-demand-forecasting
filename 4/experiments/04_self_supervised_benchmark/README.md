# Phase 4.4 runs

Use a local Python 3.12 environment with the dependencies in requirements.txt. The official CUDA 12.8 Windows wheel supports CUDA; the execution runner can fall back to CPU. From the Stage 4 root:

```powershell
$py = "experiments/04_self_supervised_benchmark/.venv_cuda/Scripts/python.exe"
& $py -B -X utf8 experiments/04_self_supervised_benchmark/run_pipeline.py
& $py -B -X utf8 experiments/04_self_supervised_benchmark/validate_results.py
& $py -B -X utf8 experiments/04_self_supervised_benchmark/render_results.py
& $py -B -X utf8 experiments/04_self_supervised_benchmark/write_reports.py
```

The pipeline first validates the official allowlist and cache, runs simple validation baselines and smoke checks, then evaluates the two architecture/objective stages. Validation choices are committed before test scoring. Each run identifier includes cohort, site, architecture, objective, input channels, and seed. The last checkpoint can resume at the next epoch when the original settings and key hashes still match. Best checkpoints are used for the fixed validation/test predictions. Reconstruction and future test targets are referenced by shared Record ID and the corresponding cached target mask rather than repeated in each prediction file.

