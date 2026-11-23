# Stage 3 hourly forecasting

This stage is complete in the original research archive. This GitHub publication contains the code, configuration, reports, metrics, figures, small audit tables, and saved one-hour predictions. Large prediction tables remain local and are listed in [the external artifact catalogue](../docs/external-artifacts.csv); no public download is currently provided.

Start with the [research index](reports/STAGE3_RESEARCH_INDEX.md), [final report](reports/STAGE3_FINAL_REPORT.md), and [research completion checklist](reports/STAGE3_COMPLETION_CHECKLIST.md). The original checklist describes research validation, not a claim that every original archive artifact is published here. See [publication and reproduction details](../docs/data-and-repro.md).

## Verify this publication

Use Python 3.12 in a separate environment. From the repository root:

```sh
python -m venv .venv-stage3
# Activate .venv-stage3 for your shell, then:
python -m pip install -r 3/experiments/03_tree_models_1h/requirements.txt
cd 3
python -B -X utf8 verify_release.py
python -B -X utf8 verify_release.py --features
```

The first check verifies the publication manifest and recomputes one-hour MAE/RMSE from saved predictions. The second also tests feature availability, future/target perturbations, chronological label release, and boundary handling for both sites and all four horizons. These checks do not train models or require the private Stage 2 tree. They write no research outputs.

The original `experiments/02_feature_validation/run_validation.py --check-only` additionally requires the exact original sibling `2/` archive for its historical replay. Use `verify_release.py --features` for the published subset instead. Original `--verify` entry points for 04/06/07/08/09 may require the omitted full predictions, inventories, and original Stage 2 tree. Restore the catalogued artifacts before using them; the publication manifest is separate from the original frozen archive manifest.

## Retraining

From `3/`, `python -B -X utf8 experiments/03_tree_models_1h/run_comparison.py --verify` trains the official models into temporary storage and checks saved artifacts. The historical verifier uses strict comparisons and figure hashes; numerical roundoff or font/backend differences can fail those checks. The published read-only score verifier uses absolute and relative tolerances of 2e-11. Do not use `--replace` on the published evidence unless intentionally creating a new research version.

## Interpretation

The target is archived record starts per source-clock hour, not energy or occupancy. The timezone and actual operating coverage are unknown. The official test uses 3,383 keys, not the 3,384-key historical Stage 2 test. Cross-site transfer uses already observed target-site demand history without target-site fitting. Models and intervals must be compared by station, horizon, demand group, and fitting protocol.
