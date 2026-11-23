# Historical Stage 2 results

The original root `models/`, `results/`, `figures/`, and `data/` paths remain available. The root [historical model results](../results/model_results.csv) use 3,384 Caltech test hours and a training-only fit. They are not Stage 3's official 3,383-hour comparison.

| Historical model | MAE | RMSE |
|---|---:|---:|
| Seasonal Naive | 0.2786643026004728 | 0.7131407574614697 |
| Ridge Regression | 0.6709775370734611 | 0.9907522902648627 |

The legacy runner now accepts either the original component directories or the repository's flat six-CSV layout. If both are present, explicitly choose `--input-layout component` or `--input-layout flat`; never combine the layouts. Missing inputs produce a complete lookup list.

From the repository root, after installing the root requirements into a separate legacy environment:

```sh
python -B -X utf8 models/train_baselines.py --data-root data/processed --input-layout flat --output-dir work/stage2-rerun
```

Outputs are written to `work/stage2-rerun`, leaving historical CSVs intact. The existing [model documentation](../models/README.md) describes the original archived layout and training protocol. Its workstation examples are historical, not the current repository's running instructions.
