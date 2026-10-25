# Multi-Horizon and Cross-Site Electric Vehicle Charging Demand Forecasting

## Project Overview

This repository supports a machine learning project on forecasting electric vehicle (EV) charging demand across multiple prediction horizons and charging sites. The study uses historical charging-session data from two ACN-Data-Static locations and investigates how well forecasting methods generalize over time and between sites.

The current repository stage includes data-processing work and two baseline models: **Seasonal Naive** and **Ridge Regression**. The preliminary results reported below are limited to a one-hour-ahead experiment at the Caltech site. Multi-horizon and cross-site experiments are planned but are not presented as completed results.

## Research Objectives

The project aims to:

1. Build a reproducible pipeline for transforming raw EV charging-session records into time-indexed demand data.
2. Evaluate forecasting accuracy at multiple future horizons.
3. Compare performance across two charging sites with different demand patterns.
4. Establish transparent baseline results before testing more advanced forecasting methods.
5. Study temporal distribution shift and cross-site generalization.

## Data Sources

The project uses **ACN-Data-Static** data from the following locations:

- **Caltech:** `Caltech/California_Garage_01`
- **JPL:** `JPL/Arroyo_Garage_01`

These identifiers should be preserved in the repository documentation so that each experiment can be traced to its source site.

> **Repository verification required:** The exact raw-data filenames, file extensions, and local directory paths must be checked against the final repository. Do not assume that the source identifiers above are identical to the filenames committed to the repository. Any access conditions or licensing requirements associated with ACN-Data-Static should also be documented before public release.

## Forecasting Target

The forecasting task is defined on an hourly time grid. At forecast origin \(t\), a model predicts the EV charging-demand count for a future time step \(t+h\), where \(h\) is the forecast horizon measured in hours.

- **Current completed evaluation:** Caltech, one-hour-ahead forecasting (\(h=1\)).
- **Planned scope:** additional forecast horizons and evaluation at or across both Caltech and JPL.

The final repository should state precisely how a session is assigned to an hourly interval—for example, by arrival time—and how boundary timestamps, missing intervals, and time zones are handled. If the implemented target differs from an hourly session or arrival count, this section must be revised to match the code and processed data.

## Data-Processing Workflow

The intended processing workflow is:

1. Load the site-specific ACN-Data-Static records.
2. Parse timestamps and apply a consistent time-zone convention.
3. Select records and fields required for the forecasting target.
4. Aggregate charging activity into a continuous hourly time series.
5. Represent hours with no qualifying activity as zero, subject to data-completeness checks.
6. Create lagged and calendar-based predictors required by each model.
7. Split observations chronologically to avoid leakage from future data.
8. Fit models using the training period only.
9. Generate predictions for the specified site, horizon, and test interval.
10. Evaluate predictions using MAE and RMSE and save the relevant outputs.

Exact preprocessing rules must be verified against the final implementation. In particular, long zero-count periods should not automatically be treated as genuine inactivity until data completeness has been checked.

## Completed Models

### Seasonal Naive

The Seasonal Naive model uses a historical observation from the relevant seasonal lag as its forecast. It provides an interpretable benchmark for determining whether learned models improve on recurring demand patterns.

> The exact seasonal lag used by the implementation must be documented after it is verified in the final code.

### Ridge Regression

Ridge Regression is a linear model with L2 regularization. It is used as a learned baseline with lagged and/or calendar features derived from the hourly demand series.

> The final feature list, regularization setting, fitting procedure, and any feature scaling must be documented from the verified implementation. Standard Ridge Regression is unconstrained and can therefore produce negative demand predictions.

## Preliminary Results

The following results have been completed for **Caltech one-hour-ahead forecasting**:

| Model | MAE | RMSE |
|---|---:|---:|
| Seasonal Naive | 0.2787 | 0.7131 |
| Ridge Regression | 0.6710 | 0.9908 |

- **Test interval:** 2020-07-15 to 2020-12-02
- **Number of test observations:** \(n=3384\)

For this experiment, Seasonal Naive achieved lower MAE and RMSE than Ridge Regression. This is a preliminary site- and horizon-specific result; it should not be interpreted as evidence that Seasonal Naive will outperform Ridge Regression at other horizons, at JPL, or in cross-site evaluation.

## Known Limitations

- **Distribution change during the test period:** The test interval may differ materially from the training distribution. Results should be interpreted in the context of this temporal shift.
- **Long zero-count intervals:** Extended sequences of zero demand require verification. They may reflect genuine inactivity, incomplete data, filtering choices, or an issue in timestamp aggregation.
- **Negative Ridge predictions:** The Ridge model may output negative values, which are not physically meaningful for a non-negative demand count. Results should report whether predictions were evaluated as produced or post-processed; any clipping rule must be applied consistently and documented.
- **Limited completed evaluation:** The reported results cover only Caltech at a one-hour horizon. Multi-horizon, JPL, and cross-site results remain future work.
- **Implementation details pending repository validation:** Data filenames, processed-data schemas, feature definitions, script entry points, and commands must be checked against the final repository before they are treated as authoritative.

## Suggested Repository Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt              # or environment.yml / pyproject.toml
├── data/
│   ├── README.md                 # provenance, access, schema, and licensing
│   ├── raw/                      # usually excluded if redistribution is restricted
│   │   ├── Caltech/
│   │   │   └── California_Garage_01/
│   │   └── JPL/
│   │       └── Arroyo_Garage_01/
│   └── processed/
├── notebooks/                    # exploration only; name notebooks descriptively
├── src/
│   ├── data/                     # loading, validation, and preprocessing
│   ├── features/                 # lag and calendar feature construction
│   ├── models/                   # Seasonal Naive and Ridge implementations
│   └── evaluation/               # metrics, plots, and result export
├── configs/                      # site, horizon, split, and model settings
├── scripts/                      # verified command-line entry points
├── results/
│   ├── metrics/
│   ├── predictions/
│   └── figures/
└── tests/
```

This is a recommended organization, not a claim about the repository's current contents. Rename, add, or remove paths so that this section matches the final committed structure. Large or restricted raw datasets should not be committed unless redistribution is permitted; provide retrieval instructions instead.

## Dependencies and Environment

The exact dependency list must be generated from the verified project environment. A typical Python forecasting workflow may use packages such as:

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib and/or Seaborn
- Jupyter, if notebooks are retained

Package names and versions should be recorded in one authoritative environment file, such as `requirements.txt`, `environment.yml`, or `pyproject.toml`. Remove dependencies that the final code does not use and pin versions where reproducibility requires it.

## Running the Project

No unverified command is presented here as directly runnable. Before release, replace the placeholders below with commands that have been tested from the repository root.

```bash
# 1. Create and activate the documented environment.
# <verified environment setup command>

# 2. Obtain or place the ACN-Data-Static files at the documented paths.
# <verified data acquisition or placement instructions>

# 3. Build the processed hourly data.
# <verified preprocessing command>

# 4. Run a model for a selected site and forecast horizon.
# <verified training/evaluation command>

# 5. Reproduce the reported metrics and outputs.
# <verified results-generation command>
```

Before replacing these placeholders, confirm:

- the actual filenames and paths for both sites;
- the expected working directory;
- all required command-line arguments or configuration files;
- the train/test split and time-zone convention;
- the output locations for predictions, metrics, and figures; and
- that the documented commands complete successfully in a clean environment.

## Future Work

The following items are planned and should not be treated as completed:

- Validate the long zero-count intervals against the source data and ingestion process.
- Finalize and document the target construction, time-zone handling, and chronological split.
- Verify all repository paths, filenames, dependencies, and execution commands.
- Evaluate additional forecast horizons using a consistent protocol.
- Run corresponding experiments for `JPL/Arroyo_Garage_01`.
- Evaluate cross-site transfer, including training on one site and testing on the other where methodologically appropriate.
- Investigate transformations, constrained models, or documented clipping to address negative Ridge predictions.
- Compare additional statistical and machine learning models against the two completed baselines.
- Add automated tests for preprocessing, feature construction, leakage prevention, and metric calculation.
- Record reproducibility metadata such as random seeds, package versions, data snapshots, and configuration files.

## Reproducibility Checklist

Before publishing the repository, verify that:

- [ ] Data provenance and access instructions are complete.
- [ ] Raw-data redistribution is permitted, or raw files are excluded.
- [ ] Actual data filenames and paths match the documentation.
- [ ] The target and seasonal lag are defined exactly as implemented.
- [ ] Train, validation, and test boundaries are documented.
- [ ] Feature generation uses no future information.
- [ ] Dependency versions are recorded.
- [ ] Every documented command has been run successfully from a clean checkout.
- [ ] Saved predictions reproduce the reported MAE and RMSE.
- [ ] Negative-prediction handling is explicit.
- [ ] Zero-count intervals have been investigated and documented.

## Current Project Status

**Completed:** data-processing work for the current experiment, Seasonal Naive baseline, Ridge Regression baseline, and preliminary Caltech one-hour-ahead evaluation.

**Not yet established as completed:** additional forecast horizons, JPL evaluation, cross-site evaluation, advanced models, final reproducibility packaging, and verified end-to-end run commands.

