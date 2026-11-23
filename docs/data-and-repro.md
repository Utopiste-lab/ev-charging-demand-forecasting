# Stage 3 data and reproduction

This publication updates Stage 3 only. Existing Stage 4 files are not a complete Stage 4 release and were not validated as part of this update.

## Data and provenance

The existing repository already contains `3/data/processed/hourly_counts_v1.csv`: 41,603 hourly rows, comprising Caltech (22,728) and JPL (18,875). This update preserves that file byte for byte. Its SHA-256 is `e068400dcfa4fbbf04ef37d336d3fd4c0a19a6464ff1348aff981cf0cb9d9a2f`. The [provenance JSON](../3/data/processed/data_provenance.json) retains source identifiers and hashes; only its private workstation root has been replaced with a descriptive label. No new raw archive or session-level identifier dataset is published.

One archived source record is counted at the hour indicated by its filename, in a left-closed/right-open interval. Source-clock timestamps are retained without timezone conversion. A zero means no recorded starts; actual operation and archive completeness remain unknown. First/last-hour flags preserve boundary limitations. See the [data contract](../3/config/data_contract.yaml).

ACN-Data-Static source sites are `Caltech/California_Garage_01` and `JPL/Arroyo_Garage_01`. Data source and citation entries are provided at [ACN Data](https://ev.caltech.edu/dataset) and [ACN-Data-Static](https://github.com/zach401/ACN-Data-Static). Check the source's access and redistribution terms before distributing further derived or raw assets. User authorization to update this repository does not select a code license or establish third-party data redistribution rights. No LICENSE has been invented in this update; project rights holders should select one.

## Publication levels

| Level | Included | How to verify |
|---|---|---|
| Read results | All Stage 3 reports, metric tables, figures and selected audit tables | Follow the research index; inspect exact CSV values |
| Recompute one-hour scores | Official one-hour saved predictions, baseline predictions, metrics and hourly input | Run `python -B -X utf8 3/verify_release.py` from the repository root |
| Verify feature causality | Hourly input, shared code, configuration, original A–D checks | Run `python -B -X utf8 3/verify_release.py --features` |
| Full archive replay | Requires large prediction tables and private historical inventories not included here | Restore exact original layout and hashes before using original full replay scripts |

Install dependencies as documented in the [Stage 3 README](../3/README.md). Dependencies retain the original relative requirements chains and pinned research versions. The lightweight score and causality checks need no raw archive. Commands resolve inputs from the script location; they do not overwrite frozen outputs.

The [publication manifest](artifact-index.csv) records repository paths, byte sizes, source paths, and SHA-256 for the published Stage 3 files. It excludes itself to avoid recursive hashing. The [external artifact catalogue](external-artifacts.csv) records omitted original files with exact byte sizes and hashes. `local-only` means unavailable from this repository; it is not a working download link. The archived `final_artifact_manifest.csv` and `previous_stage2_stage3_sha256.csv` are historical evidence and must not be used as checksums for this publication subset. The publication provenance JSON intentionally differs from the original archive.

Large cross-site, adaptive, quantile, OOF and rolling predictions should be published separately only after their distribution terms have been resolved. A future release must provide a versioned download, byte size, SHA-256 and extraction paths. No claim of full 01–09 retraining or full probability-score recomputation is made for this subset.

## Protocol and limits

Official fixed tests use 2020-07-15 00:00 to 2020-12-02 22:00, inclusive (3,383 target hours). Models use 34 features available at issue time, with chronological release-aware partition boundaries. Learned point models select on validation MAE, then refit on released training plus validation labels. Experiment 09 has an independent calibration period and a different final fitting window. Cross-site runs use matched source history and already observed target history; they are not a test of transfer with no target observations.

Low overall MAE does not imply positive-demand detection: Caltech constant-zero MAE is 0.177949 with zero positive recall. Probability interval coverage must also be read by zero/positive groups. Retain the [legacy Stage 2](stage2-legacy.md) results separately; changes in test keys and fitting protocol prevent interpreting the Stage 2/3 score difference as pure algorithm improvement.
