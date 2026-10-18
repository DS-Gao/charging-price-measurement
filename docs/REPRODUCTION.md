# Reproduction scope

## Price construction and cross-window assessment

The `core` workflow recalculates full-month layer summaries and covariance identities for ORIGINAL, common-date (CD) and date-weighted (DW) constructions. Fixed-member subsets are checked separately. CD uses dates observed on both sides; DW gives equal date mass within its transaction distribution and is not a median of daily medians.

Cross-window inputs are E (electricity contrast), LOCAL (local total-price contrast) and CAL (E plus a leave-one-station-out layer calibration). Formation inputs, available subsequent-window targets, common evaluation membership and formation-based weights remain distinct. Endpoint-derived contrasts are checked against stored full-precision inputs before the stored floating-point values are reused to preserve exact calibration ties.

Default bootstrap mode reaggregates saved draws. `--bootstrap regenerate` reruns 2,000 shared-station resamples with seed 20260920; station multiplicities are shared across layers, prices/calibration are fixed, and formation weights are rebuilt before conditioning. This does not propagate all latent-price reconstruction error.

Raw mode reads explicitly supplied public transaction files and rebuilds endpoints. It still uses the packaged original panel as a membership/reference object. It does not execute every historical cohort analysis, covariance fit or Monte Carlo experiment.

## Distribution workflow

The four stages recompute panel descriptions, common-task comparisons, signed/threshold extensions and amount/screening analyses. Both positive and adverse findings are retained. A frozen numerical ledger identifies historical source, row, column and value for summaries that are extracted rather than rerun. Saved covariance draws and saved masking draws are reaggregated, not generated.

Station-equal quantiles use NumPy type 7. TX/KWH weighted quantiles merge tied support values and use the specified exact-cumulative-hit midpoint rule with absolute tolerance 1e-12, rtol=0. Screening includes all tied values at its threshold rather than arbitrarily breaking ties. Zero and unavailable values are not interchangeable.

## Synthetic reconstruction experiment

`simulation` executes 15 known-truth scenarios (five error/support mechanisms by three transaction/date layouts). The copied statistical algorithm, scenarios and seeds are unchanged; its output directory is now explicit. Formal default: 2,000 repetitions per scenario. A small test demonstrates execution and invariant checks only. It cannot establish Beijing latent-price accuracy or coverage of the historical variance-share model.

## Not reproduced by this package

The complete historical model-fitting and simulation sequence, the original pairing permutation generation, full UrbanEV ingestion/mask generation, and every final figure layout are not end-to-end regenerated. Their included numerical summaries are labelled as references. This release is not evidence that real latent prices or original variance-share intervals have known coverage.

## Integrity and provenance

`python scripts/verify_release.py` checks file hashes, parses every Python source, and detects missing packaged files. The root entry point performs this check before executing a workflow; individual routines also compare estimates and expected inputs. Checksums cover the reorganized release. Original scientific input bytes are retained, with previous locations recorded in `docs/file_origin_map.csv`. Do not use Python's `-O` option because scientific checks use assertions.
