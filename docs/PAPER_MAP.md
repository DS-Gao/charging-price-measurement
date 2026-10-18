# Paper-to-code map

Current manuscript: *Component-resolved measurement of time-of-use prices for public charging assessment*. Topics and stable labels are preferred over historical figure numbering.

| Paper topic | Code | Output / scope |
|---|---|---|
| Local contrasts, observation support and date construction | `src/prices/reconstruction_engine.py`, `module_a_stats.py` | ORIGINAL/CD/DW layer summaries and fixed-member comparisons; optional raw endpoints |
| Cross-window price information and coverage | `src/prices/module_b_stats.py`, `fast_bootstrap.py` | E/CAL/LOCAL point estimates, coverage and shared-station intervals |
| Component pairing and represented populations | `src/distribution/portable_main_results.py` | Component quantiles, pairing points, weighted points and saved-draw summaries |
| Common-task comparisons and input screening | `src/distribution/build_task_comparisons.py`, `analyze_input_application.py` | Label agreement, amount distributions, tied-threshold screening and concentration |
| SI1 signed and threshold extensions | `src/extensions/analyze_extensions.py` | Signed/threshold/support tables from the included derived panels |
| SI1 external examples and masking | `src/prices/urbanev_adapter.py`, `src/extensions/fix_e2_reporting.py` | Five trace cases recalculated; 900 saved masking draws reaggregated |
| SI2 known-truth reconstruction | `src/simulation/reconstruction.py` | Runnable 15-scenario transaction experiment |
| Other historical reliability/model results | `config/reference_sources.json`, `data/reference/historical/` | Frozen numerical extraction with exact source/row/column locators, not a model refit |
| Appendix A/B technical definitions | Price and distribution functions above | Implementation reference; the manuscript defines the scientific objects |

The `core` output directory contains `module_a_layer_summary_recalculated.csv`, `module_a_covariance_recalculated.csv`, `module_b_point_estimates_recalculated.csv`, `module_b_coverage_recalculated.csv` and `module_b_intervals_*.csv`.

The `distribution` output directory contains `main/`, `common/`, `extensions/`, `application/` and `figures/`. `main/frozen_numeric_ledger.csv` identifies the summaries that were read from references rather than newly estimated.

`manuscript_object_index.csv` identifies all 85 current displays and their source locations in the manuscript project. This index is navigation, not a claim that all displays are regenerated here. `file_origin_map.csv` records the migration from the preceding local package; those previous paths are provenance, not runtime dependencies.
