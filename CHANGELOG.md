# Changes

## Repository organization, 2026-10-18

- Separate analysis routines (`src`), execution scripts (`scripts`), configurations (`config`), derived data and numerical references (`data`).
- Retain all scientific input bytes, thresholds, weights and seeds; preserve previous file identities in `docs/file_origin_map.csv`.
- Add a small execution example, numerical regression tests and cross-platform continuous-integration commands.
- Replace nested revision-package entry points with a single root runner and a shared integrity manifest.
- Rewrite the homepage around the research, runnable tasks and paper-to-code correspondence.

This packaging revision does not add a new statistical method or extend the empirical validation claims.
