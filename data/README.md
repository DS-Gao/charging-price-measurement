# Data layout

`derived/` contains analysis-ready station/boundary inputs; `reference/` contains expected estimates, saved resampling results and clearly identified historical summaries.

The 900 masking-draw CSVs remain individually inspectable under `reference/masking/draws/`. Historical summary sources are indexed in `config/reference_sources.json`. No input was rounded or recomputed merely to rearrange this repository.

See `DATA_SOURCES.md` at the repository root for original dataset citations and licensing; see `docs/DATA_DICTIONARY.md` for fields and units. Do not place private records or credentials here. Supply the original public transactions through `--raw-dir` rather than committing them to Git.
