# Data sources and terms

Dataset metadata and licenses were checked through the repository APIs on 2026-10-17.

## Beijing transaction data

Qiong Wang, Shuo Liu, Zhihan Su, Yipeng Zhang, Tao Zhang, Xiaohua Liu and Xiaochen Liu. **A city-scale dataset of public electric vehicle charging transactions and infrastructure**, version 2. DOI: https://doi.org/10.6084/m9.figshare.31952289.v2. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Metadata: https://api.figshare.com/v2/articles/31952289/versions/2

For the optional raw workflow, obtain `orders_2025-01_public.parquet` and `orders_2025-07_public.parquet` from that release and place them together in the directory passed with `--raw-dir`. The program checks SHA-256 hashes before using the files. Do not substitute another dataset version silently.

Beijing panels in this repository are transformations of these public data: eligibility filtering, endpoint aggregation, local contrasts, weights and statistical summaries. Preserve the source attribution and this description of modifications when redistributing. Raw orders, device-level records and full source infrastructure files are not bundled.

## UrbanEV

Han Li, Haohao Qu, Xiaojun Tan, Linlin You, Rui Zhu and Wenqi Fan. **Urbanev: An open benchmark dataset for urban electric vehicle charging demand prediction**. DOI: https://doi.org/10.5061/dryad.np5hqc04z. License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

Metadata: https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.np5hqc04z

The included UrbanEV panel and selected trace records derive from public station price observations. The trace is a station-level price example, not an individual charging-session record. The complete UrbanEV archive is not bundled. Saved masking draws support reaggregation; the public workflow does not regenerate the original full masking experiment.

## Scope of permissions

- Original code/documentation: MIT, see `LICENSE`.
- Beijing-derived panels and numerical derivatives: retain CC BY 4.0 attribution and indication of transformations.
- UrbanEV observations: CC0; source citation retained for scientific traceability.
- Mixed-system tables: preserve both source citations; do not describe every repository file as MIT data.
- Entire manuscript PDFs, editorial correspondence, credentials, raw orders and software runtime binaries are excluded.
