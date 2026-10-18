# Component-resolved measurement of EV charging prices

Research code for **Component-resolved measurement of time-of-use prices for public charging assessment**  
Dongsheng Gao · Junjie Hu · Jaeyoung Jay Lee

This repository measures local electricity, service and total-price contrasts in public EV charging. It examines observation support, date-aligned price construction, component pairing and population weights, then compares historical price inputs on a shared subsequent-window target.

The study uses Beijing public charging transactions and independent UrbanEV station-price observations. See [data sources and licenses](DATA_SOURCES.md) and the [paper-to-code map](docs/PAPER_MAP.md).

## Quick start

Requires Python 3.12. Clone the repository, create a virtual environment, and install the tested dependencies:

```sh
git clone https://github.com/DS-Gao/charging-price-measurement.git
cd charging-price-measurement
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python run.py core --output ../results_core
```

The core command runs from the included derived inputs. It writes numerical comparisons, cross-window estimates and diagnostic figures. The expected formation entry is **6,883 units**, with **5,846** units in the common evaluation set. Every workflow requires a new output directory outside the repository.

## Reproduce the analyses

| Task | Command | What is calculated |
|---|---|---|
| Price construction and cross-window assessment | `python run.py core --output ../results_core` | ORIGINAL/CD/DW summaries, E/CAL/LOCAL estimates, coverage and intervals from saved draws |
| Regenerate cross-window resampling | `python run.py core --bootstrap regenerate --output ../results_bootstrap` | The same estimates plus 2,000 new shared-station resamples, compared with references |
| Components, support, weights and screening | `python run.py distribution --output ../results_distribution` | Derived-panel analyses and reaggregation of saved covariance and masking draws |
| Small synthetic example | `python run.py simulation --reps 8 --output ../results_example` | 15 known-truth reconstruction scenarios, eight repetitions each |
| Raw Beijing reconstruction | `python run.py raw --raw-dir ../raw_data --output ../results_raw` | Rebuild endpoints from the two separately downloaded transaction files |

For the formal reconstruction simulation, use `--phase formal --reps 2000`. This is substantially larger than the small execution example.

## Layout

```text
charging-price-measurement/
├── run.py                  # Single entry point
├── src/
│   ├── prices/             # Reconstruction, estimators and resampling
│   ├── distribution/       # Components, weights and input-use analyses
│   ├── extensions/         # Signed contrasts, thresholds and masking summaries
│   └── simulation/         # Known-truth reconstruction experiment
├── scripts/                # Workflow runners and file verification
├── config/                 # Analysis settings and reference-source index
├── data/
│   ├── derived/            # Analysis-ready inputs
│   └── reference/          # Expected estimates, saved draws and historical summaries
├── examples/               # Small example and expected output guide
├── docs/                   # Reproduction scope, data dictionary and paper mapping
└── tests/                  # Numerical and command-line regression checks
```

Raw transaction files and the full UrbanEV archive are obtained from their original repositories. The supplied derived data, saved draws and reference summaries preserve their original numerical values. Generated results go into the output directory, separate from source code and inputs.

## Reproduction scope

The package distinguishes **recalculation from derived inputs**, **raw reconstruction**, **saved-draw reaggregation**, and **frozen-summary extraction**. The complete historical covariance fitting, pairing-permutation generation and UrbanEV mask generation are not rerun by the public entry points. Diagnostic plots are not exact reproductions of every final manuscript layout. Details and actual checks are in [REPRODUCTION.md](docs/REPRODUCTION.md) and [VALIDATION.md](docs/VALIDATION.md).

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite the code, and cite the source datasets separately. The manuscript has no assigned publication DOI in this release.

Original code and documentation: [MIT](LICENSE). Beijing-derived data retain CC BY 4.0 attribution; UrbanEV data are CC0. See [DATA_SOURCES.md](DATA_SOURCES.md) for the distinction.

Questions or reproducibility issues can be reported through the repository's issue tracker.
