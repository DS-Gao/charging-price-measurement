# Transaction reconstruction and error propagation: prospective supplement

Design dated 2026-09-13. Exploratory extension of the measurement-framework
study, specified before its pilot and formal results were inspected. It is not
preregistration or external validation. Public-release edits on 2026-10-17
remove internal workflow bookkeeping and describe the packaged scope; the
statistical specification below is retained. The separate V2c experiment
discussed for target comparison is not implemented by this entry point.

## Aim and separation of targets

V2b tests the recovery of population medians of eligible, contained transactions,
their adjacent contrasts, electricity qualification and cross-station IQR. V2c
separately tests interval coverage for scalar averages of repeated boundary
contrasts with known shared-segment covariance (see shared_segment/design.md).
Neither evaluates decision-time tariffs, selection of crossing sessions, the
empirical Beijing error distribution, or coverage of fitted variance components.

The distinction between a fixed finite panel and a sampled station population is
essential. V2b holds 180 latent stations fixed and redraws transactions/days and,
in selected scenarios, endpoint availability. It does not use a station sampling
interval to claim coverage of this fixed panel. V2c draws independent stations
from its explicit Gaussian population and evaluates its own linear estimands.

## V2b latent panel and known truth

One direction-standardized boundary (q=+1) has two endpoint segments. There are
30 stations in each of six electricity-change types: -0.020, 0.003, 0.009,
0.011, 0.050 and 0.200 RMB/kWh. A fixed uniform perturbation within +/-0.0004
avoids artificial exact ties. Before electricity levels are 0.60--0.75 and
service levels 0.45--0.50. Service changes cycle through -0.070, -0.020, 0,
0.025 and 0.080 with fixed jitter +/-0.002. The entire panel is generated once
using seed 9132610, exported and held fixed. These illustrative values are not
estimates of a market data-generating process. Weak and near-threshold types are
deliberately included, not fitted to observed category frequencies.

For each transaction, E=e_location+class_E+day_E+U_E and
S=s_location+class_S+day_S+U_S, with T=E+S exactly. Independent U_E/U_S are uniform
within their stated halfwidths. All generated component prices are positive.
Inputs represent already eligible, fully contained transactions. A separate
positive-energy fee/energy round-trip check verifies the conversion identity;
timestamp filtering, missing bill fields and cross-boundary billing are outside
this experiment.

Five processes are crossed with three support layouts (15 scenarios):

| Process | Transaction halfwidths E/S | Additional structure |
|---|---|---|
| iid_low | 0.015 / 0.020 | Independent single-price noise |
| iid_high | 0.060 / 0.080 | Fourfold transaction noise |
| multi_price | 0.015 / 0.020 | Three shared transaction classes |
| shared_day | 0.015 / 0.020 | Uniform(-1,1) shock common to a station/endpoint/day, multiplied by 0.040 for E and 0.060 for S |
| selective_support | 0.015 / 0.020 | Endpoint pass probability 0.55 for true E*>0.01, 0.90 otherwise; independent between endpoints and independent of price noise conditional on true type |

Support layouts are 20 transactions over 5 days (4/day), 80 over 5 (16/day),
and 80 over 20 (4/day), at EACH endpoint. Thus 20/5 versus 80/5 varies
transactions while holding day support fixed; 80/5 versus 80/20 varies days
while holding transaction count fixed. Equal transactions/day make the marginal
transaction price distribution and its median the same target in all layouts.
All endpoints meet the manuscript's 20-transaction/5-day rule except explicitly
failed endpoints in selective_support, whose support metadata are 8/2. Prices
for those failed endpoints are not used or imputed. Boundary A requires both
endpoints to pass. Selection probabilities are illustrative, not a fitted
observation model; the resulting coverage does not estimate Beijing coverage.

Multi-price class offsets are E=(0,0.025,-0.020), S=(0,-0.140,0.120).
Before probabilities are (0.55,0.30,0.15), after (0.35,0.45,0.20).
Classes are iid across transactions and shared between E and S within each
transaction. Population component medians are roots of their exact mixtures of
uniform CDFs. Total medians use the exact convolution CDF of U_E+U_S within each
class. Thus median(T) need not equal median(E)+median(S); the known aggregation
residual is retained rather than silently defining it as zero. CDF roots are
checked independently by numerical convolution/quadrature. For the symmetric
single-price and shared-day processes, population medians equal their locations.
Shared-day truth is the marginal distribution over newly generated days, not
the realized finite set of day shocks.

## Estimation and performance

Apply sample medians separately to E, S and T at each endpoint, and subtract
after-minus-before. Electricity categories are E*>0.01 (R), E*<-0.01
(discordant), and otherwise weak. Total categories use +/-0.03. Report full
confusion counts and replicate-level category error, true/estimated prevalence,
R sensitivity and precision, and error near the qualification threshold.

Primary price metrics on A: E*, T*, D30=30(T*-E*) and residual error, including
bias, mean absolute error and mean squared error. All reference RMB use 30 kWh;
they are not actual bills or behavioral thresholds.

FIQR uses NumPy linear/type-7 unweighted quartiles, matching the manuscript.
Separate (i) price error with true R fixed, (ii) error on estimated R, and
(iii) the change in the true-price FIQR induced by replacing true R with
estimated R. The last two sum to the combined error exactly. Report undefined
groups and preserve missing outputs; never silently omit failed replicates.

For selection, report A coverage and distinguish true mean |B30| on all 180
stations from its true value on each observed A. Then separate the estimator's
error on A from the composition difference between A and the full panel. This
is an observation-selection diagnostic, not a bias correction or causal result.

Each scalar performance metric is summarized over independent panel
replications, with its Monte Carlo standard error. Confusion cells pooled over
stations are counts, not independent-trial confidence intervals. The three
layouts use separate streams; any difference between their mean metrics uses
the corresponding independent-stream MCSE. No empirical sampling CI is
claimed for the fixed latent panel.

## Execution, checks and stopping rule

Run 200 independent replications per scenario as a pilot; proceed to 2,000 per
scenario using a new seed namespace if truth, finite-price, decomposition and
support checks hold. Formal total: 30,000 synthetic finite-panel data sets.
Report all 15 scenarios, seeds, runtime, package versions, failures, and every
outer replicate. A no-noise check must exactly recover prices and categories.
Re-run a fixed formal prefix to check exact reproducibility. Freeze this file's
hash in the manifest. Amendments after seeing results must be separately dated.
Low accuracy or coverage is a result, not an execution failure.

The design follows aims/data-generation/estimands/methods/performance reporting
as described by [Morris, White and Crowther (2019)](https://discovery.ucl.ac.uk/id/eprint/10066118/).

