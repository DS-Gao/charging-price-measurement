# Small reproducible example

From the repository root:

```sh
python run.py simulation --phase pilot --reps 8 --output ../reconstruction_example
```

This generates all 15 scenarios with eight repetitions each (120 finite panels). No external data are needed. Look for:

- `truth_checks.json`: independently checked population medians and no-noise recovery.
- `known_truth_panel.csv`: the fixed latent station panel for each process.
- `performance_summary.csv`: bias/error summaries and Monte Carlo standard errors.
- `outer_replicates.parquet`: every generated repetition, including unavailable statistics.
- `run_checks.json`: exact prefix reproduction and decomposition checks.

Eight repetitions demonstrate execution only. Use the formal command documented in the main README to reproduce the larger design. This example does not calibrate errors to Beijing or certify historical variance-share coverage.
