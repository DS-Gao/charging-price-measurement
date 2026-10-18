# Inputs and units

- Beijing raw transactions (download separately): station/pile identifiers, start/end timestamps, duration, energy in kWh and electricity/service/total charges in RMB. Only the explicit raw workflow reads these files.
- `fullmonth_unit_rows.csv`: station-month-boundary derived rows with ORIGINAL/CD/DW endpoint prices, contrasts, counts, dates, support and membership flags. Prices/contrasts use RMB/kWh; amounts for reference 30 kWh use RMB.
- `module_b_row_level_inputs.csv.gz`: formation-window E, LOCAL, calibration, subsequent-window Y, observed-target/evaluation indicators, station identifiers and formation weights. CAL is E plus the layer calibration; it is not a pure calibration term.
- `frozen_original_panel.parquet`: fixed eligibility/reference panel used to check reconstructed membership, not an independently known true-price panel.
- `unit_level_signed.csv`: common schema for Beijing and UrbanEV local contrasts, A/R/H/C membership, components, signed amount D30 and related diagnostics. The observation systems remain distinct.
- `beijing_panel.parquet`, `beijing_menu.parquet`, `urbanev_panel.parquet`: derived boundary and support panels used for extensions.
- `module_b_bootstrap_draws.csv.gz`: 2,000 saved shared-station bootstrap draws of cross-window metrics.
- `inputs/e2/draws/`: 900 saved UrbanEV masking result files, not the entire raw station time series.
- `data/reference/historical/`: retained statistical panels, summary tables and saved draws, arranged by analysis topic. The reference-source index preserves source lineage.

E, S and T denote electricity, service and observed total components. D30 scales a contrast discrepancy by a reference 30 kWh; it is not a realized household bill. ST, TX and KWH distinguish station-equal, transaction and energy weights. A, R, H and C are analysis membership sets and should not be interpreted as interchangeable denominators. NA means unavailable; zero is an observed numerical value. Consult the corresponding manuscript definitions for event conditioning and population restrictions.
