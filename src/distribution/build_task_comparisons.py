"""Recompute bounded same-task comparisons from the frozen extension CSVs.

This module intentionally reads only the derived extension outputs.  It does
not read raw orders, source parquet panels, or run simulations.  The input
directory and output directory are command-line parameters so the analysis can
be replayed from a portable revision package.

The three reported comparisons are:

* E-star versus T-star labels on the same A units at kappa 0.02, 0.03, and
  0.05 RMB/kWh;
* A versus C support membership using the frozen R and H flags; and
* signed H amount categories at the fixed 0.90 RMB / 30 kWh reference.

All threshold tails are strict.  A value exactly at a threshold is neutral,
and the near amount interval includes both endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = (
    ROOT
    / "outputs"
    / "09_price_architecture"
    / "focused_manuscript_revision_20260914"
    / "analysis_results"
    / "extensions"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "09_price_architecture"
    / "energy_measurement_revision_20260914"
    / "analysis_results"
    / "task_comparisons"
)

REQUIRED_INPUTS = (
    "unit_level_signed.csv",
    "summary_short.csv",
    "input_audit_3x3.csv",
    "input_audit_C.csv",
)
OPTIONAL_METADATA = (
    "analysis_metadata.json",
    "source_audit.json",
    "baseline_counts.json",
    "count_closure_checks.json",
)

SYSTEM_ORDER = ("Beijing", "UrbanEV")
LABEL_ORDER = ("negative", "neutral", "positive")
KAPPAS = (0.02, 0.03, 0.05)
PRIMARY_E_THRESHOLD = 0.01
PRIMARY_T_THRESHOLD = 0.03
REFERENCE_KWH = 30.0
PRIMARY_AMOUNT_RMB = 0.90
PROXIMITY_WINDOW = 1e-10
CHECK_TOLERANCE_RMB = 1e-9

EXPECTED_PRIMARY = {
    "Beijing": {"A": 8133, "R": 7531, "H": 7479, "C": 3916},
    "UrbanEV": {"A": 49369, "R": 4921, "H": 1377, "C": 48180},
}
EXPECTED_INPUT_PRIMARY = {
    "Beijing": {"A": 8133, "mismatch": 92, "both_neutral": 564},
    "UrbanEV": {"A": 49369, "mismatch": 3893, "both_neutral": 44093},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_clean(value: Any) -> Any:
    """Convert NumPy/pandas values to strict JSON values."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): json_clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [json_clean(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_clean(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def finite_values(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def finite_median(series: pd.Series) -> float | None:
    values = finite_values(series)
    return float(values.median()) if len(values) else None


def finite_abs_median(series: pd.Series) -> float | None:
    values = finite_values(series).abs()
    return float(values.median()) if len(values) else None


def share(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def parse_bool(series: pd.Series, column: str) -> pd.Series:
    """Parse frozen CSV boolean fields while rejecting ambiguous values."""

    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{column} contains missing boolean values")
        return series.astype(bool)
    text = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "t": True,
        "yes": True,
        "false": False,
        "0": False,
        "f": False,
        "no": False,
    }
    parsed = text.map(mapping)
    bad = text[parsed.isna()]
    if len(bad):
        raise ValueError(f"{column} contains unrecognized boolean values: {bad.unique().tolist()}")
    return parsed.astype(bool)


def strict_label(values: pd.Series, kappa: float) -> pd.Series:
    """Return negative, neutral, positive under strict tail thresholds."""

    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric < -kappa, numeric > kappa],
            ["negative", "positive"],
            default="neutral",
        ),
        index=values.index,
        dtype="object",
    )


def strict_amount_label(values: pd.Series, amount: float) -> pd.Series:
    return strict_label(values, amount)


def input_paths(input_dir: Path) -> dict[str, Path]:
    return {name: input_dir / name for name in REQUIRED_INPUTS}


def load_inputs(input_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Path]]:
    paths = input_paths(input_dir)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required frozen inputs in {input_dir}: {missing}")
    frames = {
        "units": pd.read_csv(
            paths["unit_level_signed.csv"],
            low_memory=False,
            float_precision="round_trip",
            dtype={"system": "string", "station_id": "string", "month": "string"},
        ),
        "summary": pd.read_csv(
            paths["summary_short.csv"],
            low_memory=False,
            float_precision="round_trip",
            dtype={"system": "string"},
        ),
        "source_audit_3x3": pd.read_csv(
            paths["input_audit_3x3.csv"],
            low_memory=False,
            float_precision="round_trip",
            dtype={"system": "string", "E_label": "string", "T_label": "string"},
        ),
        "source_audit_C": pd.read_csv(
            paths["input_audit_C.csv"],
            low_memory=False,
            float_precision="round_trip",
            dtype={"system": "string", "set_label": "string"},
        ),
    }
    metadata: dict[str, Any] = {}
    for name in OPTIONAL_METADATA:
        path = input_dir / name
        if path.is_file():
            metadata[name] = json.loads(path.read_text(encoding="utf-8"))
    return frames, metadata, paths


def prepare_units(units: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "system",
        "station_id",
        "month",
        "boundary",
        "boundary_value",
        "source_row",
        "A",
        "R",
        "H",
        "C",
        "E_star",
        "T_star",
        "D30",
        "D30_abs",
        "total_category",
        "signed_category",
    }
    missing = sorted(required - set(units.columns))
    if missing:
        raise ValueError(f"unit_level_signed.csv is missing columns: {missing}")
    result = units.copy()
    result["system"] = result["system"].astype("string")
    unknown_systems = sorted(set(result["system"].dropna()) - set(SYSTEM_ORDER))
    if unknown_systems:
        raise ValueError(f"Unexpected systems in unit panel: {unknown_systems}")
    if result["system"].isna().any():
        raise ValueError("unit panel contains missing system values")
    for flag in ("A", "R", "H", "C"):
        result[flag] = parse_bool(result[flag], flag)
    for column in ("E_star", "T_star", "D30", "D30_abs"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    key_columns = ["system", "station_id", "month", "boundary"]
    duplicate_keys = int(result.duplicated(key_columns).sum())
    if duplicate_keys:
        raise ValueError(f"unit panel has {duplicate_keys} duplicate unit keys")
    subset_checks = {
        "R_subset_A": bool((~result["R"] | result["A"]).all()),
        "H_subset_R": bool((~result["H"] | result["R"]).all()),
        "C_subset_A": bool((~result["C"] | result["A"]).all()),
        "duplicate_unit_keys": duplicate_keys,
        "systems": [str(x) for x in sorted(result["system"].unique())],
    }
    if not all(subset_checks[key] for key in ("R_subset_A", "H_subset_R", "C_subset_A")):
        raise ValueError(f"Frozen membership subset check failed: {subset_checks}")
    return result, subset_checks


def primary_counts(units: pd.DataFrame) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for system in SYSTEM_ORDER:
        group = units.loc[units["system"].eq(system)]
        a = group.loc[group["A"]]
        output[system] = {
            "A": int(len(a)),
            "R": int(a["R"].sum()),
            "H": int(a["H"].sum()),
            "C": int(a["C"].sum()),
        }
    return output


def build_label_audit(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    for kappa in KAPPAS:
        for system in SYSTEM_ORDER:
            group = units.loc[units["system"].eq(system) & units["A"]].copy()
            group["E_label"] = strict_label(group["E_star"], kappa)
            group["T_label"] = strict_label(group["T_star"], kappa)
            if group[["E_star", "T_star"]].isna().any().any():
                raise ValueError(f"Nonfinite E_star/T_star values on A for {system}")
            a_n = int(len(group))
            mismatch = group["E_label"] != group["T_label"]
            both_neutral = group["E_label"].eq("neutral") & group["T_label"].eq("neutral")
            either_nonneutral = ~both_neutral
            mismatch_union = mismatch & either_nonneutral
            for e_label in LABEL_ORDER:
                for t_label in LABEL_ORDER:
                    cell = group.loc[
                        group["E_label"].eq(e_label) & group["T_label"].eq(t_label)
                    ]
                    matrix_rows.append(
                        {
                            "system": system,
                            "kappa_RMB_per_kWh": kappa,
                            "E_label": e_label,
                            "T_label": t_label,
                            "n": int(len(cell)),
                            "share_of_A": share(len(cell), a_n),
                            "D30_signed_median_RMB": finite_median(cell["D30"]),
                            "D30_absolute_median_RMB": finite_abs_median(cell["D30"]),
                        }
                    )
            summary_rows.append(
                {
                    "system": system,
                    "kappa_RMB_per_kWh": kappa,
                    "A_n": a_n,
                    "mismatch_n": int(mismatch.sum()),
                    "mismatch_share_of_A": share(int(mismatch.sum()), a_n),
                    "both_neutral_n": int(both_neutral.sum()),
                    "both_neutral_share_of_A": share(int(both_neutral.sum()), a_n),
                    "either_nonneutral_n": int(either_nonneutral.sum()),
                    "either_nonneutral_share_of_A": share(int(either_nonneutral.sum()), a_n),
                    "mismatch_within_either_nonneutral_n": int(mismatch_union.sum()),
                    "mismatch_within_either_nonneutral_share": share(
                        int(mismatch_union.sum()), int(either_nonneutral.sum())
                    ),
                    "union_denominator_definition": "E or T nonneutral",
                    "union_interpretation": "outcome-selected descriptive; not independent primary target",
                }
            )
            for component, labels in (("E", group["E_label"]), ("T", group["T_label"])):
                for label in LABEL_ORDER:
                    n = int(labels.eq(label).sum())
                    margin_rows.append(
                        {
                            "system": system,
                            "kappa_RMB_per_kWh": kappa,
                            "component": component,
                            "label": label,
                            "n": n,
                            "share_of_A": share(n, a_n),
                        }
                    )
    matrix = pd.DataFrame(matrix_rows)
    summary = pd.DataFrame(summary_rows)
    margins = pd.DataFrame(margin_rows)
    return matrix, summary, margins


def build_threshold_proximity(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    thresholds = {
        "E_star": sorted(set(KAPPAS + (PRIMARY_E_THRESHOLD,))),
        "T_star": sorted(set(KAPPAS + (PRIMARY_T_THRESHOLD,))),
    }
    for system in SYSTEM_ORDER:
        group = units.loc[units["system"].eq(system) & units["A"]]
        for component, values in (("E_star", group["E_star"]), ("T_star", group["T_star"])):
            numeric = pd.to_numeric(values, errors="coerce")
            for threshold in thresholds[component]:
                for side, signed_threshold in (("negative", -threshold), ("positive", threshold)):
                    distances = (numeric - signed_threshold).abs()
                    near = distances <= PROXIMITY_WINDOW
                    exact = numeric == signed_threshold
                    rows.append(
                        {
                            "system": system,
                            "scope": "A",
                            "component": component,
                            "side": side,
                            "threshold_RMB_per_kWh": threshold,
                            "window_RMB_per_kWh": PROXIMITY_WINDOW,
                            "n_within_window": int(near.sum()),
                            "n_exact_serialized": int(exact.sum()),
                            "minimum_absolute_distance_RMB_per_kWh": float(distances.min()),
                        }
                    )
    return pd.DataFrame(rows)


def build_reconciliation(
    units: pd.DataFrame,
    source_audit_3x3: pd.DataFrame,
    source_audit_c: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    primary_label_checks: dict[str, Any] = {}
    for system in SYSTEM_ORDER:
        group = units.loc[units["system"].eq(system)]
        a = group.loc[group["A"]]
        r = a.loc[a["R"]]
        h = a.loc[a["H"]]
        r_recomputed = a["E_star"] > PRIMARY_E_THRESHOLD
        h_recomputed = r["T_star"] > PRIMARY_T_THRESHOLD
        amount_recomputed = strict_amount_label(h["D30"], PRIMARY_AMOUNT_RMB).replace(
            {"neutral": "near"}
        )
        source_amount = h["signed_category"].astype("string")
        total_recomputed = strict_label(a["T_star"], PRIMARY_T_THRESHOLD).map(
            {"positive": "retained", "neutral": "neutralized", "negative": "reversed"}
        )
        source_total = a["total_category"].astype("string")
        for check_name, scope, source, recomputed, denominator in (
            ("R_membership", "A", a["R"], r_recomputed, len(a)),
            ("H_membership", "R", r["H"], h_recomputed, len(r)),
            ("total_category", "A", source_total, total_recomputed, len(a)),
            ("H_signed_amount_category", "H", source_amount, amount_recomputed, len(h)),
        ):
            mismatch_n = int((source.astype("string") != pd.Series(recomputed, index=source.index).astype("string")).sum())
            rows.append(
                {
                    "system": system,
                    "check": check_name,
                    "scope": scope,
                    "n_scope": int(denominator),
                    "mismatch_n": mismatch_n,
                    "match": bool(mismatch_n == 0),
                    "frozen_or_source": "frozen membership/category",
                    "recomputed_rule": {
                        "R_membership": "A and E_star > 0.01",
                        "H_membership": "R and T_star > 0.03",
                        "total_category": "T_star < -0.03 / within +/-0.03 / > 0.03",
                        "H_signed_amount_category": "D30 < -0.90 / within +/-0.90 / > 0.90",
                    }[check_name],
                }
            )
        d30_formula = REFERENCE_KWH * (a["T_star"] - a["E_star"])
        formula_error = (a["D30"] - d30_formula).abs()
        abs_error = (a["D30_abs"] - a["D30"].abs()).abs()
        common = a.loc[a["C"]]
        common_formula_error = (common["D30"] - REFERENCE_KWH * (common["T_star"] - common["E_star"])).abs()
        primary_label_checks[system] = {
            "R_frozen_vs_E_gt_0_01_mismatch_n": int((a["R"] != r_recomputed).sum()),
            "H_frozen_vs_R_and_T_gt_0_03_mismatch_n": int((r["H"] != h_recomputed).sum()),
            "source_total_category_vs_T_threshold_mismatch_n": int((source_total != total_recomputed).sum()),
            "source_signed_category_vs_D30_threshold_mismatch_n": int((source_amount != amount_recomputed.astype("string")).sum()),
            "max_abs_D30_formula_error_RMB": float(formula_error.max()),
            "max_abs_D30_abs_error_RMB": float(abs_error.max()),
            "common_units_n": int(len(common)),
            "common_unit_point_estimator_max_abs_difference_RMB": float(common_formula_error.max()) if len(common) else None,
            "common_unit_point_estimator_unchanged": bool(
                len(common) == 0 or float(common_formula_error.max()) <= CHECK_TOLERANCE_RMB
            ),
        }

        # The source C audit includes an explicit membership-only row.  Check
        # it when present, while keeping the frozen unit panel authoritative.
        source_c_check = source_audit_c.loc[
            source_audit_c["system"].eq(system)
            & source_audit_c["set_label"].eq("C_membership_only_check")
        ]
        if len(source_c_check) != 1:
            checks.setdefault("source_C_membership_rows", {})[system] = {
                "rows_found": int(len(source_c_check)),
                "match": False,
            }
        else:
            check_row = source_c_check.iloc[0]
            values = {
                column: float(pd.to_numeric(check_row[column], errors="coerce"))
                for column in (
                    "max_abs_E_difference_after_membership_only",
                    "max_abs_T_difference_after_membership_only",
                    "max_abs_D30_difference_after_membership_only",
                )
            }
            checks.setdefault("source_C_membership_rows", {})[system] = {
                "rows_found": 1,
                "values": values,
                "match": all(value <= CHECK_TOLERANCE_RMB for value in values.values()),
            }
    checks["primary_label_reconciliation"] = primary_label_checks
    return pd.DataFrame(rows), checks


def build_eligibility_comparison(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        group = units.loc[units["system"].eq(system)]
        # R and H membership below deliberately use the frozen flags in the
        # unit-level CSV.  The E/T threshold rules are reconciled separately.
        a = group.loc[group["A"]]
        c = a.loc[a["C"]]
        a_r = a.loc[a["R"]]
        c_r = c.loc[c["R"]]
        a_h = a.loc[a["H"]]
        c_h = c.loc[c["H"]]
        rows.append(
            {
                "system": system,
                "A_n": int(len(a)),
                "C_n": int(len(c)),
                "lost_A_n": int(len(a) - len(c)),
                "C_share_of_A": share(len(c), len(a)),
                "lost_A_share_of_A": share(len(a) - len(c), len(a)),
                "A_R_n": int(len(a_r)),
                "C_R_n": int(len(c_r)),
                "lost_R_n": int(len(a_r) - len(c_r)),
                "lost_R_share_of_A_R": share(len(a_r) - len(c_r), len(a_r)),
                "A_H_n": int(len(a_h)),
                "C_H_n": int(len(c_h)),
                "lost_H_n": int(len(a_h) - len(c_h)),
                "lost_H_share_of_A_H": share(len(a_h) - len(c_h), len(a_h)),
                "A_H_over_R": share(len(a_h), len(a_r)),
                "C_H_over_R": share(len(c_h), len(c_r)),
                "A_H_D30_signed_median_RMB": finite_median(a_h["D30"]),
                "C_H_D30_signed_median_RMB": finite_median(c_h["D30"]),
                "A_H_D30_absolute_median_RMB": finite_abs_median(a_h["D30"]),
                "C_H_D30_absolute_median_RMB": finite_abs_median(c_h["D30"]),
                "A_H_D30_finite_n": int(len(finite_values(a_h["D30"]))),
                "C_H_D30_finite_n": int(len(finite_values(c_h["D30"]))),
            }
        )
        # Every C row is an A row, so this compares the same serialized unit
        # estimate with its formula identity and records the common-unit check.
        common_d30 = c["D30"] - REFERENCE_KWH * (c["T_star"] - c["E_star"])
        common_abs = c["D30_abs"] - c["D30"].abs()
        common_rows.append(
            {
                "system": system,
                "common_units_n": int(len(c)),
                "common_units_D30_finite_n": int(len(finite_values(c["D30"]))),
                "common_units_D30_abs_finite_n": int(len(finite_values(c["D30_abs"]))),
                "common_unit_point_estimator_max_abs_difference_RMB": float(common_d30.abs().max()) if len(c) else None,
                "common_unit_absolute_estimator_max_abs_difference_RMB": float(common_abs.abs().max()) if len(c) else None,
                "common_unit_point_estimator_unchanged": bool(
                    len(c) == 0 or float(common_d30.abs().max()) <= CHECK_TOLERANCE_RMB
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(common_rows)


def build_h_amount(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        h = units.loc[units["system"].eq(system) & units["A"] & units["H"]]
        d30 = finite_values(h["D30"])
        labels = strict_amount_label(d30, PRIMARY_AMOUNT_RMB)
        counts = {label: int(labels.eq(label).sum()) for label in LABEL_ORDER}
        abs_gt = int((d30.abs() > PRIMARY_AMOUNT_RMB).sum())
        rows.append(
            {
                "system": system,
                "H_n": int(len(h)),
                "D30_finite_n": int(len(d30)),
                "negative_n": counts["negative"],
                "near_n": counts["neutral"],
                "positive_n": counts["positive"],
                "negative_share_of_H": share(counts["negative"], len(h)),
                "near_share_of_H": share(counts["neutral"], len(h)),
                "positive_share_of_H": share(counts["positive"], len(h)),
                "abs_gt_0_90_n": abs_gt,
                "abs_gt_0_90_share_of_H": share(abs_gt, len(h)),
                "abs_gt_equals_negative_plus_positive": bool(
                    abs_gt == counts["negative"] + counts["positive"]
                ),
                "D30_signed_median_RMB": finite_median(d30),
                "D30_absolute_median_RMB": finite_abs_median(d30),
            }
        )
    return pd.DataFrame(rows)


def make_figure(label_matrix: pd.DataFrame, label_summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = label_matrix.loc[label_matrix["kappa_RMB_per_kWh"].eq(0.03)].copy()
    primary_summary = label_summary.loc[label_summary["kappa_RMB_per_kWh"].eq(0.03)].copy()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8), constrained_layout=False)
    heatmap = plt.get_cmap("Blues")
    for column, system in enumerate(SYSTEM_ORDER):
        cells = primary.loc[primary["system"].eq(system)]
        pivot = (
            cells.pivot(index="E_label", columns="T_label", values="n")
            .reindex(index=LABEL_ORDER, columns=LABEL_ORDER)
            .fillna(0)
        )
        values = pivot.to_numpy(dtype=float)
        ax = axes[0, column]
        vmax = max(float(values.max()), 1.0)
        ax.imshow(values, cmap=heatmap, vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(np.arange(3), ["negative", "neutral", "positive"])
        ax.set_yticks(np.arange(3), ["negative", "neutral", "positive"])
        ax.set_xlabel("T label")
        ax.set_ylabel("E label")
        ax.set_title(f"{chr(97 + column)}) {system}: common A labels", loc="left", fontweight="bold")
        for row_index, e_label in enumerate(LABEL_ORDER):
            for col_index, t_label in enumerate(LABEL_ORDER):
                cell = cells.loc[
                    cells["E_label"].eq(e_label) & cells["T_label"].eq(t_label)
                ].iloc[0]
                text_color = "white" if values[row_index, col_index] > vmax * 0.52 else "#1f2937"
                ax.text(
                    col_index,
                    row_index,
                    f"{int(cell['n']):,}\n{100 * float(cell['share_of_A']):.2f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8.5,
                )
        summary = primary_summary.loc[primary_summary["system"].eq(system)].iloc[0]
        ax.text(
            0.0,
            -0.22,
            f"A={int(summary['A_n']):,}; mismatch={int(summary['mismatch_n']):,}; "
            f"both neutral={int(summary['both_neutral_n']):,}",
            transform=ax.transAxes,
            fontsize=8.3,
        )
        ax.spines[:].set_visible(False)

        # Each lower panel contains only one system's denominators.  The
        # union is reported as a descriptive denominator, not an estimand.
        labels = ["E nonneutral", "T nonneutral", "either nonneutral", "mismatch"]
        counts = [
            int((cells.loc[cells["E_label"] != "neutral", "n"]).sum()),
            int((cells.loc[cells["T_label"] != "neutral", "n"]).sum()),
            int(summary["either_nonneutral_n"]),
            int(summary["mismatch_n"]),
        ]
        ax_bar = axes[1, column]
        positions = np.arange(len(labels))
        bars = ax_bar.bar(positions, counts, color=["#4C78A8", "#F2A541", "#72B7B2", "#B279A2"], width=0.66)
        ax_bar.set_xticks(positions, labels, rotation=20, ha="right")
        ax_bar.set_ylabel("Count of A units")
        ax_bar.set_title(f"{chr(99 + column)}) {system}: denominators", loc="left", fontweight="bold")
        ax_bar.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        ax_bar.set_axisbelow(True)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_visible(False)
        y_max = max(counts) * 1.18 if max(counts) else 1.0
        ax_bar.set_ylim(0, y_max)
        for bar, count in zip(bars, counts):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.02,
                f"{count:,}\n({100 * count / int(summary['A_n']):.2f}%)",
                ha="center",
                va="bottom",
                fontsize=8.2,
            )
        ax_bar.text(0.0, -0.30,
                    f"Mismatch within either-nonneutral union: {100 * float(summary['mismatch_within_either_nonneutral_share']):.2f}%",
                    transform=ax_bar.transAxes, fontsize=8.2)
    fig.suptitle(
        "Electricity-only versus total-price input labels on common supported units",
        fontsize=13,
        fontweight="bold",
        x=0.05,
        ha="left",
    )
    fig.text(
        0.05,
        0.015,
        "Primary kappa = 0.03 RMB/kWh. Heatmap cells show n and % of A; tails are strict and neutral includes both endpoints. "
        "The either-nonneutral union is an outcome-selected descriptive denominator.",
        fontsize=8.2,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.16, wspace=0.28, hspace=0.62)
    pdf_path = figure_dir / "Figure_energy_input_audit.pdf"
    png_path = figure_dir / "Figure_energy_input_audit.png"
    pdf_metadata = {
        "Title": "Electricity-only versus total-price input labels on common supported units",
        "Author": "Energy measurement revision",
        "Creator": "build_task_comparisons.py",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, bbox_inches="tight", metadata=pdf_metadata)
    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=240,
        metadata={"Software": "build_task_comparisons.py", "Title": "Energy input audit"},
    )
    plt.close(fig)
    return [pdf_path, png_path]


def tex_number(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def tex_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{int(value):,}".replace(",", "{,}")


def tex_pct(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{100 * float(value):.{digits}f}\\%"


def build_input_table(label_summary: pd.DataFrame, label_matrix: pd.DataFrame) -> str:
    summary_lines: list[str] = []
    for row in label_summary.to_dict("records"):
        summary_lines.append(
            " & ".join(
                [
                    str(row["system"]),
                    tex_number(row["kappa_RMB_per_kWh"], 2),
                    tex_int(row["A_n"]),
                    tex_int(row["mismatch_n"]),
                    tex_pct(row["mismatch_share_of_A"], 2),
                    tex_int(row["both_neutral_n"]),
                    tex_int(row["either_nonneutral_n"]),
                    tex_int(row["mismatch_within_either_nonneutral_n"]),
                    tex_pct(row["mismatch_within_either_nonneutral_share"], 2),
                ]
            )
            + r" \\",
        )
    primary = label_matrix.loc[label_matrix["kappa_RMB_per_kWh"].eq(0.03)]
    cells: list[str] = []
    short_labels = {"negative": "neg", "neutral": "zero", "positive": "pos"}
    for system in SYSTEM_ORDER:
        group = primary.loc[primary["system"].eq(system)]
        tokens = []
        for e_label in LABEL_ORDER:
            for t_label in LABEL_ORDER:
                row = group.loc[group["E_label"].eq(e_label) & group["T_label"].eq(t_label)].iloc[0]
                tokens.append(
                    f"{short_labels[e_label]}/{short_labels[t_label]}="
                    f"{tex_int(row['n'])} ({tex_pct(row['share_of_A'], 1)})"
                )
        cells.append(f"{system}: " + "; ".join(tokens) + r"\\")
    return (
        "\\begin{table}[p]\n"
        "\\centering\n"
        "\\caption{Same-task electricity-only versus total-price input audit on common supported $A$ units.}\n"
        "\\label{tab:energy-input-denominators}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}lrrrrrrrr@{}}\n"
        "\\toprule\n"
        "System & $\\kappa$ & $A$ & Mismatch & \\shortstack{Share\\\\of $A$} & \\shortstack{Both\\\\neutral} & $V_\\kappa$ & \\shortstack{Mismatch\\\\in $V_\\kappa$} & \\shortstack{Share in\\\\$V_\\kappa$}\\\\\n"
        "\\midrule\n"
        + "\n".join(summary_lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\vspace{3pt}\n"
        "\\begin{minipage}{0.98\\textwidth}\\footnotesize\n"
        "Labels use negative $< -\\kappa$, neutral $[-\\kappa,\\kappa]$, and positive $>\\kappa$; all rows use the same $A$ denominator within system. "
        "The either-nonneutral union is an outcome-selected descriptive denominator, not an independent primary target. "
        "The complete primary cross-tabulations are shown in Figure~\\ref{fig:energy-input-audit}; all cells for every tolerance are supplied in the analytical CSV. $\\kappa$ is in RMB/kWh.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )


def build_support_table(eligibility: pd.DataFrame) -> str:
    count_lines: list[str] = []
    metric_lines: list[str] = []
    for row in eligibility.to_dict("records"):
        count_lines.append(
            " & ".join(
                [
                    str(row["system"]),
                    tex_int(row["A_n"]),
                    tex_int(row["C_n"]),
                    tex_int(row["lost_A_n"]),
                    tex_int(row["A_R_n"]),
                    tex_int(row["C_R_n"]),
                    tex_int(row["lost_R_n"]),
                    tex_int(row["A_H_n"]),
                    tex_int(row["C_H_n"]),
                    tex_int(row["lost_H_n"]),
                ]
            )
            + r" \\",
        )
        metric_lines.append(
            " & ".join(
                [
                    str(row["system"]),
                    tex_pct(row["A_H_over_R"], 2),
                    tex_pct(row["C_H_over_R"], 2),
                    tex_number(row["A_H_D30_signed_median_RMB"], 4),
                    tex_number(row["C_H_D30_signed_median_RMB"], 4),
                    tex_number(row["A_H_D30_absolute_median_RMB"], 2),
                    tex_number(row["C_H_D30_absolute_median_RMB"], 2),
                ]
            )
            + r" \\",
        )
    return (
        "\\begin{table}[p]\n"
        "\\centering\n"
        "\\caption{Support-membership comparison using frozen primary $R$ and $H$ flags.}\n"
        "\\label{tab:energy-support-comparison}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}lrrrrrrrrr@{}}\n"
        "\\toprule\n"
        "System & $A$ & $C$ & Lost $A$ & $R_A$ & $R_C$ & Lost $R$ & $H_A$ & $H_C$ & Lost $H$\\\\\n"
        "\\midrule\n"
        + "\n".join(count_lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\vspace{5pt}\n"
        "\\begin{tabular}{@{}lrrrrrr@{}}\n"
        "\\toprule\n"
        "System & \\shortstack{$H/R$\\\\in $A$} & \\shortstack{$H/R$\\\\in $C$} & \\shortstack{Signed\\\\median on $H$} & \\shortstack{Signed median\\\\on $H\\cap C$} & \\shortstack{Absolute\\\\median on $H$} & \\shortstack{Absolute median\\\\on $H\\cap C$}\\\\\n"
        "\\midrule\n"
        + "\n".join(metric_lines)
        + "\n\\bottomrule\n\\end{tabular}\n"
        "\\vspace{3pt}\n"
        "\\begin{minipage}{0.98\\textwidth}\\footnotesize\n"
        "$R$ and $H$ use the frozen primary membership fields ($E^*>0.01$ and $T^*>0.03$ were checked only for reconciliation). $C$ is a subset of $A$. "
        "All medians describe $D_{30}$ on $H$ or $H\\cap C$, not on all $A$ or $C$ units. Units are RMB per reference 30 kWh; all contributing values are finite. Each common unit retains its original point estimate; changes in aggregate medians arise from membership. "
        "The two systems are summarized independently.\n"
        "\\end{minipage}\n"
        "\\end{table}\n"
    )


def format_near_threshold_report(proximity: pd.DataFrame) -> str:
    nonzero = proximity.loc[proximity["n_within_window"] > 0]
    if nonzero.empty:
        return "No A units were within the configured threshold proximity window."
    lines = []
    for row in nonzero.to_dict("records"):
        signed = "+" if row["side"] == "positive" else "-"
        lines.append(
            f"- {row['system']} {row['component']} {signed}{row['threshold_RMB_per_kWh']:.2f}: "
            f"{int(row['n_within_window'])} within {PROXIMITY_WINDOW:.0e} "
            f"({int(row['n_exact_serialized'])} exactly equal in serialized CSV)."
        )
    return "\n".join(lines)


def build_readme(
    output_dir: Path,
    input_dir: Path,
    input_hashes: dict[str, Any],
    counts: dict[str, dict[str, int]],
    label_summary: pd.DataFrame,
    eligibility: pd.DataFrame,
    h_amount: pd.DataFrame,
    proximity: pd.DataFrame,
    reconciliation: dict[str, Any],
    artifacts: list[Path],
) -> str:
    primary_rows = label_summary.loc[label_summary["kappa_RMB_per_kWh"].eq(0.03)]
    lines = [
        "# Same-task comparison analysis",
        "",
        "This bounded analysis recomputes descriptive comparisons from the frozen "
        "extension CSVs. It reads no raw orders, source parquet panels, or new "
        "simulations. Input hashes and replay checks are recorded in "
        "`task_comparison_checks.json`.",
        "",
        "## Estimands and conventions",
        "",
        "The input audit uses the same supported population A within each system. "
        "For each kappa, `c_kappa(x)` is negative when `x < -kappa`, neutral when "
        "`-kappa <= x <= kappa`, and positive when `x > kappa`; the thresholds "
        "are in RMB/kWh. A mismatch is `E_label != T_label`. Both-neutral is the "
        "intersection of the two neutral labels. The either-nonneutral union is "
        "the complement of both-neutral, or E nonneutral OR T nonneutral. Its "
        "conditional mismatch share is an outcome-selected descriptive quantity "
        "and is not an independent primary target.",
        "",
        "The primary eligibility counts use the frozen unit-level R and H flags: "
        "R is the serialized primary electricity-qualified membership and H is "
        "the serialized primary total-direction membership within R. The script "
        "checks their corresponding E-star > 0.01 and T-star > 0.03 rules but does "
        "not replace the frozen flags when computing A/C losses or H/R. C is the "
        "frozen complete-support subset of A.",
        "",
        "For H units, `D30` is the frozen signed total-minus-electricity contrast "
        "in RMB per reference 30 kWh. The amount reference is 0.90 RMB (= 30 x "
        "0.03 RMB/kWh). Negative and positive tails are strict (`D30 < -0.90`, "
        "`D30 > 0.90`); near includes both endpoints. The absolute exceedance "
        "count is `abs(D30) > 0.90` and must equal negative plus positive.",
        "",
        "A/C loss shares use A, R, and H denominators respectively. H/R uses R as "
        "the denominator. H signed and absolute medians use finite D30 values "
        "within H. Each C unit retains its serialized point estimate; the common "
        "unit formula check verifies `D30 = 30(T_star - E_star)` to the stated "
        f"tolerance ({CHECK_TOLERANCE_RMB:.0e} RMB).",
        "",
        "## Primary count evidence",
        "",
        "The expected and recomputed counts are:",
        "",
        "| System | A | R | H | C |",
        "|---|---:|---:|---:|---:|",
    ]
    for system in SYSTEM_ORDER:
        row = counts[system]
        lines.append(f"| {system} | {row['A']:,} | {row['R']:,} | {row['H']:,} | {row['C']:,} |")
    lines.extend(
        [
            "",
            "At primary kappa = 0.03, the label audit gives:",
            "",
            "| System | A | mismatch | both neutral | either nonneutral | mismatch within union |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary_rows.to_dict("records"):
        lines.append(
            f"| {row['system']} | {int(row['A_n']):,} | {int(row['mismatch_n']):,} "
            f"({100 * row['mismatch_share_of_A']:.2f}%) | {int(row['both_neutral_n']):,} "
            f"({100 * row['both_neutral_share_of_A']:.2f}%) | {int(row['either_nonneutral_n']):,} "
            f"| {int(row['mismatch_within_either_nonneutral_n']):,} "
            f"({100 * row['mismatch_within_either_nonneutral_share']:.2f}%) |"
        )
    lines.extend(
        [
            "",
            "The A/C and H amount tables are in `eligibility_A_C.csv` and "
            "`H_amount_reference.csv`. Full 3 x 3 cells for all three kappas are "
            "in `input_audit_3x3.csv`; component margins are in "
            "`input_audit_margins.csv`.",
            "",
            "## Threshold precision and reconciliation",
            "",
            "The input CSV is a serialized frozen draw. Threshold labels are applied "
            "to the parsed serialized values using the strict convention above. If "
            "a future export changes rounding, observations exactly at a threshold "
            "could change labels; the source SHA-256 hashes preserve the replayed "
            "version. `threshold_proximity.csv` reports counts within "
            f"{PROXIMITY_WINDOW:.0e} RMB/kWh of both signed sides of every primary "
            "and label-sensitivity threshold.",
            "",
            format_near_threshold_report(proximity),
            "",
            "`label_reconciliation.csv` records exact frozen R/H and source-category "
            "agreement. The checks also verify finite D30 values on H and exact "
            "negative/near/positive plus absolute count closure at 0.90 RMB.",
            "",
            "## Replay",
            "",
            "```powershell",
            "python `",
            "  workflows/distribution/src/distribution/build_task_comparisons.py `",
            f"  --input-dir \"{input_dir}\" --output-dir \"{output_dir}\"",
            "```",
            "",
            "## Artifacts",
            "",
        ]
    )
    for artifact in artifacts:
        lines.append(f"- `{artifact.relative_to(output_dir).as_posix()}`")
    lines.extend(
        [
            "",
            "The figure has one heatmap and one denominator panel per system. It is "
            "intended as a within-system measurement audit and makes no causal "
            "comparison between Beijing and UrbanEV.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    frames, metadata, paths = load_inputs(input_dir)
    units, subset_checks = prepare_units(frames["units"])
    counts = primary_counts(units)
    if counts != EXPECTED_PRIMARY:
        raise AssertionError(f"Primary count mismatch: expected={EXPECTED_PRIMARY}, computed={counts}")

    label_matrix, label_summary, margins = build_label_audit(units)
    eligibility, common_checks = build_eligibility_comparison(units)
    h_amount = build_h_amount(units)
    proximity = build_threshold_proximity(units)
    reconciliation_frame, reconciliation_checks = build_reconciliation(
        units, frames["source_audit_3x3"], frames["source_audit_C"]
    )

    # Validate the primary labels against the supplied prior audit and the
    # count contract. The prior audit is a cross-check, not a source for the
    # recomputed cells.
    primary_summary = label_summary.loc[label_summary["kappa_RMB_per_kWh"].eq(0.03)]
    expected_input = {
        row["system"]: {
            "A": int(row["A_n"]),
            "mismatch": int(row["mismatch_n"]),
            "both_neutral": int(row["both_neutral_n"]),
        }
        for row in primary_summary.to_dict("records")
    }
    if expected_input != EXPECTED_INPUT_PRIMARY:
        raise AssertionError(
            f"Primary input audit mismatch: expected={EXPECTED_INPUT_PRIMARY}, computed={expected_input}"
        )
    source_3x3 = frames["source_audit_3x3"].copy()
    source_3x3["kappa_RMB_per_kWh"] = pd.to_numeric(source_3x3["kappa_RMB_per_kWh"], errors="coerce")
    source_3x3["n"] = pd.to_numeric(source_3x3["n"], errors="coerce")
    merged_3x3 = label_matrix.loc[label_matrix["kappa_RMB_per_kWh"].eq(0.03)].merge(
        source_3x3[["system", "E_label", "T_label", "kappa_RMB_per_kWh", "n"]],
        on=["system", "E_label", "T_label", "kappa_RMB_per_kWh"],
        how="outer",
        suffixes=("_recomputed", "_source"),
        indicator=True,
    )
    source_3x3_match = bool(
        len(merged_3x3) == 18
        and merged_3x3["_merge"].eq("both").all()
        and (merged_3x3["n_recomputed"] == merged_3x3["n_source"]).all()
    )
    if not source_3x3_match:
        raise AssertionError("Primary 3 x 3 input audit does not reconcile to source input_audit_3x3.csv")

    source_summary = frames["summary"].copy()
    source_summary["system"] = source_summary["system"].astype("string")
    source_summary_counts = {
        row["system"]: {key: int(row[f"{key}_n"]) for key in ("A", "R", "H", "C")}
        for row in source_summary.to_dict("records")
    }
    source_summary_match = source_summary_counts == EXPECTED_PRIMARY
    if not source_summary_match:
        raise AssertionError(f"summary_short.csv count mismatch: {source_summary_counts}")

    source_c = frames["source_audit_C"].copy()
    source_c["n"] = pd.to_numeric(source_c["n"], errors="coerce")
    source_c_counts = {
        (row["system"], row["set_label"]): int(row["n"])
        for row in source_c.to_dict("records")
        if row["set_label"] in {"A", "C"}
    }
    c_source_match = all(
        source_c_counts.get((system, set_label)) == counts[system][set_label]
        for system in SYSTEM_ORDER
        for set_label in ("A", "C")
    )
    if not c_source_match:
        raise AssertionError(f"input_audit_C.csv membership mismatch: {source_c_counts}")

    h_amount_checks: dict[str, Any] = {}
    source_summary_by_system = source_summary.set_index("system")
    for row in h_amount.to_dict("records"):
        system = row["system"]
        source = source_summary_by_system.loc[system]
        source_values = {
            "negative_n": int(source["H_negative_n"]),
            "near_n": int(source["H_near_n"]),
            "positive_n": int(source["H_positive_n"]),
            "abs_gt_0_90_n": int(source["H_abs_gt_amount_n"]),
        }
        computed_values = {key: int(row[key]) for key in source_values}
        source_category_match = bool(computed_values == source_values)
        if not source_category_match:
            raise AssertionError(f"H amount category mismatch for {system}: {source_values} vs {computed_values}")
        h_amount_checks[system] = {
            "source_summary_match": source_category_match,
            "computed": computed_values,
            "closure": bool(row["abs_gt_equals_negative_plus_positive"]),
        }

    all_a = units.loc[units["A"]]
    finite_checks = {
        "A_rows": int(len(all_a)),
        "A_E_star_finite_n": int(len(finite_values(all_a["E_star"]))),
        "A_T_star_finite_n": int(len(finite_values(all_a["T_star"]))),
        "A_D30_finite_n": int(len(finite_values(all_a["D30"]))),
        "H_D30_finite_by_system": {
            system: int(len(finite_values(units.loc[units["system"].eq(system) & units["A"] & units["H"], "D30"])))
            for system in SYSTEM_ORDER
        },
    }
    finite_ok = (
        finite_checks["A_E_star_finite_n"] == finite_checks["A_rows"]
        and finite_checks["A_T_star_finite_n"] == finite_checks["A_rows"]
        and finite_checks["A_D30_finite_n"] == finite_checks["A_rows"]
        and all(finite_checks["H_D30_finite_by_system"][system] == counts[system]["H"] for system in SYSTEM_ORDER)
    )
    if not finite_ok:
        raise AssertionError(f"Nonfinite values on required A/H units: {finite_checks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "input_audit_3x3.csv", label_matrix)
    write_csv(output_dir / "input_audit_summary.csv", label_summary)
    write_csv(output_dir / "input_audit_margins.csv", margins)
    write_csv(output_dir / "eligibility_A_C.csv", eligibility)
    write_csv(output_dir / "common_unit_estimator_checks.csv", common_checks)
    write_csv(output_dir / "H_amount_reference.csv", h_amount)
    write_csv(output_dir / "threshold_proximity.csv", proximity)
    write_csv(output_dir / "label_reconciliation.csv", reconciliation_frame)
    figure_paths = make_figure(label_matrix, label_summary, output_dir)
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "energy_input_denominators.tex").write_text(
        build_input_table(label_summary, label_matrix), encoding="utf-8"
    )
    (table_dir / "energy_support_comparison.tex").write_text(
        build_support_table(eligibility), encoding="utf-8"
    )

    input_row_counts = {
        "unit_level_signed.csv": int(len(frames["units"])),
        "summary_short.csv": int(len(frames["summary"])),
        "input_audit_3x3.csv": int(len(frames["source_audit_3x3"])),
        "input_audit_C.csv": int(len(frames["source_audit_C"])),
    }
    input_hashes = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "rows": input_row_counts[name],
        }
        for name, path in paths.items()
    }
    for name in OPTIONAL_METADATA:
        path = input_dir / name
        if path.is_file():
            input_hashes[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": int(path.stat().st_size),
            }
    checks = {
        "schema": "energy-measurement-revision/task-comparisons-v1",
        "status": "PASS",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_hashes": input_hashes,
        "optional_metadata_loaded": sorted(metadata),
        "parameters": {
            "kappas_RMB_per_kWh": list(KAPPAS),
            "primary_e_threshold_RMB_per_kWh": PRIMARY_E_THRESHOLD,
            "primary_t_threshold_RMB_per_kWh": PRIMARY_T_THRESHOLD,
            "reference_kWh": REFERENCE_KWH,
            "primary_amount_threshold_RMB": PRIMARY_AMOUNT_RMB,
            "threshold_tail_convention": "negative < -threshold; neutral inclusive; positive > threshold",
            "threshold_proximity_window_RMB_per_kWh": PROXIMITY_WINDOW,
            "weighting": "equal unit within each system",
            "membership_source": "frozen R/H/C flags from unit_level_signed.csv",
            "input_union_role": "outcome-selected descriptive denominator; not independent primary target",
        },
        "expected_primary": EXPECTED_PRIMARY,
        "computed_primary": counts,
        "checks": {
            "unit_subset_checks": subset_checks,
            "summary_short_counts_match": source_summary_match,
            "input_audit_primary_expected_match": expected_input == EXPECTED_INPUT_PRIMARY,
            "input_audit_primary_source_3x3_match": source_3x3_match,
            "input_audit_C_source_match": c_source_match,
            "finite_values": finite_checks,
            "finite_values_ok": finite_ok,
            "h_amount": h_amount_checks,
            "reconciliation": reconciliation_checks,
            "common_unit_estimator": common_checks.to_dict("records"),
            "all_reconciliation_rows_match": bool(reconciliation_frame["match"].all()),
            "all_common_estimators_unchanged": bool(common_checks["common_unit_point_estimator_unchanged"].all()),
            "threshold_proximity_nonzero_rows": int((proximity["n_within_window"] > 0).sum()),
        },
    }
    if not checks["checks"]["all_reconciliation_rows_match"]:
        raise AssertionError("One or more frozen/source category reconciliations failed")
    if not checks["checks"]["all_common_estimators_unchanged"]:
        raise AssertionError("Common-unit point estimator identity failed")

    # Add stable hashes for all generated artifacts except this checks file;
    # excluding the file itself avoids a circular hash dependency.
    artifacts = [
        output_dir / name
        for name in (
            "input_audit_3x3.csv",
            "input_audit_summary.csv",
            "input_audit_margins.csv",
            "eligibility_A_C.csv",
            "common_unit_estimator_checks.csv",
            "H_amount_reference.csv",
            "threshold_proximity.csv",
            "label_reconciliation.csv",
        )
    ] + figure_paths + [
        table_dir / "energy_input_denominators.tex",
        table_dir / "energy_support_comparison.tex",
    ]
    checks["output_artifacts"] = {
        str(path.relative_to(output_dir)).replace("\\", "/"): {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in artifacts
    }
    checks_path = output_dir / "task_comparison_checks.json"
    write_json(checks_path, checks)
    artifacts_with_readme = artifacts + [checks_path]
    readme = build_readme(
        output_dir,
        input_dir,
        input_hashes,
        counts,
        label_summary,
        eligibility,
        h_amount,
        proximity,
        reconciliation_checks,
        artifacts_with_readme,
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8", newline="\n")
    checks["output_artifacts"]["README.md"] = {
        "sha256": sha256_file(readme_path),
        "bytes": int(readme_path.stat().st_size),
    }
    # The checks JSON is written one final time after README content is fixed.
    write_json(checks_path, checks)
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "primary_counts": counts,
        "primary_input_audit": expected_input,
        "figure_paths": [str(path) for path in figure_paths],
        "table_paths": [str(table_dir / "energy_input_denominators.tex"), str(table_dir / "energy_support_comparison.tex")],
        "checks_path": str(checks_path),
        "readme_path": str(readme_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.input_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
