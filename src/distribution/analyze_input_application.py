"""Reproduce the locked amount, screening, and concentration extension.

The command reads only the frozen unit-level input named by ``--input`` and
the locked protocol named by ``--protocol``.  It produces deterministic CSV,
figure, table, check, and README artifacts under ``--output``.

This is a descriptive observed-input sensitivity analysis.  Screening labels
are never interpreted as truth accuracy, bill savings, user choice, demand
response, or a flexible-load ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SYSTEM_ORDER = ("Beijing", "UrbanEV")
POPULATION_ORDER = ("A", "R", "H")
SCREEN_POPULATIONS = ("A", "R")
LABEL_ORDER = ("negative", "neutral", "positive")
GROUP_ORDER = ("station", "month", "boundary")
NOMINAL_FRACTIONS = (0.2, 0.1)
PRIMARY_FRACTION = 0.2
KAPPA = 0.03
ELECTRICITY_PRIMARY_THRESHOLD = 0.01
TOTAL_PRIMARY_THRESHOLD = 0.03
AMOUNT_THRESHOLD_RMB = 0.90
REFERENCE_KWH = 30.0
TIE_GRID_RMB_PER_KWH = 1e-6
EXPECTED_PRIMARY = {
    "Beijing": {"A": 8133, "R": 7531, "H": 7479, "C": 3916},
    "UrbanEV": {"A": 49369, "R": 4921, "H": 1377, "C": 48180},
}

REQUIRED_COLUMNS = {
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_clean(value: Any) -> Any:
    """Convert NumPy/pandas values into strict JSON values."""

    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): json_clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_clean(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
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
        newline="\n",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")


def parse_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise ValueError(f"{column} contains missing boolean values")
        return series.astype(bool)
    text = series.astype("string").str.strip().str.lower()
    parsed = text.map(
        {
            "true": True,
            "1": True,
            "t": True,
            "yes": True,
            "false": False,
            "0": False,
            "f": False,
            "no": False,
        }
    )
    bad = text[parsed.isna()]
    if len(bad):
        raise ValueError(f"{column} contains unrecognized boolean values: {bad.unique().tolist()}")
    return parsed.astype(bool)


def strict_label(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ValueError("A required label score contains a missing value")
    return pd.Series(
        np.select(
            [numeric < -threshold, numeric > threshold],
            ["negative", "positive"],
            default="neutral",
        ),
        index=values.index,
        dtype="object",
    )


def share(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def finite_array(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def safe_median(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(np.median(array)) if len(array) else None


def amount_statistics(values: pd.Series, include_quantiles: bool = True) -> dict[str, Any]:
    """Summarize original parsed D30 values without rounding the input."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    abs_finite = np.abs(finite)
    result: dict[str, Any] = {
        "n": int(len(numeric)),
        "finite_n": int(len(finite)),
        "missing_or_nonfinite_n": int(len(numeric) - len(finite)),
    }
    if not len(finite):
        for key in (
            "D30_signed_mean_RMB",
            "D30_signed_std_RMB",
            "D30_signed_min_RMB",
            "D30_signed_max_RMB",
            "D30_signed_median_RMB",
            "D30_absolute_mean_RMB",
            "D30_absolute_std_RMB",
            "D30_absolute_min_RMB",
            "D30_absolute_max_RMB",
            "D30_absolute_median_RMB",
        ):
            result[key] = None
        if include_quantiles:
            for prefix in ("D30_signed", "D30_absolute"):
                for qname in ("q10", "q25", "q50", "q75", "q90"):
                    result[f"{prefix}_{qname}_RMB"] = None
    else:
        result.update(
            {
                "D30_signed_mean_RMB": float(np.mean(finite)),
                "D30_signed_std_RMB": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                "D30_signed_min_RMB": float(np.min(finite)),
                "D30_signed_max_RMB": float(np.max(finite)),
                "D30_signed_median_RMB": float(np.median(finite)),
                "D30_absolute_mean_RMB": float(np.mean(abs_finite)),
                "D30_absolute_std_RMB": float(np.std(abs_finite, ddof=1)) if len(abs_finite) > 1 else 0.0,
                "D30_absolute_min_RMB": float(np.min(abs_finite)),
                "D30_absolute_max_RMB": float(np.max(abs_finite)),
                "D30_absolute_median_RMB": float(np.median(abs_finite)),
            }
        )
        if include_quantiles:
            quantiles = (("q10", 0.10), ("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q90", 0.90))
            for qname, q in quantiles:
                result[f"D30_signed_{qname}_RMB"] = float(np.quantile(finite, q, method="linear"))
                result[f"D30_absolute_{qname}_RMB"] = float(np.quantile(abs_finite, q, method="linear"))
    negative_tail = finite < -AMOUNT_THRESHOLD_RMB
    near_tail = np.abs(finite) <= AMOUNT_THRESHOLD_RMB
    positive_tail = finite > AMOUNT_THRESHOLD_RMB
    negative_signed = finite < 0
    zero_reference = finite == 0
    positive_signed = finite > 0
    result.update(
        {
            "negative_amount_n": int(negative_tail.sum()),
            "near_amount_n": int(near_tail.sum()),
            "positive_amount_n": int(positive_tail.sum()),
            "abs_gt_0_90_n": int((np.abs(finite) > AMOUNT_THRESHOLD_RMB).sum()),
            "abs_gt_0_90_share": share(int((np.abs(finite) > AMOUNT_THRESHOLD_RMB).sum()), len(finite)),
            "amount_category_closure": bool(
                int(negative_tail.sum() + near_tail.sum() + positive_tail.sum()) == len(finite)
            ),
            "negative_signed_n": int(negative_signed.sum()),
            "zero_reference_n": int(zero_reference.sum()),
            "positive_signed_n": int(positive_signed.sum()),
            "signed_reference_closure": bool(
                int(negative_signed.sum() + zero_reference.sum() + positive_signed.sum()) == len(finite)
            ),
        }
    )
    return result


def load_protocol(protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED_PLAN_SNAPSHOT":
        raise ValueError(f"Unexpected protocol status: {protocol.get('status')!r}")
    screening = protocol.get("screening", {})
    concentration = protocol.get("concentration", {})
    if screening.get("primary_population") != "A" or screening.get("secondary_population") != "R":
        raise ValueError("Protocol screening populations are not the locked A/R populations")
    if tuple(float(x) for x in screening.get("nominal_fractions", [])) != NOMINAL_FRACTIONS:
        raise ValueError("Protocol nominal screening fractions differ from locked definitions")
    if float(screening.get("tie_grid_RMB_per_kWh")) != TIE_GRID_RMB_PER_KWH:
        raise ValueError("Protocol tie grid differs from the locked 1e-6 grid")
    if tuple(screening.get("strata", [])) != ("system", "month", "boundary"):
        raise ValueError("Protocol screening strata differ from the locked fixed strata")
    if concentration.get("groups") != list(GROUP_ORDER):
        raise ValueError("Protocol concentration grouping differs from the locked groups")
    return protocol


def load_units(input_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    units = pd.read_csv(
        input_path,
        low_memory=False,
        float_precision="round_trip",
        dtype={
            "system": "string",
            "station_id": "string",
            "month": "string",
        },
    )
    missing = sorted(REQUIRED_COLUMNS - set(units.columns))
    if missing:
        raise ValueError(f"unit_level_signed.csv is missing columns: {missing}")
    units["system"] = units["system"].astype("string")
    units["station_id"] = units["station_id"].astype("string")
    units["month"] = units["month"].astype("string")
    if units["system"].isna().any() or units["station_id"].isna().any() or units["month"].isna().any():
        raise ValueError("unit panel contains missing fixed grouping keys")
    unknown_systems = sorted(set(units["system"].dropna()) - set(SYSTEM_ORDER))
    if unknown_systems:
        raise ValueError(f"Unexpected systems: {unknown_systems}")
    for flag in ("A", "R", "H", "C"):
        units[flag] = parse_bool(units[flag], flag)
    for column in ("boundary", "boundary_value", "source_row"):
        units[column] = pd.to_numeric(units[column], errors="coerce")
        if units[column].isna().any():
            raise ValueError(f"{column} contains missing/non-numeric values")
    for column in ("E_star", "T_star", "D30", "D30_abs"):
        units[column] = pd.to_numeric(units[column], errors="coerce")
    duplicates = int(units.duplicated(["system", "station_id", "month", "boundary"]).sum())
    if duplicates:
        raise ValueError(f"unit panel has {duplicates} duplicate unit keys")
    subset_checks = {
        "R_subset_A": bool((~units["R"] | units["A"]).all()),
        "H_subset_R": bool((~units["H"] | units["R"]).all()),
        "C_subset_A": bool((~units["C"] | units["A"]).all()),
        "duplicate_unit_keys": duplicates,
        "rows_total": int(len(units)),
    }
    if not all(subset_checks[name] for name in ("R_subset_A", "H_subset_R", "C_subset_A")):
        raise ValueError(f"Frozen membership subset check failed: {subset_checks}")
    return units, subset_checks


def primary_counts(units: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for system in SYSTEM_ORDER:
        frame = units.loc[units["system"].eq(system)]
        result[system] = {
            population: int(frame[population].sum()) for population in ("A", "R", "H", "C")
        }
    return result


def check_source_categories(units: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for system in SYSTEM_ORDER:
        frame = units.loc[units["system"].eq(system) & units["A"]].copy()
        d30 = frame["D30"]
        t_label = strict_label(frame["T_star"], TOTAL_PRIMARY_THRESHOLD).map(
            {"negative": "reversed", "neutral": "neutralized", "positive": "retained"}
        )
        source_total = frame["total_category"].astype("string")
        source_total_mismatch = int((source_total != t_label.astype("string")).sum())
        h = frame.loc[frame["H"]]
        amount_label = strict_label(h["D30"], AMOUNT_THRESHOLD_RMB).map(
            {"negative": "negative", "neutral": "near", "positive": "positive"}
        )
        source_amount_mismatch = int(
            (h["signed_category"].astype("string") != amount_label.astype("string")).sum()
        )
        rows[system] = {
            "total_category_mismatch_n": source_total_mismatch,
            "signed_category_mismatch_n": source_amount_mismatch,
            "D30_formula_max_abs_error_RMB": float(
                np.nanmax(
                    np.abs(
                        frame["D30"].to_numpy()
                        - REFERENCE_KWH * (frame["T_star"] - frame["E_star"])
                    )
                )
            ),
            "D30_abs_formula_max_abs_error_RMB": float(
                np.nanmax(np.abs(frame["D30_abs"].to_numpy() - np.abs(frame["D30"].to_numpy())))
            ),
        }
    return rows


def make_amount_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        base = units.loc[units["system"].eq(system)]
        for population in POPULATION_ORDER:
            frame = base.loc[base[population]].copy()
            row: dict[str, Any] = {
                "scope": "system",
                "system": system,
                "population": population,
                "population_definition": f"serialized {population}=True membership within system",
            }
            row.update(amount_statistics(frame["D30"]))
            rows.append(row)
    return pd.DataFrame(rows)


def fixed_layer_keys(units: pd.DataFrame) -> list[tuple[str, str, int, int]]:
    base = units.loc[units["A"], ["system", "month", "boundary", "boundary_value"]].drop_duplicates()
    keys = [
        (str(row.system), str(row.month), int(row.boundary), int(row.boundary_value))
        for row in base.itertuples(index=False)
    ]
    return sorted(keys, key=lambda x: (SYSTEM_ORDER.index(x[0]), x[1], x[2], x[3]))


def round_score_grid(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("A screening score contains nonfinite values")
    return np.rint(np.abs(numeric) / TIE_GRID_RMB_PER_KWH).astype(np.int64)


def select_score(grid: np.ndarray, nominal_fraction: float) -> dict[str, Any]:
    n = int(len(grid))
    if n == 0:
        return {
            "required_k": 0,
            "cutoff_grid": None,
            "cutoff_RMB_per_kWh": None,
            "selected": np.zeros(0, dtype=bool),
            "selected_n": 0,
            "cutoff_tie_n": 0,
            "tie_added_n": 0,
            "all_zero": False,
            "flat": False,
        }
    required_k = int(math.ceil(nominal_fraction * n))
    cutoff_grid = int(np.sort(grid)[-required_k])
    selected = grid >= cutoff_grid
    selected_n = int(selected.sum())
    all_zero = bool(np.all(grid == 0))
    flat = bool(np.all(grid == grid[0]))
    return {
        "required_k": required_k,
        "cutoff_grid": cutoff_grid,
        "cutoff_RMB_per_kWh": float(cutoff_grid * TIE_GRID_RMB_PER_KWH),
        "selected": selected,
        "selected_n": selected_n,
        "cutoff_tie_n": int(np.sum(grid == cutoff_grid)),
        "tie_added_n": int(selected_n - required_k),
        "all_zero": all_zero,
        "flat": flat,
    }


def row_identity_metrics(e_selected: np.ndarray, t_selected: np.ndarray) -> dict[str, Any]:
    e_n = int(e_selected.sum())
    t_n = int(t_selected.sum())
    intersection_n = int(np.logical_and(e_selected, t_selected).sum())
    union_n = int(np.logical_or(e_selected, t_selected).sum())
    symmetric_difference_n = int(np.logical_xor(e_selected, t_selected).sum())
    return {
        "E_selected_n": e_n,
        "T_selected_n": t_n,
        "intersection_n": intersection_n,
        "union_n": union_n,
        "symmetric_difference_n": symmetric_difference_n,
        "E_T_count_identity": bool(union_n == e_n + t_n - intersection_n),
        "symmetric_difference_identity": bool(symmetric_difference_n == e_n + t_n - 2 * intersection_n),
        "Jaccard": share(intersection_n, union_n),
    }


def screening_group_definitions(e_selected: np.ndarray, t_selected: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "E_selected_only": e_selected & ~t_selected,
        "T_selected_only": t_selected & ~e_selected,
        "intersection": e_selected & t_selected,
        "union": e_selected | t_selected,
        "symmetric_difference": e_selected ^ t_selected,
    }


def compact_amount_group_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    stats = amount_statistics(frame["D30"], include_quantiles=True)
    return {
        "group_n": stats["n"],
        "group_D30_finite_n": stats["finite_n"],
        "D30_signed_mean_RMB": stats["D30_signed_mean_RMB"],
        "D30_signed_median_RMB": stats["D30_signed_median_RMB"],
        "D30_signed_q10_RMB": stats["D30_signed_q10_RMB"] if "D30_signed_q10_RMB" in stats else None,
        "D30_signed_q90_RMB": stats["D30_signed_q90_RMB"] if "D30_signed_q90_RMB" in stats else None,
        "D30_absolute_mean_RMB": stats["D30_absolute_mean_RMB"],
        "D30_absolute_median_RMB": stats["D30_absolute_median_RMB"],
        "abs_gt_0_90_n": stats["abs_gt_0_90_n"],
        "abs_gt_0_90_share": stats["abs_gt_0_90_share"],
        "negative_amount_n": stats["negative_amount_n"],
        "near_amount_n": stats["near_amount_n"],
        "positive_amount_n": stats["positive_amount_n"],
        "amount_category_closure": stats["amount_category_closure"],
    }


def make_screening_outputs(
    units: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    layer_rows: list[dict[str, Any]] = []
    group_amount_rows: list[dict[str, Any]] = []
    pooled_state: dict[tuple[str, str, float], dict[str, Any]] = {}
    primary_screen_diff = pd.Series(False, index=units.index, dtype=bool)
    layer_keys = fixed_layer_keys(units)

    for population in SCREEN_POPULATIONS:
        for nominal_fraction in NOMINAL_FRACTIONS:
            for system, month, boundary, boundary_value in layer_keys:
                frame = units.loc[
                    units["system"].eq(system)
                    & units["month"].eq(month)
                    & units["boundary"].eq(boundary)
                    & units[population]
                ].copy()
                frame = frame.sort_index()
                e_grid = round_score_grid(frame["E_star"])
                t_grid = round_score_grid(frame["T_star"])
                e_result = select_score(e_grid, nominal_fraction)
                t_result = select_score(t_grid, nominal_fraction)
                e_selected = e_result["selected"]
                t_selected = t_result["selected"]
                identity = row_identity_metrics(e_selected, t_selected)
                n = int(len(frame))
                layer_row: dict[str, Any] = {
                    "population": population,
                    "system": system,
                    "month": month,
                    "boundary": boundary,
                    "boundary_value": boundary_value,
                    "nominal_fraction": nominal_fraction,
                    "stratum_n": n,
                    "required_k_E": e_result["required_k"],
                    "required_k_T": t_result["required_k"],
                    "E_cutoff_grid_units": e_result["cutoff_grid"],
                    "T_cutoff_grid_units": t_result["cutoff_grid"],
                    "E_cutoff_RMB_per_kWh": e_result["cutoff_RMB_per_kWh"],
                    "T_cutoff_RMB_per_kWh": t_result["cutoff_RMB_per_kWh"],
                    "E_cutoff_tie_n": e_result["cutoff_tie_n"],
                    "T_cutoff_tie_n": t_result["cutoff_tie_n"],
                    "E_tie_added_n": e_result["tie_added_n"],
                    "T_tie_added_n": t_result["tie_added_n"],
                    "E_score_grid_min_units": int(e_grid.min()) if n else None,
                    "E_score_grid_max_units": int(e_grid.max()) if n else None,
                    "E_score_grid_n_unique": int(np.unique(e_grid).size) if n else 0,
                    "T_score_grid_min_units": int(t_grid.min()) if n else None,
                    "T_score_grid_max_units": int(t_grid.max()) if n else None,
                    "T_score_grid_n_unique": int(np.unique(t_grid).size) if n else 0,
                    "E_all_zero_score_layer": e_result["all_zero"],
                    "T_all_zero_score_layer": t_result["all_zero"],
                    "E_flat_score_layer": e_result["flat"],
                    "T_flat_score_layer": t_result["flat"],
                    "E_no_discrimination_layer": bool(e_result["selected_n"] == n),
                    "T_no_discrimination_layer": bool(t_result["selected_n"] == n),
                    "E_selection_actual_fraction": share(identity["E_selected_n"], n),
                    "T_selection_actual_fraction": share(identity["T_selected_n"], n),
                    "intersection_fraction": share(identity["intersection_n"], n),
                    "union_fraction": share(identity["union_n"], n),
                    "symmetric_difference_fraction": share(identity["symmetric_difference_n"], n),
                }
                layer_row.update(identity)
                layer_rows.append(layer_row)

                if population == "A" and nominal_fraction == PRIMARY_FRACTION:
                    primary_screen_diff.loc[frame.index] = e_selected ^ t_selected

                key = (population, system, nominal_fraction)
                state = pooled_state.setdefault(
                    key,
                    {
                        "n_layers": 0,
                        "stratum_n": 0,
                        "required_k_E": 0,
                        "required_k_T": 0,
                        "E_selected_n": 0,
                        "T_selected_n": 0,
                        "intersection_n": 0,
                        "union_n": 0,
                        "symmetric_difference_n": 0,
                        "E_cutoff_tie_n": 0,
                        "T_cutoff_tie_n": 0,
                        "E_tie_added_n": 0,
                        "T_tie_added_n": 0,
                        "E_all_zero_score_layers": 0,
                        "T_all_zero_score_layers": 0,
                        "E_flat_score_layers": 0,
                        "T_flat_score_layers": 0,
                        "E_no_discrimination_layers": 0,
                        "T_no_discrimination_layers": 0,
                        "E_selected_frames": [],
                        "T_selected_frames": [],
                        "intersection_frames": [],
                        "union_frames": [],
                        "symmetric_difference_frames": [],
                    },
                )
                state["n_layers"] += 1
                state["stratum_n"] += n
                for name in (
                    "required_k_E",
                    "required_k_T",
                    "E_selected_n",
                    "T_selected_n",
                    "intersection_n",
                    "union_n",
                    "symmetric_difference_n",
                    "E_cutoff_tie_n",
                    "T_cutoff_tie_n",
                    "E_tie_added_n",
                    "T_tie_added_n",
                ):
                    state[name] += int(layer_row[name])
                state["E_all_zero_score_layers"] += int(e_result["all_zero"])
                state["T_all_zero_score_layers"] += int(t_result["all_zero"])
                state["E_flat_score_layers"] += int(e_result["flat"])
                state["T_flat_score_layers"] += int(t_result["flat"])
                state["E_no_discrimination_layers"] += int(e_result["selected_n"] == n)
                state["T_no_discrimination_layers"] += int(t_result["selected_n"] == n)
                groups = screening_group_definitions(e_selected, t_selected)
                for group_name, mask in groups.items():
                    group_frame = frame.iloc[np.flatnonzero(mask)]
                    group_row: dict[str, Any] = {
                        "scope": "layer",
                        "population": population,
                        "system": system,
                        "month": month,
                        "boundary": boundary,
                        "boundary_value": boundary_value,
                        "nominal_fraction": nominal_fraction,
                        "screening_group": group_name,
                        "stratum_n": n,
                    }
                    group_row.update(compact_amount_group_statistics(group_frame))
                    group_amount_rows.append(group_row)
                    state[f"{group_name}_frames"] = state.get(f"{group_name}_frames", [])
                    state[f"{group_name}_frames"].append(group_frame)

    pooled_rows: list[dict[str, Any]] = []
    for (population, system, nominal_fraction), state in pooled_state.items():
        n = int(state["stratum_n"])
        pooled_row = {
            "scope": "pooled",
            "population": population,
            "system": system,
            "nominal_fraction": nominal_fraction,
            "layers_n": state["n_layers"],
            "stratum_n": n,
            "required_k_E_sum": state["required_k_E"],
            "required_k_T_sum": state["required_k_T"],
            "E_selected_n": state["E_selected_n"],
            "T_selected_n": state["T_selected_n"],
            "intersection_n": state["intersection_n"],
            "union_n": state["union_n"],
            "symmetric_difference_n": state["symmetric_difference_n"],
            "E_selection_actual_fraction": share(state["E_selected_n"], n),
            "T_selection_actual_fraction": share(state["T_selected_n"], n),
            "intersection_fraction": share(state["intersection_n"], n),
            "union_fraction": share(state["union_n"], n),
            "symmetric_difference_fraction": share(state["symmetric_difference_n"], n),
            "Jaccard": share(state["intersection_n"], state["union_n"]),
            "E_cutoff_tie_n_sum": state["E_cutoff_tie_n"],
            "T_cutoff_tie_n_sum": state["T_cutoff_tie_n"],
            "E_tie_added_n_sum": state["E_tie_added_n"],
            "T_tie_added_n_sum": state["T_tie_added_n"],
            "E_all_zero_score_layers_n": state["E_all_zero_score_layers"],
            "T_all_zero_score_layers_n": state["T_all_zero_score_layers"],
            "E_flat_score_layers_n": state["E_flat_score_layers"],
            "T_flat_score_layers_n": state["T_flat_score_layers"],
            "E_no_discrimination_layers_n": state["E_no_discrimination_layers"],
            "T_no_discrimination_layers_n": state["T_no_discrimination_layers"],
            "E_selected_all_units_layers_n": state["E_no_discrimination_layers"],
            "T_selected_all_units_layers_n": state["T_no_discrimination_layers"],
            "E_T_count_identity": bool(
                state["union_n"] == state["E_selected_n"] + state["T_selected_n"] - state["intersection_n"]
            ),
            "symmetric_difference_identity": bool(
                state["symmetric_difference_n"]
                == state["E_selected_n"] + state["T_selected_n"] - 2 * state["intersection_n"]
            ),
        }
        pooled_rows.append(pooled_row)
        for group_name in (
            "E_selected_only",
            "T_selected_only",
            "intersection",
            "union",
            "symmetric_difference",
        ):
            frames = state.get(f"{group_name}_frames", [])
            combined = pd.concat(frames, axis=0) if frames else units.iloc[0:0]
            group_row = {
                "scope": "pooled",
                "population": population,
                "system": system,
                "month": "",
                "boundary": "",
                "boundary_value": "",
                "nominal_fraction": nominal_fraction,
                "screening_group": group_name,
                "stratum_n": n,
                "layers_n": state["n_layers"],
            }
            group_row.update(compact_amount_group_statistics(combined))
            group_amount_rows.append(group_row)
    layer_df = pd.DataFrame(layer_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    group_df = pd.DataFrame(group_amount_rows)
    for frame in (layer_df, pooled_df, group_df):
        frame.sort_values(
            [column for column in ("population", "system", "nominal_fraction", "month", "boundary", "screening_group") if column in frame.columns],
            key=lambda col: col.astype(str),
            inplace=True,
            kind="mergesort",
        )
        frame.reset_index(drop=True, inplace=True)
    masks = {"primary_screening_symmetric_difference": primary_screen_diff}
    return layer_df, pooled_df, group_df, masks


def make_label_crosstab(units: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for system in SYSTEM_ORDER:
        frame = units.loc[units["system"].eq(system) & units["A"]].copy()
        frame["E_label"] = strict_label(frame["E_star"], KAPPA)
        frame["T_label"] = strict_label(frame["T_star"], KAPPA)
        mismatch = frame["E_label"] != frame["T_label"]
        either_nonneutral = ~(
            frame["E_label"].eq("neutral") & frame["T_label"].eq("neutral")
        )
        summary[system] = {
            "A_n": int(len(frame)),
            "mismatch_n": int(mismatch.sum()),
            "both_neutral_n": int((~either_nonneutral).sum()),
            "either_nonneutral_n": int(either_nonneutral.sum()),
            "mismatch_within_either_nonneutral_n": int((mismatch & either_nonneutral).sum()),
        }
        for e_label in LABEL_ORDER:
            for t_label in LABEL_ORDER:
                cell = frame.loc[frame["E_label"].eq(e_label) & frame["T_label"].eq(t_label)]
                rows.append(
                    {
                        "system": system,
                        "kappa_RMB_per_kWh": KAPPA,
                        "E_label": e_label,
                        "T_label": t_label,
                        "n": int(len(cell)),
                        "share_of_A": share(len(cell), len(frame)),
                        "mismatch": bool(e_label != t_label),
                        "either_nonneutral": bool(not (e_label == "neutral" and t_label == "neutral")),
                        "D30_signed_median_RMB": safe_median(finite_array(cell["D30"])),
                        "D30_absolute_median_RMB": safe_median(np.abs(finite_array(cell["D30"]))),
                    }
                )
    return pd.DataFrame(rows), summary


EVENT_DEFINITIONS = (
    (
        "labels_unequal_kappa_0_03",
        "E/T labels unequal at kappa=0.03 on A",
    ),
    (
        "abs_D30_gt_0_90_RMB",
        "abs(D30)>0.90 RMB on A",
    ),
    (
        "primary_screening_symmetric_difference",
        "A primary 20% screening symmetric difference",
    ),
)


def group_sort_key(value: Any) -> tuple[str, str]:
    return (str(type(value)), str(value))


def concentration_group_column(group_type: str) -> str:
    return {"station": "station_id", "month": "month", "boundary": "boundary"}[group_type]


def make_concentration_outputs(
    units: pd.DataFrame,
    label_summary: dict[str, Any],
    screening_masks: dict[str, pd.Series],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    a = units["A"]
    frame_a = units.loc[a].copy()
    frame_a["E_label"] = strict_label(frame_a["E_star"], KAPPA)
    frame_a["T_label"] = strict_label(frame_a["T_star"], KAPPA)
    event_masks = {
        "labels_unequal_kappa_0_03": frame_a["E_label"] != frame_a["T_label"],
        "abs_D30_gt_0_90_RMB": frame_a["D30"].abs() > AMOUNT_THRESHOLD_RMB,
        "primary_screening_symmetric_difference": screening_masks[
            "primary_screening_symmetric_difference"
        ].loc[frame_a.index].to_numpy(dtype=bool),
    }
    by_group: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    check_rows: dict[str, Any] = {}
    for group_type in GROUP_ORDER:
        group_column = concentration_group_column(group_type)
        rows: list[dict[str, Any]] = []
        for event_name, event_description in EVENT_DEFINITIONS:
            event_mask = event_masks[event_name]
            for system in SYSTEM_ORDER:
                system_frame = frame_a.loc[frame_a["system"].eq(system)]
                system_event = event_mask[frame_a["system"].eq(system)]
                system_event_n = int(system_event.sum())
                system_a_n = int(len(system_frame))
                group_counts = (
                    system_frame.groupby(group_column, dropna=False, sort=False, observed=True)
                    .size()
                    .rename("denominator_n")
                )
                event_counts = (
                    system_frame.loc[np.asarray(system_event, dtype=bool)]
                    .groupby(group_column, dropna=False, sort=False, observed=True)
                    .size()
                    .rename("event_n")
                )
                all_groups = sorted(set(group_counts.index.tolist()) | set(event_counts.index.tolist()), key=group_sort_key)
                group_rows: list[dict[str, Any]] = []
                for group_value in all_groups:
                    denom_n = int(group_counts.get(group_value, 0))
                    event_n = int(event_counts.get(group_value, 0))
                    group_rows.append(
                        {
                            "system": system,
                            "population": "A",
                            "event": event_name,
                            "event_description": event_description,
                            "group_type": group_type,
                            "group_value": group_value,
                            "denominator_n": denom_n,
                            "event_n": event_n,
                            "event_share_of_system_event": share(event_n, system_event_n),
                            "event_share_of_system_A": share(event_n, system_a_n),
                            "within_group_event_rate": share(event_n, denom_n),
                        }
                    )
                group_rows.sort(key=lambda row: (-int(row["event_n"]), group_sort_key(row["group_value"])))
                positive_rank = 0
                for rank, row in enumerate(group_rows, start=1):
                    row["rank_by_event_n"] = rank
                    row["positive_event_group"] = bool(int(row["event_n"]) > 0)
                    if row["positive_event_group"]:
                        positive_rank += 1
                        row["positive_event_rank"] = positive_rank
                    else:
                        row["positive_event_rank"] = None
                    row["top1_contributor"] = bool(row["positive_event_group"] and positive_rank <= 1)
                    row["top5_contributor"] = bool(row["positive_event_group"] and positive_rank <= 5)
                    row["top10_contributor"] = bool(row["positive_event_group"] and positive_rank <= 10)
                rows.extend(group_rows)
                top_k_values = (0, 1, 5, 10) if group_type == "station" else (0,)
                positive_group_rows = [row for row in group_rows if row["positive_event_group"]]
                for top_k in top_k_values:
                    excluded = positive_group_rows[:top_k]
                    excluded_event_n = sum(int(row["event_n"]) for row in excluded)
                    excluded_denominator_n = sum(int(row["denominator_n"]) for row in excluded)
                    remaining_event_n = system_event_n - excluded_event_n
                    remaining_denominator_n = system_a_n - excluded_denominator_n
                    if top_k == 0:
                        exclusion_status = "full"
                    elif system_event_n == 0:
                        exclusion_status = "not_applicable_zero_event"
                    elif len(positive_group_rows) < top_k:
                        exclusion_status = "applied_positive_groups_only"
                    else:
                        exclusion_status = "applied"
                    summary_rows.append(
                        {
                            "system": system,
                            "population": "A",
                            "event": event_name,
                            "event_description": event_description,
                            "group_type": group_type,
                            "top_k_excluded": top_k,
                            "full_event_n": system_event_n,
                            "full_denominator_A_n": system_a_n,
                            "full_event_share_of_A": share(system_event_n, system_a_n),
                            "all_groups_n": len(group_rows),
                            "positive_event_groups_n": len(positive_group_rows),
                            "excluded_event_n": excluded_event_n,
                            "excluded_denominator_A_n": excluded_denominator_n,
                            "remaining_event_n": remaining_event_n,
                            "remaining_denominator_A_n": remaining_denominator_n,
                            "remaining_event_rate": share(remaining_event_n, remaining_denominator_n),
                            "remaining_share_of_original_event": share(remaining_event_n, system_event_n),
                            "remaining_share_of_original_A": share(remaining_event_n, system_a_n),
                            "top_k_exclusion_status": exclusion_status,
                            "top_k_exclusion_not_applicable": bool(exclusion_status == "not_applicable_zero_event"),
                        }
                    )
                check_key = f"{group_type}|{event_name}|{system}"
                check_rows[check_key] = {
                    "group_denominator_sum": int(sum(int(row["denominator_n"]) for row in group_rows)),
                    "system_A_n": system_a_n,
                    "group_event_sum": int(sum(int(row["event_n"]) for row in group_rows)),
                    "system_event_n": system_event_n,
                }
        by_group[group_type] = pd.DataFrame(rows)
        by_group[group_type].sort_values(
            ["event", "system", "rank_by_event_n", "group_value"],
            key=lambda col: col.astype(str),
            inplace=True,
            kind="mergesort",
        )
        by_group[group_type].reset_index(drop=True, inplace=True)
    concentration_summary = pd.DataFrame(summary_rows)
    concentration_summary.sort_values(
        ["event", "system", "group_type", "top_k_excluded"],
        inplace=True,
        kind="mergesort",
    )
    concentration_summary.reset_index(drop=True, inplace=True)
    return by_group, concentration_summary, check_rows


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def make_figure(
    units: pd.DataFrame,
    screening_by_layer: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.2), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat
    colors = {"Beijing": "#2166ac", "UrbanEV": "#b2182b"}
    linestyles = {"A": "-", "R": "--", "H": ":"}
    labels = {"A": "A", "R": "R", "H": "H"}

    # Full-range signed ECDF: preserve discrete masses without creating a
    # clipping-induced pile-up at the edge of a plotting window.
    for system in SYSTEM_ORDER:
        for population in POPULATION_ORDER:
            values = finite_array(units.loc[units["system"].eq(system) & units[population], "D30"])
            if not len(values):
                continue
            values = np.sort(values)
            ax_a.step(
                values,
                np.arange(1, len(values) + 1) / len(values),
                where="post",
                linewidth=1.0,
                color=colors[system],
                linestyle=linestyles[population],
                alpha=0.9,
                label=f"{system} {labels[population]}",
            )
    ax_a.axvline(0, color="#666666", linewidth=0.7)
    ax_a.set_title("(a) Signed D30 distribution")
    ax_a.set_xscale("symlog", linthresh=0.1)
    ax_a.set_ylim(0, 1.02)
    ax_a.set_xlabel("D30 (RMB / reference 30 kWh; symlog)")
    ax_a.set_ylabel("Cumulative fraction")
    ax_a.legend(ncol=2, frameon=False, loc="lower right")

    # Full-range empirical absolute distribution on a symlog axis, retaining
    # the large mass at zero and the long tail.
    for system in SYSTEM_ORDER:
        for population in POPULATION_ORDER:
            values = np.sort(np.abs(finite_array(units.loc[units["system"].eq(system) & units[population], "D30"])))
            if not len(values):
                continue
            y = np.arange(1, len(values) + 1) / len(values)
            ax_b.step(values, y, where="post", color=colors[system], linestyle=linestyles[population], linewidth=1.0)
    ax_b.axvline(AMOUNT_THRESHOLD_RMB, color="#555555", linestyle="--", linewidth=0.8)
    ax_b.set_xscale("symlog", linthresh=0.1)
    ax_b.set_xlim(left=0)
    ax_b.set_ylim(0, 1.01)
    ax_b.set_title("(b) Absolute amount distribution")
    ax_b.set_xlabel("|D30| (RMB; symlog x-axis)")
    ax_b.set_ylabel("Cumulative fraction")
    ax_b.text(0.02, 0.03, "Dashed line: |D30| = 0.90 RMB", transform=ax_b.transAxes, fontsize=6.5)

    primary = screening_by_layer.loc[screening_by_layer["nominal_fraction"].eq(PRIMARY_FRACTION)].copy()
    primary["series"] = primary["system"] + " " + primary["population"]
    for (system, population), group in primary.groupby(["system", "population"], sort=False):
        group = group.sort_values(["month", "boundary"], kind="mergesort")
        marker_size = 12 + 12 * np.log1p(group["E_tie_added_n"].to_numpy() + group["T_tie_added_n"].to_numpy())
        ax_c.scatter(
            group["E_selection_actual_fraction"],
            group["T_selection_actual_fraction"],
            s=marker_size,
            color=colors[system],
            marker="o" if population == "A" else "s",
            alpha=0.62,
            edgecolor="white",
            linewidth=0.35,
            label=f"{system} {population}",
        )
    ax_c.axvline(PRIMARY_FRACTION, color="#777777", linewidth=0.7, linestyle="--")
    ax_c.axhline(PRIMARY_FRACTION, color="#777777", linewidth=0.7, linestyle="--")
    ax_c.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=0.7, linestyle=":")
    ax_c.set_xlim(-0.02, 1.05)
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.set_title("(c) Actual 20% input screening fractions")
    ax_c.set_xlabel("E* selected fraction")
    ax_c.set_ylabel("T* selected fraction")
    ax_c.legend(ncol=2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95, loc="lower right", bbox_to_anchor=(1, 0.09))
    ax_c.text(0.02, 0.025, "Larger points: more cutoff ties; dashed = 0.20", transform=ax_c.transAxes, fontsize=6.5)

    order = [(system, population) for system in SYSTEM_ORDER for population in SCREEN_POPULATIONS]
    x_positions = {key: i for i, key in enumerate(order)}
    for (system, population), group in primary.groupby(["system", "population"], sort=False):
        group = group.sort_values(["month", "boundary"], kind="mergesort").reset_index(drop=True)
        x = x_positions[(system, population)] + (np.arange(len(group)) - (len(group) - 1) / 2) * 0.008
        marker_size = 10 + 10 * np.log1p(group["E_tie_added_n"].to_numpy() + group["T_tie_added_n"].to_numpy())
        ax_d.scatter(
            x,
            group["symmetric_difference_fraction"],
            s=marker_size,
            color=colors[system],
            marker="o" if population == "A" else "s",
            alpha=0.62,
            edgecolor="white",
            linewidth=0.3,
        )
    for key, x in x_positions.items():
        group = primary.loc[primary["system"].eq(key[0]) & primary["population"].eq(key[1])]
        pooled_n = group["stratum_n"].sum()
        pooled_diff = (group["symmetric_difference_n"].sum() / pooled_n) if pooled_n else np.nan
        ax_d.plot([x - 0.14, x + 0.14], [pooled_diff, pooled_diff], color="#222222", linewidth=1.2)
    ax_d.set_xticks(list(x_positions.values()), [f"{s}\n{p}" for s, p in order])
    ax_d.set_ylim(-0.04, 1.03)
    ax_d.set_title("(d) Primary screening symmetric difference")
    ax_d.set_xlabel("Dark bar: pooled fraction in each population")
    ax_d.set_ylabel("Symmetric difference / stratum n")
    ax_d.text(0.02, 0.94, "Larger points: more cutoff ties", transform=ax_d.transAxes, fontsize=6.5)

    fig.suptitle("Observed amount distributions and input screening sensitivity", fontsize=10)
    pdf_path = output_dir / "Figure_input_application.pdf"
    png_path = output_dir / "Figure_input_application.png"
    metadata = {
        "Title": "Observed amount distributions and input screening sensitivity",
        "Author": "Deterministic analysis script",
        "Subject": "Applied Energy revision input application analysis",
        "Creator": "analyze_input_application.py",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    with PdfPages(pdf_path, metadata=metadata) as pdf:
        pdf.savefig(fig, metadata=metadata)
    fig.savefig(png_path, metadata={"Software": "analyze_input_application.py"})
    plt.close(fig)
    return [pdf_path, png_path]


def fnum(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{100 * float(value):.{digits}f}\\%"


def latex_escape(value: Any) -> str:
    text = str(value)
    return re.sub(r"([_%&#{}])", r"\\\1", text)


def build_latex_tables(
    amount_summary: pd.DataFrame,
    screening_pooled: pd.DataFrame,
    concentration_summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    amount = amount_summary.loc[amount_summary["scope"].eq("system")].copy()
    amount_lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{Continuous signed and absolute amount summaries by frozen population.}",
        r"\label{tab:input-application-amount}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"System & Pop. & $n$ & median $D_{30}$ & $q_{10}$ & $q_{90}$ & median $|D_{30}|$ & $|D_{30}|>0.90$ \\",
        r"\midrule",
    ]
    for row in amount.itertuples(index=False):
        amount_lines.append(
            f"{latex_escape(row.system)} & {row.population} & {int(row.n):,} & "
            f"{fnum(row.D30_signed_median_RMB)} & {fnum(row.D30_signed_q10_RMB)} & "
            f"{fnum(row.D30_signed_q90_RMB)} & {fnum(row.D30_absolute_median_RMB)} & "
            f"{int(row.abs_gt_0_90_n):,} ({pct(row.abs_gt_0_90_share)}) \\\\"
        )
    amount_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{0.97\textwidth}\footnotesize",
        r"Values are RMB per reference 30 kWh and use the original round-trip parsed D30 values for statistics. The three populations are the frozen serialized A/R/H memberships. The exceedance tail is strict; the near interval is $[-0.90,0.90]$ RMB.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    amount_path = table_dir / "input_application_amount_summary.tex"
    amount_path.write_text("\n".join(amount_lines) + "\n", encoding="utf-8", newline="\n")

    screening = screening_pooled.loc[screening_pooled["nominal_fraction"].eq(PRIMARY_FRACTION)].copy()
    screening_lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{Pooled actual selection and agreement at the nominal 20\% screening fraction.}",
        r"\label{tab:input-application-screening}",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"System & Pop. & $n$ & E selected (actual) & T selected (actual) & intersection & union & sym. diff. & Jaccard \\",
        r"\midrule",
    ]
    for row in screening.itertuples(index=False):
        screening_lines.append(
            f"{latex_escape(row.system)} & {row.population} & {int(row.stratum_n):,} & "
            f"{int(row.E_selected_n):,} ({pct(row.E_selection_actual_fraction)}) & "
            f"{int(row.T_selected_n):,} ({pct(row.T_selection_actual_fraction)}) & "
            f"{int(row.intersection_n):,} & {int(row.union_n):,} & "
            f"{int(row.symmetric_difference_n):,} ({pct(row.symmetric_difference_fraction)}) & "
            f"{fnum(row.get('Jaccard') if hasattr(row, 'get') else None)} \\\\"
        )
    # Namedtuple rows do not have a pooled Jaccard field in the primary output;
    # calculate it in the text from the displayed counts where needed.
    screening_lines = [
        line.replace("-- \\\\", "-- \\\\") for line in screening_lines
    ]
    # Rewrite the data lines with Jaccard from explicit row counts.  This keeps
    # the CSV and table definitions identical without adding a redundant field.
    header_len = 8
    rewritten = screening_lines[:header_len]
    for row in screening.itertuples(index=False):
        jaccard = share(int(row.intersection_n), int(row.union_n))
        rewritten.append(
            f"{latex_escape(row.system)} & {row.population} & {int(row.stratum_n):,} & "
            f"{int(row.E_selected_n):,} ({pct(row.E_selection_actual_fraction)}) & "
            f"{int(row.T_selected_n):,} ({pct(row.T_selection_actual_fraction)}) & "
            f"{int(row.intersection_n):,} & {int(row.union_n):,} & "
            f"{int(row.symmetric_difference_n):,} ({pct(row.symmetric_difference_fraction)}) & "
            f"{fnum(jaccard)} \\\\"
        )
    rewritten += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{0.97\textwidth}\footnotesize",
        r"A and R are screened separately within fixed disjoint system--month--boundary strata. Cutoffs use absolute E* and T* scores rounded to a $10^{-6}$ RMB/kWh grid; every unit tied at the kth rounded score is included. These are observed input-sensitivity summaries and do not measure truth accuracy.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    screening_path = table_dir / "input_application_screening_summary.tex"
    screening_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8", newline="\n")

    conc = concentration_summary.loc[
        concentration_summary["group_type"].eq("station")
        & concentration_summary["top_k_excluded"].isin([0, 1, 5, 10])
    ].copy()
    event_short = {
        "labels_unequal_kappa_0_03": "label mismatch",
        "abs_D30_gt_0_90_RMB": r"$|D_{30}|>0.90$",
        "primary_screening_symmetric_difference": r"20\% sym. diff.",
    }
    conc_lines = [
        r"\begin{table}[htbp]",
        r"\centering\scriptsize",
        r"\caption{Station concentration of three descriptive A-population events.}",
        r"\label{tab:input-application-concentration}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"System & Event & top $k$ removed & remaining events & remaining A & remaining rate \\",
        r"\midrule",
    ]
    for row in conc.itertuples(index=False):
        if row.top_k_exclusion_not_applicable:
            continue
        conc_lines.append(
            f"{latex_escape(row.system)} & {event_short[row.event]} & {int(row.top_k_excluded)} & "
            f"{int(row.remaining_event_n):,} & {int(row.remaining_denominator_A_n):,} & "
            f"{pct(row.remaining_event_rate)} \\\\"
        )
    conc_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{0.97\textwidth}\footnotesize",
        r"Top contributors are ranked by event count within system. Both event and A denominators exclude the same top $k$ stations; $k=0$ retains the full population. UrbanEV has no primary-screen discrepancy events, so its top-$k$ exclusions are inapplicable and omitted. Events are descriptive post-observation summaries.",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    conc_path = table_dir / "input_application_concentration_summary.tex"
    conc_path.write_text("\n".join(conc_lines) + "\n", encoding="utf-8", newline="\n")
    return [amount_path, screening_path, conc_path]


def build_readme(
    input_path: Path,
    protocol_path: Path,
    output_dir: Path,
    counts: dict[str, dict[str, int]],
    screening_pooled: pd.DataFrame,
    concentration_summary: pd.DataFrame,
    artifacts: list[Path],
) -> str:
    lines = [
        "# Input application: amount, screening, and concentration",
        "",
        "This package reproduces the locked post-review descriptive extension from one frozen unit-level CSV. It reads no raw orders, source parquet panels, manuscript files, or prior summary CSVs.",
        "",
        "## Reproducibility command",
        "",
        "```powershell",
        "python `",
        "  workflows/distribution/src/distribution/analyze_input_application.py `",
        f"  --input \"{input_path}\" `",
        f"  --protocol \"{protocol_path}\" `",
        f"  --output \"{output_dir}\"",
        "```",
        "",
        "The input is parsed with pandas float_precision=round_trip. The script verifies the protocol's input SHA-256 before reading the panel.",
        "",
        "## Definitions",
        "",
        "A, R, and H amount summaries use the complete frozen serialized memberships. D30 is the signed amount in RMB per reference 30 kWh; quantiles use NumPy linear (type 7) interpolation on the original parsed values. The amount tails are strict at ±0.90 RMB, with the near interval including both endpoints. Signed reference counts at zero are also retained.",
        "",
        "A and R are screened separately within fixed disjoint system--month--boundary strata. For each nominal fraction q, k=ceil(qn). Absolute E* and T* scores are rounded to a 10^-6 RMB/kWh grid for ordering and ties only; every unit at or above the kth rounded score is included. All-zero score layers and flat-score layers remain in the output and are flagged separately. A screening no-discrimination flag means selected_n equals the full stratum n, including cases where a broad rounded cutoff tie selects the whole layer. Pooled rows sum disjoint layer counts while retaining the original A or R denominator.",
        "",
        "Concentration events are E/T label inequality at common kappa=0.03 on A, abs(D30)>0.90 RMB on A, and the A primary 20% screening symmetric difference. Station, month, and boundary summaries report event and A denominators, event shares, and within-group rates. Top-k rows rank contributors by event count and remove them from both event and denominator for the remaining rate.",
        "",
        "These summaries describe observed input sensitivity on identical local tasks. They are not truth accuracy, bills, savings, user choice, demand response, or flexible-load rankings. Concentration exclusions are sensitivity descriptions, not new primary samples.",
        "",
        "## Frozen primary count check",
        "",
        "| System | A | R | H | C |",
        "|---|---:|---:|---:|---:|",
    ]
    for system in SYSTEM_ORDER:
        row = counts[system]
        lines.append(f"| {system} | {row['A']:,} | {row['R']:,} | {row['H']:,} | {row['C']:,} |")
    lines += [
        "",
        "The machine-readable closure and identity checks are in `input_application_checks.json`. The 3 x 3 input-label cross-tab is in `input_label_crosstab.csv`; all required tables and figures are listed below.",
        "",
        "## Artifacts",
        "",
    ]
    for artifact in artifacts:
        lines.append(f"- `{artifact.relative_to(output_dir).as_posix()}`")
    lines.append("")
    return "\n".join(lines)


def build(input_path: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    protocol_path = protocol_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = load_protocol(protocol_path)
    expected_hash = protocol.get("input", {}).get("sha256")
    input_hash = sha256_file(input_path)
    if expected_hash and input_hash != expected_hash:
        raise ValueError(f"Input SHA-256 does not match locked protocol: {input_hash} != {expected_hash}")
    units, subset_checks = load_units(input_path)
    counts = primary_counts(units)
    if counts != EXPECTED_PRIMARY:
        raise AssertionError(f"Primary count mismatch: expected={EXPECTED_PRIMARY}, computed={counts}")
    amount_summary = make_amount_summary(units)
    label_crosstab, label_checks = make_label_crosstab(units)
    source_category_checks = check_source_categories(units)
    screening_by_layer, screening_pooled, screening_group_amounts, screening_masks = make_screening_outputs(units)
    concentration_by_group, concentration_summary, concentration_checks = make_concentration_outputs(
        units, label_checks, screening_masks
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "amount_summary.csv", amount_summary)
    write_csv(output_dir / "screening_by_layer.csv", screening_by_layer)
    write_csv(output_dir / "screening_pooled.csv", screening_pooled)
    write_csv(output_dir / "screening_group_amounts.csv", screening_group_amounts)
    write_csv(output_dir / "input_label_crosstab.csv", label_crosstab)
    write_csv(output_dir / "concentration_by_station.csv", concentration_by_group["station"])
    write_csv(output_dir / "concentration_by_month.csv", concentration_by_group["month"])
    write_csv(output_dir / "concentration_by_boundary.csv", concentration_by_group["boundary"])
    write_csv(output_dir / "concentration_summary.csv", concentration_summary)
    figure_paths = make_figure(units, screening_by_layer, output_dir)
    table_paths = build_latex_tables(amount_summary, screening_pooled, concentration_summary, output_dir)

    # Meaningful checks are computed from the independent representations,
    # rather than mirror tests that repeat the implementation line by line.
    amount_check: dict[str, Any] = {}
    for system in SYSTEM_ORDER:
        frame = units.loc[units["system"].eq(system) & units["A"]]
        d30 = frame["D30"].to_numpy(dtype=float)
        formula_error = np.abs(d30 - REFERENCE_KWH * (frame["T_star"].to_numpy() - frame["E_star"].to_numpy()))
        abs_error = np.abs(frame["D30_abs"].to_numpy() - np.abs(d30))
        amount_check[system] = {
            "A_finite_D30_n": int(np.isfinite(d30).sum()),
            "D30_formula_max_abs_error_RMB": float(np.max(formula_error)),
            "D30_abs_formula_max_abs_error_RMB": float(np.max(abs_error)),
        }
    screening_check_rows = screening_by_layer.to_dict("records")
    screening_identity_ok = bool(
        screening_by_layer["E_T_count_identity"].all()
        and screening_by_layer["symmetric_difference_identity"].all()
        and screening_pooled["E_T_count_identity"].all()
        and screening_pooled["symmetric_difference_identity"].all()
    )
    pooled_groups = screening_group_amounts.loc[screening_group_amounts["scope"].eq("pooled")].copy()
    pooled_group_pivot = pooled_groups.pivot_table(
        index=["population", "system", "nominal_fraction"],
        columns="screening_group",
        values="group_n",
        aggfunc="sum",
        fill_value=0,
    )
    group_amount_identity_ok = True
    for key, row in pooled_group_pivot.iterrows():
        e_only = int(row.get("E_selected_only", 0))
        t_only = int(row.get("T_selected_only", 0))
        intersection = int(row.get("intersection", 0))
        union = int(row.get("union", 0))
        symmetric_difference = int(row.get("symmetric_difference", 0))
        group_amount_identity_ok &= (
            e_only + intersection == int(
                screening_pooled.set_index(["population", "system", "nominal_fraction"]).loc[key, "E_selected_n"]
            )
            and t_only + intersection == int(
                screening_pooled.set_index(["population", "system", "nominal_fraction"]).loc[key, "T_selected_n"]
            )
            and union == intersection + e_only + t_only
            and symmetric_difference == e_only + t_only
        )
    all_zero_rows = screening_by_layer.loc[
        screening_by_layer["E_all_zero_score_layer"] | screening_by_layer["T_all_zero_score_layer"]
    ]
    all_zero_check = {
        "all_zero_score_layer_rows_n": int(len(all_zero_rows)),
        "all_zero_E_rows": int(screening_by_layer["E_all_zero_score_layer"].sum()),
        "all_zero_T_rows": int(screening_by_layer["T_all_zero_score_layer"].sum()),
        "flat_E_rows": int(screening_by_layer["E_flat_score_layer"].sum()),
        "flat_T_rows": int(screening_by_layer["T_flat_score_layer"].sum()),
        "selected_all_units_E_rows": int(screening_by_layer["E_no_discrimination_layer"].sum()),
        "selected_all_units_T_rows": int(screening_by_layer["T_no_discrimination_layer"].sum()),
        "all_zero_E_selection_is_full_layer": bool(
            screening_by_layer.loc[screening_by_layer["E_all_zero_score_layer"], "E_selected_n"].eq(
                screening_by_layer.loc[screening_by_layer["E_all_zero_score_layer"], "stratum_n"]
            ).all()
        ),
        "all_zero_T_selection_is_full_layer": bool(
            screening_by_layer.loc[screening_by_layer["T_all_zero_score_layer"], "T_selected_n"].eq(
                screening_by_layer.loc[screening_by_layer["T_all_zero_score_layer"], "stratum_n"]
            ).all()
        ),
        "all_zero_layers_retained_in_output": True,
        "cutoff_tie_rows_n": int(
            (screening_by_layer["E_cutoff_tie_n"] > 0).sum() + (screening_by_layer["T_cutoff_tie_n"] > 0).sum()
        ),
        "tie_added_units_total": int(
            screening_by_layer["E_tie_added_n"].sum() + screening_by_layer["T_tie_added_n"].sum()
        ),
    }
    concentration_identity_ok = all(
        int(values["group_denominator_sum"]) == int(values["system_A_n"])
        and int(values["group_event_sum"]) == int(values["system_event_n"])
        for values in concentration_checks.values()
    )
    cross_tab_ok = all(
        int(label_checks[system]["A_n"]) == int(
            label_crosstab.loc[label_crosstab["system"].eq(system), "n"].sum()
        )
        for system in SYSTEM_ORDER
    )
    finite_ok = bool(
        units.loc[units["A"], ["E_star", "T_star", "D30", "D30_abs"]].notna().all().all()
        and units.loc[units["H"], ["D30", "D30_abs"]].notna().all().all()
    )
    checks = {
        "schema": "applied-energy/input-application-v1",
        "status": "PASS",
        "input": {
            "path": str(input_path),
            "sha256": input_hash,
            "protocol_path": str(protocol_path),
            "protocol_expected_sha256": expected_hash,
            "sha256_match": bool(input_hash == expected_hash),
            "float_precision": "round_trip",
            "rows": int(len(units)),
            "columns": list(units.columns),
        },
        "parameters": {
            "populations": list(POPULATION_ORDER),
            "screening_populations": list(SCREEN_POPULATIONS),
            "strata": ["system", "month", "boundary"],
            "nominal_fractions": list(NOMINAL_FRACTIONS),
            "tie_grid_RMB_per_kWh": TIE_GRID_RMB_PER_KWH,
            "amount_threshold_RMB": AMOUNT_THRESHOLD_RMB,
            "reference_kWh": REFERENCE_KWH,
            "label_kappa_RMB_per_kWh": KAPPA,
            "threshold_rule": "strict tails; inclusive neutral/near interval; all cutoff ties included",
            "concentration_events": [name for name, _ in EVENT_DEFINITIONS],
            "interpretation": "observed input sensitivity and post-observation descriptive concentration; not truth accuracy",
        },
        "expected_primary_counts": EXPECTED_PRIMARY,
        "computed_primary_counts": counts,
        "checks": {
            "full_primary_counts_match": bool(counts == EXPECTED_PRIMARY),
            "subset_checks": subset_checks,
            "finite_required_values": finite_ok,
            "source_category_and_amount_identities": source_category_checks,
            "amount_formula_checks": amount_check,
            "cross_tab_counts": label_checks,
            "cross_tab_count_closure": cross_tab_ok,
            "screening_layer_count": int(len(screening_by_layer)),
            "screening_pooled_count": int(len(screening_pooled)),
            "screening_group_amount_count": int(len(screening_group_amounts)),
            "screening_identities": {
                "layer_and_pooled_count_identities": screening_identity_ok,
                "pooled_group_amount_identities": group_amount_identity_ok,
                "layer_denominator_closure": bool(
                    screening_by_layer.groupby(["population", "system", "nominal_fraction"], observed=True)["stratum_n"].sum().to_dict()
                    == screening_pooled.set_index(["population", "system", "nominal_fraction"])["stratum_n"].to_dict()
                ),
            },
            "all_zero_and_tie_handling": all_zero_check,
            "concentration_group_closure": concentration_identity_ok,
            "concentration_checks": concentration_checks,
        },
    }
    if not (
        checks["checks"]["full_primary_counts_match"]
        and finite_ok
        and all(item["total_category_mismatch_n"] == 0 and item["signed_category_mismatch_n"] == 0 for item in source_category_checks.values())
        and all(item["D30_formula_max_abs_error_RMB"] <= 1e-9 for item in amount_check.values())
        and all(item["D30_abs_formula_max_abs_error_RMB"] <= 1e-9 for item in amount_check.values())
        and cross_tab_ok
        and screening_identity_ok
        and group_amount_identity_ok
        and all_zero_check["all_zero_E_selection_is_full_layer"]
        and all_zero_check["all_zero_T_selection_is_full_layer"]
        and concentration_identity_ok
    ):
        checks["status"] = "FAIL"
        write_json(output_dir / "input_application_checks.json", checks)
        raise AssertionError("One or more input application meaning checks failed")

    artifacts = [
        output_dir / name
        for name in (
            "amount_summary.csv",
            "screening_by_layer.csv",
            "screening_pooled.csv",
            "screening_group_amounts.csv",
            "input_label_crosstab.csv",
            "concentration_by_station.csv",
            "concentration_by_month.csv",
            "concentration_by_boundary.csv",
            "concentration_summary.csv",
        )
    ] + figure_paths + table_paths
    checks["output_artifacts"] = {
        str(path.relative_to(output_dir)).replace("\\", "/"): {
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
        }
        for path in artifacts
    }
    checks_path = output_dir / "input_application_checks.json"
    write_json(checks_path, checks)
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        build_readme(input_path, protocol_path, output_dir, counts, screening_pooled, concentration_summary, artifacts + [checks_path]),
        encoding="utf-8",
        newline="\n",
    )
    checks["output_artifacts"]["README.md"] = {
        "sha256": sha256_file(readme_path),
        "bytes": int(readme_path.stat().st_size),
    }
    write_json(checks_path, checks)
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "primary_counts": counts,
        "primary_20pct_screening_rows": int(len(screening_pooled.loc[screening_pooled["nominal_fraction"].eq(PRIMARY_FRACTION)])),
        "concentration_rows": int(len(concentration_summary)),
        "figure_paths": [str(path) for path in figure_paths],
        "table_paths": [str(path) for path in table_paths],
        "checks_path": str(checks_path),
        "readme_path": str(readme_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Frozen unit_level_signed.csv")
    parser.add_argument("--protocol", type=Path, required=True, help="Locked revision_protocol.json")
    parser.add_argument("--output", type=Path, required=True, help="Output analysis_results/input_application directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.input, args.protocol, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
