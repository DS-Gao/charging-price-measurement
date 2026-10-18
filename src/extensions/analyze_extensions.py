"""Finite, frozen-panel scientific extensions for the focused revision.

The script reads only derived panels and the frozen UrbanEV panel.  It does
not read transaction orders, run simulations, or modify the frozen analysis
code.  All paths are command-line parameters so the analysis can be replayed
from a moved revision package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PRIMARY_E = 0.01
PRIMARY_T = 0.03
PRIMARY_AMOUNT_RMB_PER_KWH = 0.03
COMMON_KAPPA = 0.03
REFERENCE_KWH = 30.0
PRIMARY_AMOUNT_RMB = 0.90
AMOUNT_THRESHOLDS_RMB_PER_KWH = (0.02, 0.03, 0.05)
E_THRESHOLDS = (0.005, 0.01, 0.02)
T_THRESHOLDS = (0.02, 0.03, 0.05)
LABEL_ORDER = ("negative", "neutral", "positive")
TOTAL_ORDER = ("retained", "neutralized", "reversed")
SIGNED_ORDER = ("negative", "near", "positive")

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs" / "09_price_architecture"
DEFAULT_OUT = BASE / "focused_manuscript_revision_20260914"
DEFAULT_BEIJING_PANEL = (
    BASE / "09b_charging_activity_exposure" / "station_boundary_exposure_panel.parquet"
)
DEFAULT_BEIJING_MENU = (
    BASE / "01_menu_feasibility" / "station_month_menu_eligibility.parquet"
)
DEFAULT_EXTERNAL_PANEL = (
    BASE / "urbanev_external_analysis_20260913" / "monthly_boundary_panel.parquet"
)
DEFAULT_PROTOCOL = DEFAULT_OUT / "revision_protocol.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [json_clean(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_clean(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.12g")


def finite_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def finite_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {"1", "true", "t", "yes"}
    ).fillna(False).astype(bool)


def signed_class(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric < -threshold, numeric > threshold],
            ["negative", "positive"],
            default="near",
        ),
        index=values.index,
        dtype="object",
    )


def total_class(values: pd.Series, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric > threshold, numeric < -threshold],
            ["retained", "reversed"],
            default="neutralized",
        ),
        index=values.index,
        dtype="object",
    )


def common_label(values: pd.Series, kappa: float) -> pd.Series:
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


def counts_for(series: pd.Series, order: Iterable[str]) -> dict[str, int]:
    counts = series.value_counts(dropna=False).to_dict()
    return {str(label): int(counts.get(label, 0)) for label in order}


def pct(value: int, denominator: int) -> float | None:
    return float(value / denominator) if denominator else None


def load_beijing(panel_path: Path, menu_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = pd.read_parquet(panel_path)
    menu = pd.read_parquet(menu_path)
    required_panel = {
        "station_id",
        "month",
        "boundary_hour",
        "E_star",
        "S_star",
        "T_star",
        "signal_status",
        "service_action",
        "total_outcome",
        "denominator_A",
        "denominator_R",
    }
    missing = sorted(required_panel - set(panel.columns))
    if missing:
        raise ValueError(f"Beijing panel missing required columns: {missing}")
    required_menu = {"candidate_partition", "station_id", "month", "full_menu_main"}
    missing = sorted(required_menu - set(menu.columns))
    if missing:
        raise ValueError(f"Beijing menu missing required columns: {missing}")
    native = menu.loc[menu["candidate_partition"].eq("NATIVE_8")].copy()
    key = ["station_id", "month"]
    if native.duplicated(key).any():
        raise ValueError("NATIVE_8 menu has duplicate station-month keys")
    native["full_menu_main"] = bool_series(native["full_menu_main"])
    joined = panel.merge(
        native[key + ["full_menu_main"]],
        on=key,
        how="left",
        validate="many_to_one",
        indicator="_menu_merge",
    )
    merge_counts = joined["_menu_merge"].value_counts().to_dict()
    nonzero_merge_counts = {key: value for key, value in merge_counts.items() if int(value)}
    if set(nonzero_merge_counts) != {"both"}:
        raise ValueError(f"Beijing NATIVE_8 join is incomplete: {merge_counts}")
    joined["A"] = bool_series(joined["denominator_A"])
    joined["R"] = bool_series(joined["denominator_R"])
    joined["C"] = joined["A"] & bool_series(joined["full_menu_main"])
    joined["H"] = joined["R"] & (pd.to_numeric(joined["T_star"], errors="coerce") > PRIMARY_T)
    joined["D30"] = REFERENCE_KWH * (
        pd.to_numeric(joined["T_star"], errors="coerce")
        - pd.to_numeric(joined["E_star"], errors="coerce")
    )
    joined["D30_no_residual"] = REFERENCE_KWH * pd.to_numeric(
        joined["S_star"], errors="coerce"
    )
    joined["residual_star_reconstructed"] = (
        pd.to_numeric(joined["T_star"], errors="coerce")
        - pd.to_numeric(joined["E_star"], errors="coerce")
        - pd.to_numeric(joined["S_star"], errors="coerce")
    )
    joined["total_category"] = total_class(joined["T_star"], PRIMARY_T)
    # The frozen categorical column is a source check, not the definition.
    source_total = joined["total_outcome"].astype("string")
    joined["_total_category_source_match"] = (
        source_total == joined["total_category"].astype("string")
    )
    joined["signed_category"] = signed_class(joined["D30"], PRIMARY_AMOUNT_RMB)
    joined["signed_category_no_residual"] = signed_class(
        joined["D30_no_residual"], PRIMARY_AMOUNT_RMB
    )
    joined["system"] = "Beijing"
    joined["boundary"] = joined["boundary_hour"].astype(str)
    joined["boundary_value"] = pd.to_numeric(joined["boundary_hour"], errors="coerce")
    joined["source_row"] = np.arange(len(joined))
    checks = {
        "rows": int(len(joined)),
        "key_duplicates": int(joined[["station_id", "month", "boundary_hour"]].duplicated().sum()),
        "menu_native_rows": int(len(native)),
        "menu_native_duplicate_keys": int(native[key].duplicated().sum()),
        "menu_join_counts": {str(k): int(v) for k, v in merge_counts.items()},
        "menu_missing_after_join": int(joined["full_menu_main"].isna().sum()),
        "source_total_category_matches": bool(joined["_total_category_source_match"].all()),
        "denominator_A": int(joined["A"].sum()),
        "denominator_R": int(joined["R"].sum()),
        "complete_menu_C_within_A": int(joined["C"].sum()),
    }
    return joined, checks


def load_external(panel_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    panel = pd.read_parquet(panel_path)
    required = {
        "variant",
        "station_id",
        "month",
        "boundary_minute",
        "paired_days",
        "complete_days",
        "E_star",
        "S_star",
        "T_star",
        "residual_star",
        "D30",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"UrbanEV panel missing required columns: {missing}")
    base = panel.loc[panel["variant"].eq("base")].copy()
    base["A"] = pd.to_numeric(base["paired_days"], errors="coerce") >= 10
    base["R"] = base["A"] & (pd.to_numeric(base["E_star"], errors="coerce") > PRIMARY_E)
    base["H"] = base["R"] & (pd.to_numeric(base["T_star"], errors="coerce") > PRIMARY_T)
    base["C"] = base["A"] & (pd.to_numeric(base["complete_days"], errors="coerce") >= 10)
    recomputed_d30 = REFERENCE_KWH * (
        pd.to_numeric(base["T_star"], errors="coerce")
        - pd.to_numeric(base["E_star"], errors="coerce")
    )
    base["D30_recomputed"] = recomputed_d30
    base["D30_no_residual"] = REFERENCE_KWH * pd.to_numeric(
        base["S_star"], errors="coerce"
    )
    base["residual_star_reconstructed"] = (
        pd.to_numeric(base["T_star"], errors="coerce")
        - pd.to_numeric(base["E_star"], errors="coerce")
        - pd.to_numeric(base["S_star"], errors="coerce")
    )
    base["total_category"] = total_class(base["T_star"], PRIMARY_T)
    base["signed_category"] = signed_class(
        base["D30_recomputed"], PRIMARY_AMOUNT_RMB
    )
    base["signed_category_no_residual"] = signed_class(
        base["D30_no_residual"], PRIMARY_AMOUNT_RMB
    )
    base["system"] = "UrbanEV"
    base["boundary"] = base["boundary_minute"].astype(str)
    base["boundary_value"] = pd.to_numeric(base["boundary_minute"], errors="coerce")
    base["source_row"] = np.arange(len(base))
    checks = {
        "rows_all_variants": int(len(panel)),
        "base_rows": int(len(base)),
        "base_key_duplicates": int(
            base[["station_id", "month", "boundary_minute"]].duplicated().sum()
        ),
        "base_variant_count": int(panel["variant"].eq("base").sum()),
        "base_D30_max_abs_recompute_difference": float(
            np.nanmax(np.abs(pd.to_numeric(base["D30"], errors="coerce") - base["D30_recomputed"]))
        ),
        "A": int(base["A"].sum()),
        "R": int(base["R"].sum()),
        "H": int(base["H"].sum()),
        "C": int(base["C"].sum()),
    }
    return base, checks


def add_summary_for(system: str, frame: pd.DataFrame, source_rows: int) -> dict[str, Any]:
    a = frame.loc[frame["A"]]
    r = frame.loc[frame["R"]]
    h = frame.loc[frame["H"]]
    c = frame.loc[frame["C"]]
    r_total = counts_for(r["total_category"], TOTAL_ORDER)
    h_signed = counts_for(h["signed_category"], SIGNED_ORDER)
    h_nores = counts_for(h["signed_category_no_residual"], SIGNED_ORDER)
    amount = PRIMARY_AMOUNT_RMB
    row: dict[str, Any] = {
        "system": system,
        "source_rows": int(source_rows),
        "A_n": int(len(a)),
        "R_n": int(len(r)),
        "H_n": int(len(h)),
        "C_n": int(len(c)),
        "R_over_A": pct(len(r), len(a)),
        "H_over_R": pct(len(h), len(r)),
        "C_over_A": pct(len(c), len(a)),
        "R_retained_n": r_total["retained"],
        "R_neutralized_n": r_total["neutralized"],
        "R_reversed_n": r_total["reversed"],
        "H_negative_n": h_signed["negative"],
        "H_near_n": h_signed["near"],
        "H_positive_n": h_signed["positive"],
        "H_nores_negative_n": h_nores["negative"],
        "H_nores_near_n": h_nores["near"],
        "H_nores_positive_n": h_nores["positive"],
        "H_signed_median_RMB": finite_median(h["D30"]),
        "H_absolute_median_RMB": finite_median(h["D30"].abs()),
        "H_signed_mean_RMB": finite_mean(h["D30"]),
        "H_absolute_mean_RMB": finite_mean(h["D30"].abs()),
        "H_abs_gt_amount_n": int((h["D30"].abs() > amount).sum()),
        "H_abs_gt_amount_share": pct(int((h["D30"].abs() > amount).sum()), len(h)),
        "H_nores_changed_n": int(
            (h["signed_category"] != h["signed_category_no_residual"]).sum()
        ),
        "H_nores_changed_share": pct(
            int((h["signed_category"] != h["signed_category_no_residual"]).sum()), len(h)
        ),
    }
    for name, subset in (("A", a), ("R", r), ("H", h), ("C", c)):
        row[f"{name}_D30_signed_median_RMB"] = finite_median(subset["D30"])
        row[f"{name}_D30_absolute_median_RMB"] = finite_median(subset["D30"].abs())
    return row


def make_strata(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    # Group the complete frozen panel so a month-boundary cell with no
    # supported A units remains visible in the appendix rather than silently
    # disappearing.
    for (system, month, boundary_value), group in frame.groupby(
        ["system", "month", "boundary_value"], sort=True, dropna=False
    ):
        a = group.loc[group["A"]]
        r = a.loc[a["R"]]
        h = a.loc[a["H"]]
        total = counts_for(r["total_category"], TOTAL_ORDER)
        signed = counts_for(h["signed_category"], SIGNED_ORDER)
        numeric_boundary = pd.to_numeric(pd.Series([boundary_value]), errors="coerce").iloc[0]
        if pd.isna(numeric_boundary):
            boundary = "--"
        elif system == "Beijing":
            boundary = f"{int(round(float(numeric_boundary))):02d}:00"
        else:
            total_minutes = int(round(float(numeric_boundary)))
            boundary = f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"
        rows.append(
            {
                "system": system,
                "month": str(month),
                "boundary": boundary,
                "boundary_value": float(numeric_boundary) if pd.notna(numeric_boundary) else None,
                "source_rows_n": int(len(group)),
                "A_n": int(len(a)),
                "R_n": int(len(r)),
                "H_n": int(len(h)),
                "C_n": int(a["C"].sum()),
                "R_retained_n": total["retained"],
                "R_neutralized_n": total["neutralized"],
                "R_reversed_n": total["reversed"],
                "H_negative_n": signed["negative"],
                "H_near_n": signed["near"],
                "H_positive_n": signed["positive"],
                "H_signed_median_RMB": finite_median(h["D30"]),
                "H_absolute_median_RMB": finite_median(h["D30"].abs()),
                "H_abs_gt_amount_n": int((h["D30"].abs() > PRIMARY_AMOUNT_RMB).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_residual_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for system, group in frame.loc[frame["H"]].groupby("system", sort=True):
        tab = pd.crosstab(
            group["signed_category"],
            group["signed_category_no_residual"],
        ).reindex(index=SIGNED_ORDER, columns=SIGNED_ORDER, fill_value=0)
        changed = int((group["signed_category"] != group["signed_category_no_residual"]).sum())
        summary_rows.append(
            {
                "system": system,
                "H_n": int(len(group)),
                "changed_n": changed,
                "changed_share": pct(changed, len(group)),
                "nores_negative_n": int((group["signed_category_no_residual"] == "negative").sum()),
                "nores_near_n": int((group["signed_category_no_residual"] == "near").sum()),
                "nores_positive_n": int((group["signed_category_no_residual"] == "positive").sum()),
            }
        )
        for observed in SIGNED_ORDER:
            for nores in SIGNED_ORDER:
                n = int(tab.loc[observed, nores])
                transition_rows.append(
                    {
                        "system": system,
                        "observed_signed_category": observed,
                        "no_residual_signed_category": nores,
                        "n": n,
                        "share_of_H": pct(n, len(group)),
                        "changed_cell": bool(observed != nores),
                    }
                )
    return pd.DataFrame(transition_rows), pd.DataFrame(summary_rows)


def make_input_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_rows: list[dict[str, Any]] = []
    c_rows: list[dict[str, Any]] = []
    for system, all_rows in frame.loc[frame["A"]].groupby("system", sort=True):
        group = all_rows.copy()
        group["E_label"] = common_label(group["E_star"], COMMON_KAPPA)
        group["T_label"] = common_label(group["T_star"], COMMON_KAPPA)
        for e_label in LABEL_ORDER:
            for t_label in LABEL_ORDER:
                cell = group.loc[(group["E_label"] == e_label) & (group["T_label"] == t_label)]
                matrix_rows.append(
                    {
                        "system": system,
                        "E_label": e_label,
                        "T_label": t_label,
                        "kappa_RMB_per_kWh": COMMON_KAPPA,
                        "n": int(len(cell)),
                        "share_of_A": pct(len(cell), len(group)),
                        "D30_signed_median_RMB": finite_median(cell["D30"]),
                        "D30_absolute_median_RMB": finite_median(cell["D30"].abs()),
                        "D30_signed_mean_RMB": finite_mean(cell["D30"]),
                        "D30_absolute_mean_RMB": finite_mean(cell["D30"].abs()),
                    }
                )
        key = ["station_id", "month", "boundary"]
        c = group.loc[group["C"]].copy()
        c_rows.extend(
            [
                {
                    "system": system,
                    "set_label": "A",
                    "n": int(len(group)),
                    "share_of_A": 1.0,
                    "R_n_primary": int(group["R"].sum()),
                    "H_n_primary": int(group["H"].sum()),
                    "D30_signed_median_RMB": finite_median(group["D30"]),
                    "D30_absolute_median_RMB": finite_median(group["D30"].abs()),
                    "E_negative_n": int((group["E_label"] == "negative").sum()),
                    "E_neutral_n": int((group["E_label"] == "neutral").sum()),
                    "E_positive_n": int((group["E_label"] == "positive").sum()),
                    "T_negative_n": int((group["T_label"] == "negative").sum()),
                    "T_neutral_n": int((group["T_label"] == "neutral").sum()),
                    "T_positive_n": int((group["T_label"] == "positive").sum()),
                },
                {
                    "system": system,
                    "set_label": "C",
                    "n": int(len(c)),
                    "share_of_A": pct(len(c), len(group)),
                    "R_n_primary": int(c["R"].sum()),
                    "H_n_primary": int(c["H"].sum()),
                    "D30_signed_median_RMB": finite_median(c["D30"]),
                    "D30_absolute_median_RMB": finite_median(c["D30"].abs()),
                    "E_negative_n": int((c["E_label"] == "negative").sum()),
                    "E_neutral_n": int((c["E_label"] == "neutral").sum()),
                    "E_positive_n": int((c["E_label"] == "positive").sum()),
                    "T_negative_n": int((c["T_label"] == "negative").sum()),
                    "T_neutral_n": int((c["T_label"] == "neutral").sum()),
                    "T_positive_n": int((c["T_label"] == "positive").sum()),
                },
            ]
        )
        # The C comparison must alter membership only.  Values for the
        # retained rows are copied from the same A rows by construction.
        if len(c):
            a_index = group.set_index(key)
            c_index = c.set_index(key)
            aligned = a_index.loc[c_index.index]
            max_e = float(np.nanmax(np.abs(aligned["E_star"] - c_index["E_star"])))
            max_t = float(np.nanmax(np.abs(aligned["T_star"] - c_index["T_star"])))
            max_d = float(np.nanmax(np.abs(aligned["D30"] - c_index["D30"])))
        else:
            max_e = max_t = max_d = None
        c_rows.append(
            {
                "system": system,
                "set_label": "C_membership_only_check",
                "n": int(len(c)),
                "share_of_A": pct(len(c), len(group)),
                "R_n_primary": int(c["R"].sum()),
                "H_n_primary": int(c["H"].sum()),
                "D30_signed_median_RMB": None,
                "D30_absolute_median_RMB": None,
                "E_negative_n": None,
                "E_neutral_n": None,
                "E_positive_n": None,
                "T_negative_n": None,
                "T_neutral_n": None,
                "T_positive_n": None,
                "max_abs_E_difference_after_membership_only": max_e,
                "max_abs_T_difference_after_membership_only": max_t,
                "max_abs_D30_difference_after_membership_only": max_d,
            }
        )
    return pd.DataFrame(matrix_rows), pd.DataFrame(c_rows)


def make_external_sensitivity(external: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    amount_primary = PRIMARY_AMOUNT_RMB
    specifications: list[tuple[str, float, float, float, str]] = []
    for value in E_THRESHOLDS:
        specifications.append(
            ("electricity_qualification", value, PRIMARY_T, amount_primary, f"E>{value:g}")
        )
    for value in T_THRESHOLDS:
        specifications.append(
            ("total_direction", PRIMARY_E, value, amount_primary, f"T>{value:g}")
        )
    for value in AMOUNT_THRESHOLDS_RMB_PER_KWH:
        specifications.append(
            (
                "amount_threshold_fixed_primary_H",
                PRIMARY_E,
                PRIMARY_T,
                value * REFERENCE_KWH,
                f"|D30|>{value * REFERENCE_KWH:g} RMB",
            )
        )
    for factor, e_threshold, t_threshold, amount_threshold, label in specifications:
        a = external["paired_days"].ge(10)
        r = a & (pd.to_numeric(external["E_star"], errors="coerce") > e_threshold)
        h = r & (pd.to_numeric(external["T_star"], errors="coerce") > t_threshold)
        c = a & external["complete_days"].ge(10)
        d30 = pd.to_numeric(external["D30_recomputed"], errors="coerce")
        r_total = total_class(external["T_star"], t_threshold).loc[r]
        h_signed = signed_class(d30, amount_threshold).loc[h]
        row: dict[str, Any] = {
            "system": "UrbanEV",
            "factor": factor,
            "variant_label": label,
            "electricity_threshold_RMB_per_kWh": e_threshold,
            "total_threshold_RMB_per_kWh": t_threshold,
            "amount_threshold_RMB_per_kWh": amount_threshold / REFERENCE_KWH,
            "amount_threshold_RMB": amount_threshold,
            "source_rows_base": int(len(external)),
            "A_n": int(a.sum()),
            "R_n": int(r.sum()),
            "H_n": int(h.sum()),
            "C_n": int(c.sum()),
            "R_retained_n": counts_for(r_total, TOTAL_ORDER)["retained"],
            "R_neutralized_n": counts_for(r_total, TOTAL_ORDER)["neutralized"],
            "R_reversed_n": counts_for(r_total, TOTAL_ORDER)["reversed"],
            "H_negative_n": counts_for(h_signed, SIGNED_ORDER)["negative"],
            "H_near_n": counts_for(h_signed, SIGNED_ORDER)["near"],
            "H_positive_n": counts_for(h_signed, SIGNED_ORDER)["positive"],
            "H_abs_gt_amount_n": int((d30.loc[h].abs() > amount_threshold).sum()),
            "H_abs_gt_amount_share": pct(int((d30.loc[h].abs() > amount_threshold).sum()), int(h.sum())),
            "R_total_near_n": int(r_total.eq("neutralized").sum()),
            "H_amount_near_n": int(h_signed.eq("near").sum()),
            "E_exact_boundary_n": int(
                (pd.to_numeric(external["E_star"], errors="coerce").loc[a] == e_threshold).sum()
            ),
            "T_exact_boundary_n": int(
                (pd.to_numeric(external["T_star"], errors="coerce").loc[r] == t_threshold).sum()
            ),
            "D30_abs_exact_boundary_n": int((d30.loc[h].abs() == amount_threshold).sum()),
            "H_signed_median_RMB": finite_median(d30.loc[h]),
            "H_absolute_median_RMB": finite_median(d30.loc[h].abs()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def make_unit_frame(beijing: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "S_star",
        "T_star",
        "D30",
        "D30_no_residual",
        "residual_star_reconstructed",
        "total_category",
        "signed_category",
        "signed_category_no_residual",
    ]
    combined = pd.concat([beijing[columns], external[columns]], ignore_index=True)
    combined["D30_abs"] = combined["D30"].abs()
    return combined


def run(
    out: Path,
    protocol_path: Path,
    beijing_panel_path: Path,
    beijing_menu_path: Path,
    external_panel_path: Path,
) -> dict[str, Any]:
    out = out.resolve()
    ext_out = out / "analysis_results" / "extensions"
    ext_out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    beijing, bj_checks = load_beijing(beijing_panel_path, beijing_menu_path)
    external, u_checks = load_external(external_panel_path)
    units = make_unit_frame(beijing, external)
    write_csv(ext_out / "unit_level_signed.csv", units)

    summary = pd.DataFrame(
        [
            add_summary_for("Beijing", beijing, len(beijing)),
            add_summary_for("UrbanEV", external.loc[external["A"]], len(external)),
        ]
    )
    write_csv(ext_out / "summary_short.csv", summary)
    write_csv(ext_out / "summary_full.csv", summary)

    strata = make_strata(units)
    write_csv(ext_out / "monthly_boundary_strata.csv", strata)
    transition, residual_summary = make_residual_tables(units)
    write_csv(ext_out / "residual_classification_transition.csv", transition)
    write_csv(ext_out / "residual_classification_summary.csv", residual_summary)

    audit_3x3, audit_c = make_input_audit(units)
    write_csv(ext_out / "input_audit_3x3.csv", audit_3x3)
    write_csv(ext_out / "input_audit_C.csv", audit_c)

    sensitivity = make_external_sensitivity(external)
    write_csv(ext_out / "threshold_sensitivity.csv", sensitivity)

    expected = protocol["baseline_counts"]
    computed = {
        "Beijing": {
            "A": int(beijing["A"].sum()),
            "R": int(beijing["R"].sum()),
            "H": int(beijing["H"].sum()),
            "C": int(beijing["C"].sum()),
        },
        "UrbanEV": {
            "A": int(external["A"].sum()),
            "R": int(external["R"].sum()),
            "H": int(external["H"].sum()),
            "C": int(external["C"].sum()),
        },
    }
    baseline_match = all(
        computed[system][key] == int(expected[system][key])
        for system in computed
        for key in ("A", "R", "H", "C")
    )
    closure_rows = []
    for row in summary.to_dict("records"):
        r_sum = row["R_retained_n"] + row["R_neutralized_n"] + row["R_reversed_n"]
        h_sum = row["H_negative_n"] + row["H_near_n"] + row["H_positive_n"]
        h_abs_sum = row["H_negative_n"] + row["H_positive_n"]
        closure_rows.append(
            {
                "system": row["system"],
                "R_category_sum": r_sum,
                "R_n": row["R_n"],
                "R_closure": bool(r_sum == row["R_n"]),
                "H_category_sum": h_sum,
                "H_n": row["H_n"],
                "H_closure": bool(h_sum == row["H_n"]),
                "H_abs_gt_amount_signed_sum": h_abs_sum,
                "H_abs_gt_amount_n": row["H_abs_gt_amount_n"],
                "H_signed_absolute_closure": bool(h_abs_sum == row["H_abs_gt_amount_n"]),
                "C_subset_A": bool(row["C_n"] <= row["A_n"]),
            }
        )
    closure = pd.DataFrame(closure_rows)
    write_csv(ext_out / "count_closure_checks.csv", closure)

    residual_identity = units.loc[units["A"]].copy()
    residual_identity["identity_error_RMB"] = residual_identity["D30"] - (
        residual_identity["D30_no_residual"]
        + REFERENCE_KWH * residual_identity["residual_star_reconstructed"]
    )
    max_identity_error = float(np.nanmax(np.abs(residual_identity["identity_error_RMB"])))
    menu_join_ok = bool(
        bj_checks["menu_missing_after_join"] == 0
        and bj_checks["menu_native_duplicate_keys"] == 0
        and bj_checks["key_duplicates"] == 0
    )
    source_audit = {
        "protocol": str(protocol_path.resolve()),
        "inputs": {
            "beijing_panel": {
                "path": str(beijing_panel_path.resolve()),
                "sha256": sha256_file(beijing_panel_path),
                "rows": int(len(beijing)),
                "columns": sorted(beijing.columns.tolist()),
            },
            "beijing_menu": {
                "path": str(beijing_menu_path.resolve()),
                "sha256": sha256_file(beijing_menu_path),
                "rows_native_NATIVE_8": int(
                    pd.read_parquet(beijing_menu_path)
                    .loc[lambda x: x["candidate_partition"].eq("NATIVE_8")]
                    .shape[0]
                ),
            },
            "external_panel": {
                "path": str(external_panel_path.resolve()),
                "sha256": sha256_file(external_panel_path),
                "rows_base": int(len(external)),
                "columns": sorted(external.columns.tolist()),
            },
        },
        "historical_e2_generator": "e2_masking.py (source identifier; not included or executed in this replay)",
        "read_scope": [
            "derived parquet panels",
            "NATIVE_8 menu eligibility",
            "revision protocol",
        ],
        "excluded": ["raw order files", "new simulations", "literature"],
    }
    write_json(ext_out / "source_audit.json", source_audit)
    write_json(ext_out / "baseline_counts.json", {"expected": expected, "computed": computed, "match": baseline_match})
    write_json(
        ext_out / "count_closure_checks.json",
        {
            "status": "PASS"
            if baseline_match and bool(closure["R_closure"].all()) and bool(closure["H_closure"].all())
            else "FAIL",
            "baseline_match": baseline_match,
            "menu_join_ok": menu_join_ok,
            "beijing": bj_checks,
            "urbanev": u_checks,
            "rows": closure.to_dict("records"),
            "residual_identity_max_abs_RMB": max_identity_error,
            "residual_identity_tolerance_RMB": 1e-9,
        },
    )
    write_json(
        ext_out / "analysis_metadata.json",
        {
            "schema": "focused-manuscript-revision/extensions-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "primary_e_threshold_RMB_per_kWh": PRIMARY_E,
                "primary_total_threshold_RMB_per_kWh": PRIMARY_T,
                "common_input_audit_kappa_RMB_per_kWh": COMMON_KAPPA,
                "reference_kWh": REFERENCE_KWH,
                "primary_amount_threshold_RMB": PRIMARY_AMOUNT_RMB,
                "amount_thresholds_RMB_per_kWh": list(AMOUNT_THRESHOLDS_RMB_PER_KWH),
                "external_electricity_thresholds_RMB_per_kWh": list(E_THRESHOLDS),
                "external_total_thresholds_RMB_per_kWh": list(T_THRESHOLDS),
                "threshold_sensitivity_registration": "post-review specified supplement; not original preregistration",
                "weighting": "equal unit within each observation system; no UrbanEV transaction weights",
            },
            "outputs": [
                "unit_level_signed.csv",
                "summary_short.csv",
                "monthly_boundary_strata.csv",
                "residual_classification_transition.csv",
                "threshold_sensitivity.csv",
                "input_audit_3x3.csv",
                "input_audit_C.csv",
                "baseline_counts.json",
                "count_closure_checks.json",
                "source_audit.json",
            ],
        },
    )
    if not baseline_match:
        raise AssertionError(f"Baseline count mismatch: expected={expected}, computed={computed}")
    if not menu_join_ok:
        raise AssertionError(f"Beijing menu join failed: {bj_checks}")
    if not bool(closure["R_closure"].all()) or not bool(closure["H_closure"].all()):
        raise AssertionError("Category closure failed")
    if max_identity_error > 1e-9:
        raise AssertionError(f"Residual identity failed: {max_identity_error}")
    return {
        "status": "PASS",
        "out": str(out),
        "summary": summary.to_dict("records"),
        "strata_rows": int(len(strata)),
        "threshold_rows": int(len(sensitivity)),
        "input_audit_rows": int(len(audit_3x3)),
        "residual_changed": residual_summary[["system", "changed_n", "changed_share"]].to_dict("records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--beijing-panel", type=Path, default=DEFAULT_BEIJING_PANEL)
    parser.add_argument("--beijing-menu", type=Path, default=DEFAULT_BEIJING_MENU)
    parser.add_argument("--external-panel", type=Path, default=DEFAULT_EXTERNAL_PANEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        args.out,
        args.protocol,
        args.beijing_panel,
        args.beijing_menu,
        args.external_panel,
    )
    print(json.dumps(json_clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
