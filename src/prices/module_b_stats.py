# Exact existing statistical functions, extracted without estimator changes.
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
FLOAT_TOL=1e-12
TOLERANCE=.03
KEYS=["month","boundary_hour","station_id"]
def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def weighted_quantile_tie(values: np.ndarray, weights: np.ndarray, q: float, tol: float = FLOAT_TOL) -> float:
    """Weighted ECDF inverse with merged ties and adjacent midpoint at exact hits."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return float("nan")
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    uniq: list[float] = []
    mass: list[float] = []
    for value, weight in zip(values, weights):
        if uniq and value == uniq[-1]:
            mass[-1] += float(weight)
        else:
            uniq.append(float(value))
            mass.append(float(weight))
    uniq_a = np.asarray(uniq, dtype=float)
    mass_a = np.asarray(mass, dtype=float)
    total = float(mass_a.sum())
    if total <= 0:
        return float("nan")
    target = float(q) * total
    cumulative = np.cumsum(mass_a)
    for i, upper in enumerate(cumulative):
        lower = 0.0 if i == 0 else float(cumulative[i - 1])
        if target < upper - tol * max(1.0, total):
            return float(uniq_a[i])
        if abs(target - upper) <= tol * max(1.0, total):
            if i + 1 < len(uniq_a):
                return float((uniq_a[i] + uniq_a[i + 1]) / 2.0)
            return float(uniq_a[i])
        if target < lower:
            return float(uniq_a[i])
    return float(uniq_a[-1])

def label_class(values: np.ndarray, direction: np.ndarray, tolerance: float = TOLERANCE) -> np.ndarray:
    standardized = np.asarray(values, dtype=float) * np.asarray(direction, dtype=float)
    return np.where(standardized > tolerance, 1, np.where(standardized < -tolerance, -1, 0))

def side_balanced_weights(before: np.ndarray, after: np.ndarray, multiplicity: np.ndarray | None = None) -> np.ndarray:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if multiplicity is None:
        multiplicity = np.ones(len(before), dtype=float)
    else:
        multiplicity = np.asarray(multiplicity, dtype=float)
    if len(before) == 0 or not np.isfinite(before).all() or not np.isfinite(after).all():
        return np.full(len(before), np.nan, dtype=float)
    b = before * multiplicity
    a = after * multiplicity
    bsum = float(b.sum())
    asum = float(a.sum())
    if bsum <= 0 or asum <= 0:
        return np.full(len(before), np.nan, dtype=float)
    return 0.5 * b / bsum + 0.5 * a / asum

def weights_for_panel(pairs: pd.DataFrame, panel: pd.DataFrame, perspective: str, multiplicity: dict[str, int] | None = None) -> np.ndarray | None:
    values = np.zeros(len(panel), dtype=float)
    if len(panel) == 0:
        return values
    mult_all = np.ones(len(pairs), dtype=float)
    if multiplicity is not None:
        mult_all = pairs["station_id"].map(multiplicity).fillna(0).to_numpy(float)
    if perspective == "ST":
        vals = panel["station_id"].map(multiplicity or {}).fillna(1).to_numpy(float) if multiplicity is not None else np.ones(len(panel), dtype=float)
        total = float(vals.sum())
        return vals / total if total > 0 else None
    before_col = "tx_before" if perspective == "TX" else "kwh_before"
    after_col = "tx_after" if perspective == "TX" else "kwh_after"
    if before_col not in pairs.columns or not np.isfinite(pairs[[before_col, after_col]].to_numpy(float)).all():
        return None
    base = side_balanced_weights(pairs[before_col].to_numpy(float), pairs[after_col].to_numpy(float), mult_all)
    if not np.isfinite(base).all():
        return None
    lookup = pd.Series(base, index=pairs.set_index(KEYS).index)
    panel_index = pd.MultiIndex.from_frame(panel[KEYS])
    vals = lookup.reindex(panel_index).to_numpy(float)
    total = float(np.nansum(vals))
    if total <= 0 or not np.isfinite(vals).all():
        return None
    return vals / total

def metric_row(panel: pd.DataFrame, config: str, weights: np.ndarray | None, scope: str, layer_order: int | str, month: str, group_label: str = "all") -> dict[str, Any]:
    base: dict[str, Any] = {
        "scope": scope,
        "layer_order": layer_order,
        "month": month,
        "group": group_label,
        "weight": "",
        "configuration": config,
        "n": int(len(panel)),
        "status": "empty" if len(panel) == 0 else "ok",
        "mae": np.nan,
        "me": np.nan,
        "p90_abs_error": np.nan,
        "label_disagreement": np.nan,
    }
    if weights is None:
        base["status"] = "weight_unavailable"
        return base
    if len(panel) == 0 or not np.isfinite(weights).all() or weights.sum() <= 0:
        base["status"] = "empty" if len(panel) == 0 else "invalid_weight"
        return base
    error = panel[config].to_numpy(float) - panel["Y"].to_numpy(float)
    target = panel["Y"].to_numpy(float)
    direction = panel["expected_direction"].to_numpy(float)
    base["mae"] = float(np.dot(weights, np.abs(error)))
    base["me"] = float(np.dot(weights, error))
    base["p90_abs_error"] = weighted_quantile_tie(np.abs(error), weights, 0.9)
    pred_labels = label_class(panel[config].to_numpy(float), direction)
    target_labels = label_class(target, direction)
    base["label_disagreement"] = float(np.dot(weights, (pred_labels != target_labels).astype(float)))
    return base

def layer_metrics(pairs: pd.DataFrame, perspective: str, multiplier: dict[str, int] | None = None, activity_info: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (month, layer_order), group in pairs.groupby(["month", "layer_order"], sort=True):
        subsets = {
            "all": group,
            "R_f": group[group["formation_R"]],
            "absZf_le_0.03": group[group["z_flat_group"].eq("absZf_le_0.03")],
            "absZf_gt_0.03": group[group["z_flat_group"].eq("absZf_gt_0.03")],
        }
        for group_label, subgroup in subsets.items():
            panel = subgroup[subgroup["V"]].copy()
            # Activity weights are defined on the complete A_f layer and then
            # conditioned to the requested evaluation subgroup.
            weights = weights_for_panel(group, panel, perspective, multiplier)
            for config in ["E", "CAL", "LOCAL"]:
                row = metric_row(panel, config, weights, "layer", int(layer_order), str(month), group_label)
                row["weight"] = perspective
                row["calibration_n_min"] = int(group["cal_n_other_Af"].min())
                row["calibration_n_max"] = int(group["cal_n_other_Af"].max())
                rows.append(row)
            # Store paired MAE gains in separate records to keep the long table tidy.
            if weights is not None and len(panel):
                metrics = {config: metric_row(panel, config, weights, "layer", int(layer_order), str(month), group_label) for config in ["E", "CAL", "LOCAL"]}
                for comparison, source in [("E_to_LOCAL", "E"), ("CAL_to_LOCAL", "CAL")]:
                    rows.append({
                        "scope": "layer", "layer_order": int(layer_order), "month": str(month), "group": group_label,
                        "weight": perspective, "configuration": comparison, "n": int(len(panel)), "status": "ok",
                        "mae": float(metrics[source]["mae"] - metrics["LOCAL"]["mae"]), "me": np.nan,
                        "p90_abs_error": np.nan, "label_disagreement": np.nan,
                        "calibration_n_min": int(group["cal_n_other_Af"].min()), "calibration_n_max": int(group["cal_n_other_Af"].max()),
                    })
    return rows

def summary_metrics(point_rows: pd.DataFrame) -> pd.DataFrame:
    base = point_rows[point_rows["configuration"].isin(["E", "CAL", "LOCAL"])].copy()
    out: list[dict[str, Any]] = []
    for perspective in sorted(base["weight"].dropna().unique()):
        for group_label in sorted(base["group"].dropna().unique()):
            for config in ["E", "CAL", "LOCAL"]:
                for month, subset in [("2025-01", base[(base.weight == perspective) & (base.group == group_label) & (base.configuration == config) & (base.month == "2025-01")]), ("2025-07", base[(base.weight == perspective) & (base.group == group_label) & (base.configuration == config) & (base.month == "2025-07")]), ("ALL", base[(base.weight == perspective) & (base.group == group_label) & (base.configuration == config)])]:
                    valid = subset[subset.status.eq("ok")]
                    row = {
                        "scope": "month_equal_layers" if month != "ALL" else "overall_equal_10_layers",
                        "layer_order": "equal_layers", "month": month, "group": group_label, "weight": perspective,
                        "configuration": config, "n": int(valid["n"].sum()) if len(valid) else 0,
                        "n_layers": int(len(valid)), "status": "ok" if len(valid) else "empty",
                        "mae": float(valid["mae"].mean()) if len(valid) else np.nan,
                        "me": float(valid["me"].mean()) if len(valid) else np.nan,
                        "p90_abs_error": float(valid["p90_abs_error"].mean()) if len(valid) else np.nan,
                        "label_disagreement": float(valid["label_disagreement"].mean()) if len(valid) else np.nan,
                        "calibration_n_min": int(valid["calibration_n_min"].min()) if len(valid) else np.nan,
                        "calibration_n_max": int(valid["calibration_n_max"].max()) if len(valid) else np.nan,
                    }
                    out.append(row)
    for perspective in sorted(base["weight"].dropna().unique()):
        for group_label in sorted(base["group"].dropna().unique()):
            for month in ["2025-01", "2025-07", "ALL"]:
                subset = base[(base.weight == perspective) & (base.group == group_label) & (base.month.eq(month) if month != "ALL" else True)]
                for comparison, source in [("E_to_LOCAL", "E"), ("CAL_to_LOCAL", "CAL")]:
                    e = subset[(subset.configuration == source) & subset.status.eq("ok")].set_index("layer_order")
                    l = subset[(subset.configuration == "LOCAL") & subset.status.eq("ok")].set_index("layer_order")
                    common = e.index.intersection(l.index)
                    gain = e.loc[common, "mae"] - l.loc[common, "mae"]
                    out.append({
                        "scope": "month_equal_layers" if month != "ALL" else "overall_equal_10_layers", "layer_order": "equal_layers", "month": month, "group": group_label, "weight": perspective,
                        "configuration": comparison, "n": int(e.loc[common, "n"].sum()) if len(common) else 0, "n_layers": int(len(common)), "status": "ok" if len(common) else "empty",
                        "mae": float(gain.mean()) if len(common) else np.nan, "me": np.nan, "p90_abs_error": np.nan, "label_disagreement": np.nan,
                        "calibration_n_min": np.nan, "calibration_n_max": np.nan,
                    })
    return pd.DataFrame(out)

def make_coverage(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (month, layer_order), group in pairs.groupby(["month", "layer_order"], sort=True):
        for entry, mask in [("A_f", np.ones(len(group), dtype=bool)), ("C_f", group["C_f"].to_numpy(bool))]:
            entry_n = int(mask.sum())
            later_n = int((mask & group["O_v"].to_numpy(bool)).sum())
            rows.append({
                "month": str(month), "layer_order": int(layer_order), "entry_set": entry,
                "n_entry": entry_n, "n_later_observable": later_n,
                "input_coverage_vs_Af": float(entry_n / len(group)) if len(group) else np.nan,
                "later_availability": float(later_n / entry_n) if entry_n else np.nan,
                "denominator_Af_layer": int(len(group)),
            })
    return pd.DataFrame(rows)
