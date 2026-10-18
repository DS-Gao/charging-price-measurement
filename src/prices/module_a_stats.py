# Exact existing statistical functions, extracted without estimator changes.
from __future__ import annotations
import numpy as np
import pandas as pd
def iqr_type7(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    q = np.quantile(values, [0.25, 0.75], method="linear")
    return float(q[1] - q[0])

def component_covariance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return np.nan
    return float(np.mean((x - x.mean()) * (y - y.mean())))

def summarize_layer(
    frame: pd.DataFrame,
    config_name: str,
    sample_scope: str,
    period: str = "fullmonth",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return construction summaries and covariance identity rows."""

    subset = frame.loc[frame["period"].eq(period)].copy()
    support = subset[f"{config_name}_support_before"] & subset[f"{config_name}_support_after"]
    if sample_scope == "fullsample":
        mask = support
    elif sample_scope in {"common_original_A", "common_original_R", "common_original_H", "common_all_configs"}:
        # Fixed-member summaries use one and the same population for every
        # price configuration.  In particular, Original is also restricted
        # to the intersection of Original/CD/DW support; its fullsample is
        # intentionally reported separately below.  This prevents the
        # original column from silently using all-original-R/H while CD/DW
        # use the smaller common-date population.
        membership = "A" if sample_scope in {"common_original_A", "common_all_configs"} else sample_scope.rsplit("_", 1)[-1]
        # Keep the original-A base explicit even for the nested R/H subsets;
        # this documents the requested original-A/R/H-and-support contract.
        mask = subset["original_A"].astype(bool) & subset[f"original_{membership}"].astype(bool)
        for c in ["original", "cd", "dw"]:
            mask &= subset[f"{c}_support_before"].astype(bool) & subset[f"{c}_support_after"].astype(bool)
    else:
        raise ValueError(sample_scope)
    subset = subset.loc[mask].copy()
    layer_rows: list[dict[str, Any]] = []
    cov_rows: list[dict[str, Any]] = []
    for (month, hour), g in subset.groupby(["month", "boundary_hour"], sort=True):
        E = g[f"{config_name}_E_star"].to_numpy(dtype=float)
        S = g[f"{config_name}_S_star"].to_numpy(dtype=float)
        T = g[f"{config_name}_T_star"].to_numpy(dtype=float)
        Z = T - E
        residual = T - E - S
        D30 = 30.0 * Z
        cov_es = component_covariance(E, S)
        cov_er = component_covariance(E, residual)
        cov_ez = component_covariance(E, Z)
        var_identity_residual = float(np.var(T) - (np.var(E) + np.var(Z) + 2.0 * cov_ez)) if len(E) else np.nan
        layer_rows.append(
            {
                "config": config_name,
                "sample_scope": sample_scope,
                "period": period,
                "month": month,
                "boundary_hour": int(hour),
                "n_units": int(len(g)),
                "median_signed_D30": float(np.median(D30)) if len(D30) else np.nan,
                "median_abs_D30": float(np.median(np.abs(D30))) if len(D30) else np.nan,
                "D30_p10": float(np.quantile(D30, 0.10, method="linear")) if len(D30) else np.nan,
                "D30_p25": float(np.quantile(D30, 0.25, method="linear")) if len(D30) else np.nan,
                "D30_p50": float(np.quantile(D30, 0.50, method="linear")) if len(D30) else np.nan,
                "D30_p75": float(np.quantile(D30, 0.75, method="linear")) if len(D30) else np.nan,
                "D30_p90": float(np.quantile(D30, 0.90, method="linear")) if len(D30) else np.nan,
                "D30_below_minus_0_90": int((D30 < -0.90).sum()),
                "D30_within_band": int((np.abs(D30) <= 0.90).sum()),
                "D30_above_plus_0_90": int((D30 > 0.90).sum()),
                "FIQR": float(iqr_type7(T) - iqr_type7(E)) if len(E) else np.nan,
                "n_original_A": int(g["original_A"].fillna(False).sum()),
                "n_original_R": int(g["original_R"].fillna(False).sum()),
                "n_original_H": int(g["original_H"].fillna(False).sum()),
                "n_config_R": int(g[f"{config_name}_R"].fillna(False).sum()),
                "n_config_H": int(g[f"{config_name}_H"].fillna(False).sum()),
                "fixed_membership_basis": (
                    "configuration-specific-supported-sample"
                    if sample_scope == "fullsample"
                    else ("original_A" if sample_scope in {"common_original_A", "common_all_configs"} else f"original_A+original_{sample_scope.rsplit('_', 1)[-1]}")
                ),
                "fixed_support_basis": (
                    config_name if sample_scope == "fullsample" else "original+cd+dw"
                ),
            }
        )
        cov_rows.append(
            {
                "config": config_name,
                "sample_scope": sample_scope,
                "period": period,
                "month": month,
                "boundary_hour": int(hour),
                "n_units": int(len(g)),
                "var_E": float(np.var(E)) if len(E) else np.nan,
                "var_S": float(np.var(S)) if len(S) else np.nan,
                "var_r": float(np.var(residual)) if len(E) else np.nan,
                "var_Z": float(np.var(Z)) if len(E) else np.nan,
                "var_T": float(np.var(T)) if len(E) else np.nan,
                "cov_E_S": cov_es,
                "cov_E_r": cov_er,
                "cov_E_Z": cov_ez,
                "cov_identity_residual": float(cov_ez - cov_es - cov_er) if np.isfinite(cov_ez) else np.nan,
                "variance_identity_residual": var_identity_residual,
                "cov_identity_pass": bool(np.isfinite(cov_ez) and abs(cov_ez - cov_es - cov_er) <= 1e-12),
                "variance_identity_pass": bool(len(E) and abs(var_identity_residual) <= 1e-12),
                "fixed_membership_basis": (
                    "configuration-specific-supported-sample"
                    if sample_scope == "fullsample"
                    else ("original_A" if sample_scope in {"common_original_A", "common_all_configs"} else f"original_A+original_{sample_scope.rsplit('_', 1)[-1]}")
                ),
                "fixed_support_basis": config_name if sample_scope == "fullsample" else "original+cd+dw",
            }
        )
    return pd.DataFrame(layer_rows), pd.DataFrame(cov_rows)

def transition_table(full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = full.loc[full["period"].eq("fullmonth")].copy()
    for config in ["cd", "dw"]:
        common = frame.loc[
            frame["original_A"]
            & frame[f"{config}_support_before"]
            & frame[f"{config}_support_after"]
        ].copy()
        for membership in ["A", "R", "H"]:
            if membership == "A":
                old = np.where(common["original_A"], "A", "not_A")
                new = np.where(common[f"{config}_support_before"] & common[f"{config}_support_after"], "A", "not_A")
            elif membership == "R":
                old = np.where(common["original_R"], "R", "not_R")
                new = np.where(common[f"{config}_R"].fillna(False), "R", "not_R")
            else:
                old = np.where(common["original_H"], "H", "not_H")
                new = np.where(common[f"{config}_H"].fillna(False), "H", "not_H")
            out = pd.DataFrame({"old": old, "new": new}).value_counts().rename("n_units").reset_index()
            out["config"] = config
            out["membership"] = membership
            out["sample_scope"] = "original_A_intersection"
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def fixed_member_scope_audit(layer_summary: pd.DataFrame) -> pd.DataFrame:
    """Audit the population contract used by every full-month summary.

    ``fullsample`` is deliberately configuration-specific.  Every
    ``common_original_*`` row, including the Original configuration, must use
    the same all-configuration support intersection.  The expected totals are
    the frozen R3 scope counts across the ten boundary layers.
    """

    expected = {
        "common_original_A": 7554,
        "common_all_configs": 7554,
        "common_original_R": 7047,
        "common_original_H": 7003,
    }
    rows: list[dict[str, Any]] = []
    for (config, scope), group in layer_summary.groupby(["config", "sample_scope"], sort=True):
        observed = int(group["n_units"].sum())
        if scope == "fullsample":
            expected_count = np.nan
            passed = True
            membership = "configuration-specific-supported-sample"
            support_basis = str(config)
        else:
            expected_count = int(expected[scope])
            passed = observed == expected_count
            membership = "original_A" if scope in {"common_original_A", "common_all_configs"} else f"original_A+original_{scope.rsplit('_', 1)[-1]}"
            support_basis = "original+cd+dw"
        rows.append(
            {
                "config": str(config),
                "sample_scope": str(scope),
                "n_units_total": observed,
                "expected_n_units_total": expected_count,
                "membership_basis": membership,
                "support_basis": support_basis,
                "scope_pass": bool(passed),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["config", "sample_scope"], kind="mergesort").reset_index(drop=True)
    fixed = audit.loc[audit["sample_scope"].ne("fullsample")]
    if not fixed["scope_pass"].all():
        raise AssertionError(f"fixed-member scope counts failed: {fixed.to_dict('records')}")
    # All three configurations must have identical fixed-member populations.
    for scope, group in fixed.groupby("sample_scope", sort=False):
        if group["n_units_total"].nunique() != 1:
            raise AssertionError(f"configuration-specific fixed population drift for {scope}")
    return audit
