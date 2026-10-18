"""Exact transaction-based comparable price construction.

The script reuses the frozen Stage 2 eligibility/calendar rules and reconstructs
comparable-date (CD) and date-weighted (DW) endpoint distributions from raw
transactions. DW is a date-equal weighted transaction distribution, not a
median of daily medians.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT
OUT = PACKAGE_ROOT / "results/module_a"
CONFIG_PATH = PACKAGE_ROOT / "config/price_construction.json"
STAGE1_PATH = PACKAGE_ROOT / "data/derived/prices/native_prices.parquet"
ORIGINAL_PANEL_PATH = PACKAGE_ROOT / "data/derived/prices/frozen_original_panel.parquet"
ORDERS = {}
RAW_COLUMNS = [
    "station_id",
    "pile_id",
    "charge_start_time",
    "charge_end_time",
    "duration_min",
    "total_elec_kwh",
    "total_elec_fee",
    "total_service_fee",
    "total_fee",
]
PRICE_COLUMNS = ["E", "S", "T"]
MASS_TOLERANCE = 1e-12
FEE_TOLERANCE = 0.0100001
BATCH_SIZE = 250_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_ready(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, np.datetime64):
        return str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def q_type7(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float(np.quantile(values, 0.5, method="linear"))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Merged-tie ECDF inverse with exact cumulative-half midpoint."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[keep]
    weights = weights[keep]
    if values.size == 0:
        return np.nan
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    unique, first = np.unique(values, return_index=True)
    sums = np.add.reduceat(weights, first)
    total = float(sums.sum())
    if total <= 0:
        return np.nan
    cumulative = np.cumsum(sums) / total
    hit = np.flatnonzero(np.abs(cumulative - 0.5) <= MASS_TOLERANCE)
    if hit.size:
        idx = int(hit[0])
        if idx + 1 < len(unique):
            return float((unique[idx] + unique[idx + 1]) / 2.0)
        return float(unique[idx])
    return float(unique[int(np.searchsorted(cumulative, 0.5, side="left"))])


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


@dataclass
class GroupData:
    E: np.ndarray
    S: np.ndarray
    T: np.ndarray
    tariff_day: np.ndarray
    kwh: np.ndarray


def group_concat(parts: list[np.ndarray], dtype: Any = float) -> np.ndarray:
    if not parts:
        return np.array([], dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def load_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    prices = pd.read_parquet(STAGE1_PATH)
    native = prices.loc[prices["candidate_partition"].eq("NATIVE_8")].copy()
    if native.duplicated(["month", "station_id", "segment_id"]).any():
        raise AssertionError("NATIVE_8 Stage 1 keys are not unique")
    panel = pd.read_parquet(ORIGINAL_PANEL_PATH)
    panel = panel.loc[panel["analysis_scope"].eq("PURE_TOU")].copy()
    if len(panel) != 8133:
        raise AssertionError(f"original A panel expected 8133 rows, found {len(panel)}")
    if panel.duplicated(["station_id", "month", "boundary_hour"]).any():
        raise AssertionError("original panel keys are not unique")
    return config, native, panel


def layer_table(config: dict[str, Any], panel: pd.DataFrame) -> pd.DataFrame:
    pure = {(str(m), int(h)) for m, h in config["pure_tou_layer_order"]}
    rows = []
    for row in config["boundary_layers"]:
        month, hour, before, after, before_label, after_label, scope, direction, analysis_scope = row
        if (str(month), int(hour)) in pure:
            rows.append(
                {
                    "month": str(month),
                    "boundary_hour": int(hour),
                    "before_segment": str(before),
                    "after_segment": str(after),
                    "before_label": str(before_label),
                    "after_label": str(after_label),
                    "policy_scope": str(scope),
                    "expected_direction": int(direction),
                    "analysis_scope": str(analysis_scope),
                }
            )
    layers = pd.DataFrame(rows)
    observed = panel[["month", "boundary_hour"]].drop_duplicates()
    check = layers.merge(observed, on=["month", "boundary_hour"], how="outer", indicator=True)
    if not check["_merge"].eq("both").all():
        raise AssertionError("config pure_tou layers and original panel layers differ")
    layers = layers.sort_values(["month", "boundary_hour"], kind="mergesort").reset_index(drop=True)
    layers["layer_order"] = np.arange(1, len(layers) + 1, dtype=int)
    return layers


def supported_pair_keys(native: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    """Return the union needed for full A, formation A_f, and validation A_v."""

    rows = []
    for layer in layers.to_dict("records"):
        frame = native.loc[native["month"].eq(layer["month"])]
        before = frame.loc[frame["segment_id"].eq(layer["before_segment"])].set_index("station_id")
        after = frame.loc[frame["segment_id"].eq(layer["after_segment"])].set_index("station_id")
        stations = before.index.intersection(after.index)
        if len(stations) == 0:
            continue
        pair = pd.DataFrame({"station_id": stations.astype(str)})
        pair["month"] = layer["month"]
        pair["boundary_hour"] = int(layer["boundary_hour"])
        pair["before_segment"] = layer["before_segment"]
        pair["after_segment"] = layer["after_segment"]
        pair["layer_order"] = int(layer["layer_order"])
        for col in ["support_main", "support_formation", "support_validation"]:
            pair[f"{col}_before"] = before.loc[stations, col].to_numpy(dtype=bool)
            pair[f"{col}_after"] = after.loc[stations, col].to_numpy(dtype=bool)
        pair["original_A"] = pair["support_main_before"] & pair["support_main_after"]
        pair["original_A_f"] = pair["support_formation_before"] & pair["support_formation_after"]
        pair["original_A_v"] = pair["support_validation_before"] & pair["support_validation_after"]
        pair = pair.loc[pair["original_A"] | pair["original_A_f"] | pair["original_A_v"]].copy()
        rows.append(pair)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["station_id", "month", "boundary_hour"]).any():
        raise AssertionError("supported boundary pair keys are not unique")
    return result.sort_values(["layer_order", "station_id"], kind="mergesort").reset_index(drop=True)


def station_codes_for_targets(pair_keys: pd.DataFrame) -> dict[str, dict[str, int]]:
    targets: dict[str, set[str]] = defaultdict(set)
    for row in pair_keys.to_dict("records"):
        targets[row["month"]].add(str(row["station_id"]))
    return {
        month: {s: i for i, s in enumerate(sorted(stations))}
        for month, stations in targets.items()
    }


def process_order(
    month: str,
    path: Path,
    target_keys: set[tuple[str, str]],
    station_codes: dict[str, int],
    segment_defs: list[tuple[str, int, int, int]],
) -> tuple[dict[tuple[str, str, str, str], list[np.ndarray]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read one source file once and retain targeted contained transactions."""

    parquet = pq.ParquetFile(path)
    seg_lookup = {row[0]: idx for idx, row in enumerate(segment_defs)}
    inverse_segment = {idx: row[0] for idx, row in enumerate(segment_defs)}
    target_codes = {
        station_codes[station] * 16 + seg_lookup[segment]
        for station, segment in target_keys
        if station in station_codes and segment in seg_lookup
    }
    if not target_codes:
        raise AssertionError(f"no raw targets for {month}")
    groups: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
    counters: dict[str, Any] = {
        "rows_seen": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.num_row_groups),
        "batches": 0,
        "price_base_rows_before_finite_filter": 0,
        "price_base_rows_dropped_nonfinite_or_fee_identity": 0,
        "price_base_rows": 0,
        "time_order_violations": 0,
        "null_end_times": 0,
        "assigned_native_rows": 0,
        "contained_native_rows": 0,
        "contained_target_rows": 0,
        "formation_target_rows": 0,
        "validation_target_rows": 0,
        "outside_split_target_rows": 0,
        "pile_id_nonnull_rows": 0,
    }
    facility_pairs: dict[str, set[str]] = defaultdict(set)
    calendar_examples: dict[str, Any] = {}
    month_start = pd.Timestamp(f"{month}-01")
    month_end = month_start + pd.offsets.MonthBegin(1)
    formation_end = month_start + pd.Timedelta(days=15)
    month_start_day = month_start.to_datetime64().astype("datetime64[D]")
    formation_end_day = formation_end.to_datetime64().astype("datetime64[D]")
    month_end_day = month_end.to_datetime64().astype("datetime64[D]")
    inverse_station = {value: key for key, value in station_codes.items()}

    for batch in parquet.iter_batches(columns=RAW_COLUMNS, batch_size=BATCH_SIZE):
        counters["batches"] += 1
        frame = batch.to_pandas()
        counters["pile_id_nonnull_rows"] += int(frame["pile_id"].notna().sum())
        nonnull_facility = frame.loc[frame["station_id"].notna() & frame["pile_id"].notna(), ["station_id", "pile_id"]]
        for station, pile in nonnull_facility.drop_duplicates().itertuples(index=False, name=None):
            facility_pairs[str(pile)].add(str(station))

        start = frame["charge_start_time"]
        end = frame["charge_end_time"]
        price_base = frame.loc[
            frame["station_id"].notna()
            & start.notna()
            & start.ge(month_start)
            & start.lt(month_end)
            & frame["duration_min"].gt(0)
            & frame["total_elec_kwh"].gt(0)
            & frame["total_elec_fee"].ge(0)
            & frame["total_service_fee"].ge(0)
            & frame["total_fee"].ge(0)
        ].copy()
        if price_base.empty:
            continue
        counters["price_base_rows_before_finite_filter"] += int(len(price_base))
        kwh = price_base["total_elec_kwh"].to_numpy(dtype=float)
        elec_fee = price_base["total_elec_fee"].to_numpy(dtype=float)
        service_fee = price_base["total_service_fee"].to_numpy(dtype=float)
        total_fee = price_base["total_fee"].to_numpy(dtype=float)
        E = elec_fee / kwh
        S = service_fee / kwh
        T = total_fee / kwh
        finite = np.isfinite(E) & np.isfinite(S) & np.isfinite(T)
        valid = finite & (np.abs(total_fee - elec_fee - service_fee) <= FEE_TOLERANCE)
        counters["price_base_rows_dropped_nonfinite_or_fee_identity"] += int((~valid).sum())
        if not valid.all():
            keep = np.flatnonzero(valid)
            price_base = price_base.iloc[keep].copy()
            E, S, T, kwh = E[valid], S[valid], T[valid], kwh[valid]
        if price_base.empty:
            continue
        counters["price_base_rows"] += int(len(price_base))
        start = price_base["charge_start_time"]
        end = price_base["charge_end_time"]
        counters["time_order_violations"] += int((end < start).fillna(False).sum())
        counters["null_end_times"] += int(end.isna().sum())
        start_minute = start.dt.hour.to_numpy(dtype=np.int16) * 60 + start.dt.minute.to_numpy(dtype=np.int16)
        day_start = start.dt.normalize().to_numpy(dtype="datetime64[ns]")
        segment_code = np.full(len(price_base), -1, dtype=np.int8)
        segment_start = day_start.copy()
        for code, (_, start_hour, end_hour, _) in enumerate(segment_defs):
            if start_hour < end_hour:
                mask = (start_minute >= start_hour * 60) & (start_minute < end_hour * 60)
                segment_start[mask] = day_start[mask] + np.timedelta64(start_hour, "h")
            else:
                mask = (start_minute >= start_hour * 60) | (start_minute < end_hour * 60)
                segment_start[mask] = day_start[mask] + np.timedelta64(start_hour, "h")
                rollover = mask & (start_minute < end_hour * 60)
                segment_start[rollover] -= np.timedelta64(1, "D")
            segment_code[mask] = code
        assigned = segment_code >= 0
        counters["assigned_native_rows"] += int(assigned.sum())
        duration_hours = np.asarray([row[3] for row in segment_defs], dtype=np.int16)
        contained = assigned & (start.to_numpy(dtype="datetime64[ns]") >= segment_start)
        contained &= end.to_numpy(dtype="datetime64[ns]") <= (
            segment_start + duration_hours[np.maximum(segment_code, 0)].astype("timedelta64[h]")
        )
        counters["contained_native_rows"] += int(contained.sum())
        if not contained.any():
            continue
        station_code = price_base["station_id"].map(station_codes).to_numpy(dtype=float)
        contained &= np.isfinite(station_code)
        if not contained.any():
            continue
        safe_station_code = np.where(np.isfinite(station_code), station_code, 0).astype(np.int64)
        safe_segment_code = np.where(segment_code >= 0, segment_code, 0).astype(np.int64)
        code_int = safe_station_code * 16 + safe_segment_code
        contained &= np.isin(code_int, np.fromiter(target_codes, dtype=np.int64))
        if not contained.any():
            continue
        counters["contained_target_rows"] += int(contained.sum())
        tariff_day = (start - pd.Timedelta(hours=23)).dt.normalize().to_numpy(dtype="datetime64[D]")
        split = np.full(len(price_base), "outside_month_split", dtype=object)
        first_half = (tariff_day >= month_start_day) & (tariff_day < formation_end_day)
        second_half = (tariff_day >= formation_end_day) & (tariff_day < month_end_day)
        split[first_half] = "formation"
        split[second_half] = "validation"
        counters["formation_target_rows"] += int((contained & first_half).sum())
        counters["validation_target_rows"] += int((contained & second_half).sum())
        counters["outside_split_target_rows"] += int((contained & ~(first_half | second_half)).sum())
        target_idx = np.flatnonzero(contained)
        # Preserve a few real timestamp examples for the calendar audit. The
        # price construction itself continues to use tariff_day only.
        for idx in target_idx:
            timestamp = pd.Timestamp(start.iloc[idx])
            if int(timestamp.hour) in {0, 6, 7, 22, 23}:
                label = f"{month}:{int(timestamp.hour):02d}"
                if label not in calendar_examples:
                    calendar_examples[label] = {
                        "charge_start_time": timestamp.isoformat(),
                        "charge_end_time": pd.Timestamp(end.iloc[idx]).isoformat() if pd.notna(end.iloc[idx]) else None,
                        "segment_id": inverse_segment[int(segment_code[idx])],
                        "tariff_day": str(tariff_day[idx]),
                        "station_id": str(price_base.iloc[idx]["station_id"]),
                    }
        for code in np.unique(code_int[target_idx]):
            use = target_idx[code_int[target_idx] == code]
            station = inverse_station[int(code) // 16]
            segment = inverse_segment[int(code) % 16]
            key_base = (month, station, segment)
            for half in ["fullmonth", "formation", "validation"]:
                idx = use if half == "fullmonth" else use[split[use] == half]
                if len(idx) == 0:
                    continue
                target = groups.setdefault(key_base + (half,), [[], [], [], [], []])
                target[0].append(E[idx])
                target[1].append(S[idx])
                target[2].append(T[idx])
                target[3].append(tariff_day[idx])
                target[4].append(kwh[idx])
    return groups, counters, {pile: sorted(stations) for pile, stations in facility_pairs.items()}, calendar_examples


def finalize_groups(
    raw_groups: dict[tuple[str, str, str, str], list[np.ndarray]],
) -> dict[tuple[str, str, str, str], GroupData]:
    output: dict[tuple[str, str, str, str], GroupData] = {}
    for key, parts in raw_groups.items():
        output[key] = GroupData(
            E=group_concat(parts[0]),
            S=group_concat(parts[1]),
            T=group_concat(parts[2]),
            tariff_day=group_concat(parts[3], dtype="datetime64[D]"),
            kwh=group_concat(parts[4]),
        )
    return output


def summary_from_arrays(E: np.ndarray, S: np.ndarray, T: np.ndarray, days: np.ndarray, kwh: np.ndarray, mode: str) -> dict[str, Any]:
    if len(E) == 0:
        return {"E": np.nan, "S": np.nan, "T": np.nan, "n_transactions": 0, "n_dates": 0, "kwh": 0.0}
    n_dates = int(np.unique(days).size)
    if mode in ["original", "cd"]:
        med = [q_type7(x) for x in [E, S, T]]
    elif mode == "dw":
        weights = np.empty(len(E), dtype=float)
        for day in np.unique(days):
            mask = days == day
            weights[mask] = 1.0 / (n_dates * int(mask.sum()))
        med = [weighted_median(x, weights) for x in [E, S, T]]
    else:
        raise ValueError(mode)
    return {
        "E": float(med[0]),
        "S": float(med[1]),
        "T": float(med[2]),
        "n_transactions": int(len(E)),
        "n_dates": n_dates,
        "kwh": float(np.sum(kwh, dtype=float)),
    }


def common_subset(a: GroupData | None, b: GroupData | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if a is None or b is None or len(a.E) == 0 or len(b.E) == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype="datetime64[D]"), np.array([], dtype=float), np.array([], dtype="datetime64[D]")
    common = np.intersect1d(np.unique(a.tariff_day), np.unique(b.tariff_day))
    if common.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype="datetime64[D]"), np.array([], dtype="datetime64[D]"), np.array([], dtype=float), common
    keep = np.isin(a.tariff_day, common)
    return a.E[keep], a.S[keep], a.T[keep], a.tariff_day[keep], a.kwh[keep], common


def endpoint_from_pair(data: GroupData | None, mode: str, pair_values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
    E, S, T, days, kwh, _ = pair_values
    return summary_from_arrays(E, S, T, days, kwh, mode)


def stage1_endpoint_lookup(native: pd.DataFrame) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in native.to_dict("records"):
        lookup[(str(row["month"]), str(row["station_id"]), str(row["segment_id"]), "all")] = {
            "original_fullmonth": {
                "E": row["median_elec_price"], "S": row["median_service_price"], "T": row["median_total_price_observed"],
                "n_transactions": int(row["n_sessions"]), "n_dates": int(row["n_tariff_days"]), "kwh": np.nan, "support": bool(row["support_main"]),
            },
            "original_formation": {
                "E": row["median_elec_price"], "S": row["median_service_price"], "T": row["median_total_price_observed"],
                "n_transactions": int(row["formation_n_sessions"]), "n_dates": int(row["formation_n_tariff_days"]), "kwh": np.nan, "support": bool(row["support_formation"]),
            },
            "original_validation": {
                "E": row["median_elec_price"], "S": row["median_service_price"], "T": row["median_total_price_observed"],
                "n_transactions": int(row["validation_n_sessions"]), "n_dates": int(row["validation_n_tariff_days"]), "kwh": np.nan, "support": bool(row["support_validation"]),
            },
        }
    return lookup


def class_membership(E_star: float, T_star: float, direction: int) -> tuple[Any, bool, bool]:
    oriented_E = direction * E_star
    oriented_T = direction * T_star
    if abs(E_star) <= 0.01:
        signal = "weak_signal"
    elif oriented_E > 0.01:
        signal = "signal_bearing"
    else:
        signal = "discordant_electricity_signal"
    R = signal == "signal_bearing"
    return signal, R, bool(R and oriented_T > 0.03)


def derive_unit_row(
    pair: dict[str, Any],
    layer: dict[str, Any],
    period: str,
    endpoint_data: dict[tuple[str, str, str, str], GroupData],
    stage_lookup: dict[tuple[str, str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    month = str(pair["month"])
    station = str(pair["station_id"])
    before_segment = str(pair["before_segment"])
    after_segment = str(pair["after_segment"])
    direction = int(layer["expected_direction"])
    bdata = endpoint_data.get((month, station, before_segment, period))
    adata = endpoint_data.get((month, station, after_segment, period))
    # Stage 1 is the original endpoint authority.  Raw kWh is attached below
    # because the Stage 1 cache stores counts and medians but not half kWh.
    sl_b = stage_lookup.get((month, station, before_segment, "all"), {}).get(f"original_{period}", {})
    sl_a = stage_lookup.get((month, station, after_segment, "all"), {}).get(f"original_{period}", {})
    original_b = dict(sl_b)
    original_a = dict(sl_a)
    if period != "fullmonth":
        if bdata is not None:
            raw_b = summary_from_arrays(bdata.E, bdata.S, bdata.T, bdata.tariff_day, bdata.kwh, "original")
            original_b.update({field: raw_b[field] for field in ["E", "S", "T", "n_transactions", "n_dates", "kwh"]})
        if adata is not None:
            raw_a = summary_from_arrays(adata.E, adata.S, adata.T, adata.tariff_day, adata.kwh, "original")
            original_a.update({field: raw_a[field] for field in ["E", "S", "T", "n_transactions", "n_dates", "kwh"]})

    # All CD/DW endpoint prices use this boundary-specific date intersection.
    if bdata is None or adata is None:
        common_dates = np.array([], dtype="datetime64[D]")
        empty = np.array([], dtype=float)
        bpair = (empty, empty, empty, np.array([], dtype="datetime64[D]"), empty, common_dates)
        apair = bpair
    else:
        common_dates = np.intersect1d(np.unique(bdata.tariff_day), np.unique(adata.tariff_day))
        bkeep = np.isin(bdata.tariff_day, common_dates)
        akeep = np.isin(adata.tariff_day, common_dates)
        bpair = (bdata.E[bkeep], bdata.S[bkeep], bdata.T[bkeep], bdata.tariff_day[bkeep], bdata.kwh[bkeep], common_dates)
        apair = (adata.E[akeep], adata.S[akeep], adata.T[akeep], adata.tariff_day[akeep], adata.kwh[akeep], common_dates)
    threshold_n, threshold_d = (20, 5) if period == "fullmonth" else (10, 3)
    b_cd = endpoint_from_pair(bdata, "cd", bpair)
    a_cd = endpoint_from_pair(adata, "cd", apair)
    b_dw = endpoint_from_pair(bdata, "dw", bpair)
    a_dw = endpoint_from_pair(adata, "dw", apair)
    b_cd["support"] = bool(len(common_dates) >= threshold_d and b_cd["n_transactions"] >= threshold_n)
    a_cd["support"] = bool(len(common_dates) >= threshold_d and a_cd["n_transactions"] >= threshold_n)
    b_dw["support"] = b_cd["support"]
    a_dw["support"] = a_cd["support"]

    row: dict[str, Any] = {
        "station_id": station,
        "month": month,
        "boundary_hour": int(layer["boundary_hour"]),
        "layer_order": int(layer["layer_order"]),
        "before_segment": before_segment,
        "after_segment": after_segment,
        "expected_direction": direction,
        "period": period,
        "common_dates": int(len(common_dates)),
        "common_date_min": str(common_dates.min()) if len(common_dates) else None,
        "common_date_max": str(common_dates.max()) if len(common_dates) else None,
        "original_support": bool(original_b.get("support", False) and original_a.get("support", False)),
        "cd_support": bool(b_cd["support"] and a_cd["support"]),
        "dw_support": bool(b_dw["support"] and a_dw["support"]),
        "original_A": bool(pair.get("original_A", False)) if period == "fullmonth" else np.nan,
        "original_A_f": bool(pair.get("original_A_f", False)) if period == "formation" else np.nan,
        "original_A_v": bool(pair.get("original_A_v", False)) if period == "validation" else np.nan,
    }
    endpoint_map = {"original": (original_b, original_a), "cd": (b_cd, a_cd), "dw": (b_dw, a_dw)}
    for config_name, (before, after) in endpoint_map.items():
        for side, data in [("before", before), ("after", after)]:
            for field in ["E", "S", "T", "n_transactions", "n_dates", "kwh", "support"]:
                row[f"{config_name}_{field}_{side}"] = data.get(field, np.nan)
        support = bool(row[f"{config_name}_support_before"] and row[f"{config_name}_support_after"])
        if support:
            dE = float(row[f"{config_name}_E_after"] - row[f"{config_name}_E_before"])
            dS = float(row[f"{config_name}_S_after"] - row[f"{config_name}_S_before"])
            dT = float(row[f"{config_name}_T_after"] - row[f"{config_name}_T_before"])
            E_star = direction * dE
            S_star = direction * dS
            T_star = direction * dT
            signal, R, H = class_membership(E_star, T_star, 1)
            row.update(
                {
                    f"{config_name}_delta_E": dE,
                    f"{config_name}_delta_S": dS,
                    f"{config_name}_delta_T": dT,
                    f"{config_name}_E_star": E_star,
                    f"{config_name}_S_star": S_star,
                    f"{config_name}_T_star": T_star,
                    f"{config_name}_r_star": T_star - E_star - S_star,
                    f"{config_name}_D30": 30.0 * (T_star - E_star),
                    f"{config_name}_signal_status": signal,
                    f"{config_name}_R": bool(R),
                    f"{config_name}_H": bool(H),
                }
            )
        else:
            for field in ["delta_E", "delta_S", "delta_T", "E_star", "S_star", "T_star", "r_star", "D30"]:
                row[f"{config_name}_{field}"] = np.nan
            for field in ["signal_status", "R", "H"]:
                row[f"{config_name}_{field}"] = np.nan
    return row


def attach_original_membership(full: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["station_id", "month", "boundary_hour", "signal_status", "total_outcome", "E_star", "T_star"]
    p = panel[cols].rename(
        columns={
            "signal_status": "original_signal_status_panel",
            "total_outcome": "original_total_outcome_panel",
            "E_star": "original_E_star_panel",
            "T_star": "original_T_star_panel",
        }
    )
    out = full.merge(p, on=["station_id", "month", "boundary_hour"], how="left", validate="one_to_one")
    out["original_R"] = out["original_signal_status_panel"].eq("signal_bearing")
    out["original_H"] = out["original_R"] & (out["original_T_star_panel"] > 0.03)
    return out


def add_original_kwh(full: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["station_id", "month", "boundary_hour", "K_minus", "K_plus", "N_minus", "N_plus", "n_days_before", "n_days_after"]
    k = panel[cols].rename(
        columns={
            "K_minus": "original_kwh_before",
            "K_plus": "original_kwh_after",
            "N_minus": "original_n_transactions_before",
            "N_plus": "original_n_transactions_after",
            "n_days_before": "original_n_dates_before",
            "n_days_after": "original_n_dates_after",
        }
    )
    overwrite = [
        "original_kwh_before",
        "original_kwh_after",
        "original_n_transactions_before",
        "original_n_transactions_after",
        "original_n_dates_before",
        "original_n_dates_after",
    ]
    full = full.drop(columns=[col for col in overwrite if col in full.columns])
    return full.merge(k, on=["station_id", "month", "boundary_hour"], how="left", validate="one_to_one")


def build_unit_rows(
    pairs: pd.DataFrame,
    layers: pd.DataFrame,
    endpoint_data: dict[tuple[str, str, str, str], GroupData],
    native: pd.DataFrame,
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    layer_lookup = {(str(r["month"]), int(r["boundary_hour"])): r for r in layers.to_dict("records")}
    stage_lookup = stage1_endpoint_lookup(native)
    full_rows: list[dict[str, Any]] = []
    half_rows: list[dict[str, Any]] = []
    for pair in pairs.to_dict("records"):
        layer = layer_lookup[(str(pair["month"]), int(pair["boundary_hour"]))]
        if bool(pair["original_A"]):
            full_rows.append(derive_unit_row(pair, layer, "fullmonth", endpoint_data, stage_lookup))
        if bool(pair["original_A_f"]):
            half_rows.append(derive_unit_row(pair, layer, "formation", endpoint_data, stage_lookup))
        if bool(pair["original_A_v"]):
            half_rows.append(derive_unit_row(pair, layer, "validation", endpoint_data, stage_lookup))
    full = attach_original_membership(pd.DataFrame(full_rows), panel)
    full = add_original_kwh(full, panel)
    return full, pd.DataFrame(half_rows)


def ensure_expected_membership(full: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    expected = {
        "A": int(len(panel)),
        "R": int(panel["signal_status"].eq("signal_bearing").sum()),
        "H": int((panel["signal_status"].eq("signal_bearing") & (panel["T_star"] > 0.03)).sum()),
    }
    observed = {
        "A": int(full["original_A"].sum()),
        "R": int(full["original_R"].sum()),
        "H": int(full["original_H"].sum()),
    }
    if expected != {"A": 8133, "R": 7531, "H": 7479} or observed != expected:
        raise AssertionError(f"original A/R/H reproduction failed: expected {expected}, observed {observed}")
    return {"expected": expected, "observed": observed, "all_match": True}


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


def facility_audit(facility_maps: list[dict[str, Any]], raw_counters: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, set[str]] = defaultdict(set)
    for mapping in facility_maps:
        for pile, stations in mapping.items():
            merged[pile].update(stations)
    multi_station_pile = {pile: sorted(stations) for pile, stations in merged.items() if len(stations) > 1}
    return {
        "schema_version": "module-a-facility-audit-v1",
        "pile_id_column_present": True,
        "pile_id_nonnull_rate": float(raw_counters["pile_id_nonnull_rows"] / raw_counters["rows_seen"]),
        "unique_pile_ids": int(len(merged)),
        "pile_ids_mapped_to_multiple_stations": int(len(multi_station_pile)),
        "multi_station_pile_examples": dict(list(multi_station_pile.items())[:20]),
        "composite_station_pile_key_available": True,
        "within_station_key_rule": "station_id + pile_id",
        "cross_month_identity_semantics_documented": False,
        "facility_subanalysis_status": "conditional_station_pile_sensitivity_available_but_not_primary",
        "reason": "pile_id is non-null and composite station_id+pile_id can be matched within a month. The source schema does not document physical-device continuity across months, so this module does not call the sensitivity a physical-gun estimand.",
    }


def make_price_object_example(full: pd.DataFrame, endpoint_data: dict[tuple[str, str, str, str], GroupData]) -> None:
    """Export a deterministic raw-field-to-price example."""

    eligible = full.loc[
        full["original_A"].astype(bool)
        & (full["original_E_star"].abs() > 0)
        & (full["original_E_before"] > 0)
        & (full["original_E_after"] > 0)
    ].sort_values(["month", "boundary_hour", "station_id"], kind="mergesort")
    if eligible.empty:
        raise AssertionError("no deterministic price-object example candidate")
    row = eligible.iloc[0]
    month, station = str(row["month"]), str(row["station_id"])
    before, after = str(row["before_segment"]), str(row["after_segment"])

    def payload(data: GroupData) -> dict[str, Any]:
        return {
            "n_transactions": int(len(data.E)),
            "n_tariff_dates": int(np.unique(data.tariff_day).size),
            "kwh": float(np.sum(data.kwh)),
            "median_E": q_type7(data.E),
            "median_S": q_type7(data.S),
            "median_T": q_type7(data.T),
            "transaction_fields": {
                "electricity_unit_price": "total_elec_fee / total_elec_kwh",
                "service_unit_price": "total_service_fee / total_elec_kwh",
                "total_unit_price": "total_fee / total_elec_kwh",
                "fee_identity_filter": "abs(total_fee-total_elec_fee-total_service_fee) <= 0.0100001",
            },
        }

    b = endpoint_data[(month, station, before, "fullmonth")]
    a = endpoint_data[(month, station, after, "fullmonth")]
    write_json(
        OUT / "data_price_object_example.json",
        {
            "selection_rule": "sorted month, boundary_hour, station_id; first full-supported unit with non-zero electricity contrast",
            "station_id": station,
            "month": month,
            "boundary_hour": int(row["boundary_hour"]),
            "before_native_segment": before,
            "after_native_segment": after,
            "expected_direction": int(row["expected_direction"]),
            "before": payload(b),
            "after": payload(a),
            "derived_original_price_object": {
                "E_star": float(row["original_E_star"]),
                "S_star": float(row["original_S_star"]),
                "T_star": float(row["original_T_star"]),
                "r_star": float(row["original_r_star"]),
                "D30_RMB_reference": float(row["original_D30"]),
            },
            "interpretation": "Raw transaction fee fields form endpoint unit prices; endpoint medians are differenced across adjacent native segments and multiplied by frozen orientation q.",
        },
    )


def write_results_memo(
    full: pd.DataFrame,
    half: pd.DataFrame,
    covariance: pd.DataFrame,
    checks: dict[str, Any],
) -> None:
    formation_n = int((half["period"].eq("formation") & half["original_support"]).sum())
    validation_n = int((half["period"].eq("validation") & half["original_support"]).sum())
    sensitivity_lines: list[str] = []
    facility_rows_path = OUT / "facility_sensitivity_unit_rows.csv"
    facility_layer_path = OUT / "facility_sensitivity_layer_summary.csv"
    chrono_layer_path = OUT / "chronological23_sensitivity_layer_summary.csv"
    cf_path = OUT / "formation_native8_support.csv"
    if facility_rows_path.exists() and facility_layer_path.exists():
        facility_rows = pd.read_csv(facility_rows_path)
        facility_layer = pd.read_csv(facility_layer_path)
        n_facility = int(facility_layer["n_facility_supported"].sum())
        n_cd = int(facility_layer["n_cd_supported"].sum())
        n_A = int(facility_layer["n_original_A"].sum())
        paired = facility_rows.loc[facility_rows["cd_support"].astype(bool) & facility_rows["facility_support"].astype(bool)]
        med_facility_delta = float(paired["facility_minus_cd_T_star"].abs().median()) if len(paired) else np.nan
        sensitivity_lines.extend(
            [
                f"The bounded date-by-pile sensitivity retains {n_facility:,} of {n_A:,} original-A units ({100*n_facility/n_A:.2f}%) and {len(paired):,} of {n_cd:,} CD-supported units ({100*len(paired)/n_cd:.2f}%) under the 20-transaction/5-date thresholds. On paired units, the median absolute facility-minus-CD total contrast is {med_facility_delta:.7f} RMB/kWh; the layer CSV gives complete coverage and tails. Broad shared-pile and surviving-date CD baselines separate pile filtering from date loss.",
            ]
        )
    if chrono_layer_path.exists():
        chrono_layer = pd.read_csv(chrono_layer_path)
        chrono_parts = []
        for row in chrono_layer.to_dict("records"):
            chrono_parts.append(f"{int(row['n_paired']):,} {row['month']} boundary-23 pairs")
        sensitivity_lines.append("The date audit confirms tariff_day = date(charge_start_time - 23 hours). Main CD keeps the same tariff-day 23:00-to-23:00 cycle label; the supplementary chronological-23 anchor uses 22_23 at tariff_day+1 and 23_07 at tariff_day. It retains " + " and ".join(chrono_parts) + "; main CD is unchanged.")
    if cf_path.exists():
        cf = pd.read_csv(cf_path)
        complete = cf.loc[cf["C_f_station_raw"].astype(bool), ["month", "station_id"]].drop_duplicates()
        counts = complete.groupby("month").size().astype(int).to_dict()
        n_month_station_entries = int(cf[["month", "station_id"]].drop_duplicates().shape[0])
        sensitivity_lines.append(f"The raw all-native-segment formation audit covers {n_month_station_entries:,} month-specific A_f station entries and reproduces complete-menu station counts of {counts} (84 January boundary units and 3,440 July boundary units).")
    lines = [
        "# AE R3 Module A results memo",
        "",
        "CD retains transactions on endpoint-common tariff dates and computes a pooled transaction median. DW keeps exactly the same transactions and dates, assigns equal mass to tariff dates, and equal conditional mass within date. It is not a median of daily medians.",
        "",
        f"Frozen original membership reproduces A/R/H = {checks['original_membership']['observed']['A']:,}/{checks['original_membership']['observed']['R']:,}/{checks['original_membership']['observed']['H']:,}. Full-month rows: {len(full):,}; formation A_f rows: {formation_n:,}; validation A_v rows: {validation_n:,}.",
        "",
        "Full-month summaries are reported for each configuration's own supported sample (fullsample) and on fixed original-A/R/H intersections. Every fixed-member row, including the Original configuration, requires simultaneous original/CD/DW support; the resulting common totals are 7,554 A, 7,047 R and 7,003 H. The common-all-config label is retained as an explicit A-scope alias. Half-month rows are the cross-window handoff and include original/CD/DW endpoint prices plus selected common-date transaction, tariff-date and kWh activity.",
        "",
        f"Covariance component identity failures: {checks['covariance']['component_identity_failures']}; variance identity failures: {checks['covariance']['variance_identity_failures']}.",
        "",
        "The pile audit shows a complete non-null pile_id field and a composite station_id+pile_id key available for within-month sensitivity. The source does not document cross-month physical-device continuity, so a pile-level result is conditional and is not the primary estimand.",
        "",
        *sum(([line, ""] for line in sensitivity_lines), []),
        "The deterministic data-to-price example is saved in data_price_object_example.json. It is selected by sorted month, boundary_hour, station_id order, then by full support and non-zero electricity contrast.",
        "",
        "No figures are generated; the parent analysis owns plotting and manuscript integration.",
    ]
    (OUT / "results_memo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    config, native, panel = load_inputs()
    layers = layer_table(config, panel)
    pairs = supported_pair_keys(native, layers)
    station_codes = station_codes_for_targets(pairs)
    segment_defs = {
        month: [tuple([str(x[0]), int(x[1]), int(x[2]), int(x[3])]) for x in config["native_8_segments"][month]]
        for month in ORDERS
    }
    target_keys_by_month: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in pairs.to_dict("records"):
        target_keys_by_month[row["month"]].add((str(row["station_id"]), str(row["before_segment"])))
        target_keys_by_month[row["month"]].add((str(row["station_id"]), str(row["after_segment"])))

    raw_groups: dict[tuple[str, str, str, str], list[np.ndarray]] = {}
    raw_counters: dict[str, Any] = {}
    facility_maps: list[dict[str, Any]] = []
    calendar_examples: dict[str, Any] = {}
    for month, path in ORDERS.items():
        groups, counters, facility, examples = process_order(
            month,
            path,
            target_keys_by_month[month],
            station_codes[month],
            segment_defs[month],
        )
        raw_groups.update(groups)
        raw_counters[month] = counters
        facility_maps.append(facility)
        calendar_examples[month] = examples
    endpoint_data = finalize_groups(raw_groups)

    full, half = build_unit_rows(pairs, layers, endpoint_data, native, panel)
    membership = ensure_expected_membership(full, panel)

    # Raw full-month medians are compared to the immutable Stage 1 authority at
    # every endpoint feeding original A. This check catches calendar or
    # containment drift before any new summary is written.
    stage_lookup = stage1_endpoint_lookup(native)
    raw_diff_rows: list[dict[str, Any]] = []
    seen_endpoints: set[tuple[str, str, str]] = set()
    for row in full[["station_id", "month", "before_segment", "after_segment"]].to_dict("records"):
        for side, seg_col in [("before", "before_segment"), ("after", "after_segment")]:
            key0 = (str(row["month"]), str(row["station_id"]), str(row[seg_col]))
            if key0 in seen_endpoints:
                continue
            seen_endpoints.add(key0)
            data = endpoint_data[(key0[0], key0[1], key0[2], "fullmonth")]
            authority = stage_lookup[(key0[0], key0[1], key0[2], "all")]["original_fullmonth"]
            for price, values in [("E", data.E), ("S", data.S), ("T", data.T)]:
                raw_diff_rows.append(
                    {
                        "month": key0[0],
                        "station_id": key0[1],
                        "segment_id": key0[2],
                        "price": price,
                        "abs_diff": abs(q_type7(values) - float(authority[price])),
                        "n_raw": int(len(values)),
                    }
                )
    raw_diff_df = pd.DataFrame(raw_diff_rows)
    raw_authority_check = {
        "max_abs_difference": float(raw_diff_df["abs_diff"].max()),
        "n_exceed_1e-12": int((raw_diff_df["abs_diff"] > 1e-12).sum()),
        "n_endpoint_checks": int(len(raw_diff_df)),
    }
    if raw_authority_check["n_exceed_1e-12"]:
        raise AssertionError(f"raw full endpoint medians differ from Stage 1: {raw_authority_check}")

    full = full.sort_values(["layer_order", "station_id"], kind="mergesort").reset_index(drop=True)
    half = half.sort_values(["period", "layer_order", "station_id"], kind="mergesort").reset_index(drop=True)
    full.to_parquet(OUT / "fullmonth_unit_rows.parquet", index=False, compression="zstd")
    half.to_parquet(OUT / "halfmonth_unit_rows.parquet", index=False, compression="zstd")
    full.to_csv(OUT / "fullmonth_unit_rows.csv", index=False, encoding="utf-8")
    half.to_csv(OUT / "halfmonth_unit_rows.csv", index=False, encoding="utf-8")

    # This table intentionally carries only station-month-segment-period
    # statistics. CD/DW are boundary-specific because their common-date set
    # depends on the paired endpoint; their exact values remain in unit rows.
    endpoint_rows: list[dict[str, Any]] = []
    for (month, station, segment, period), data in endpoint_data.items():
        authority = stage_lookup.get((month, station, segment, "all"), {}).get(f"original_{period}", {})
        raw = summary_from_arrays(data.E, data.S, data.T, data.tariff_day, data.kwh, "original")
        if period != "fullmonth":
            authority = {
                "E": raw["E"],
                "S": raw["S"],
                "T": raw["T"],
                "n_transactions": raw["n_transactions"],
                "n_dates": raw["n_dates"],
                "support": authority.get("support", False),
            }
        endpoint_rows.append(
            {
                "station_id": station,
                "month": month,
                "segment_id": segment,
                "period": period,
                "original_E": authority.get("E", np.nan),
                "original_S": authority.get("S", np.nan),
                "original_T": authority.get("T", np.nan),
                "original_n_transactions": authority.get("n_transactions", 0),
                "original_n_dates": authority.get("n_dates", 0),
                "original_support": authority.get("support", False),
                "raw_E": raw["E"],
                "raw_S": raw["S"],
                "raw_T": raw["T"],
                "raw_n_transactions": raw["n_transactions"],
                "raw_n_dates": raw["n_dates"],
                "raw_kwh": raw["kwh"],
            }
        )
    endpoint_df = pd.DataFrame(endpoint_rows).sort_values(
        ["month", "station_id", "segment_id", "period"], kind="mergesort"
    )
    endpoint_df.to_parquet(OUT / "endpoint_aggregates.parquet", index=False, compression="zstd")
    endpoint_df.to_csv(OUT / "endpoint_aggregates.csv", index=False, encoding="utf-8")

    layer_summaries: list[pd.DataFrame] = []
    covariance_summaries: list[pd.DataFrame] = []
    for config_name in ["original", "cd", "dw"]:
        for scope in ["fullsample", "common_original_A", "common_original_R", "common_original_H", "common_all_configs"]:
            layer_rows, cov_rows = summarize_layer(full, config_name, scope, period="fullmonth")
            layer_summaries.append(layer_rows)
            covariance_summaries.append(cov_rows)
    layer_summary = pd.concat(layer_summaries, ignore_index=True)
    covariance = pd.concat(covariance_summaries, ignore_index=True)
    scope_audit = fixed_member_scope_audit(layer_summary)
    layer_summary.to_csv(OUT / "fullmonth_layer_summaries.csv", index=False, encoding="utf-8")
    covariance.to_csv(OUT / "fullmonth_covariance_decomposition.csv", index=False, encoding="utf-8")
    scope_audit.to_csv(OUT / "fixed_member_scope_audit.csv", index=False, encoding="utf-8")
    transitions = transition_table(full)
    transitions.to_csv(OUT / "fullmonth_reclassification_transitions.csv", index=False, encoding="utf-8")

    half_summaries: list[pd.DataFrame] = []
    for period in ["formation", "validation"]:
        for config_name in ["original", "cd", "dw"]:
            rows, _ = summarize_layer(half, config_name, "fullsample", period=period)
            half_summaries.append(rows)
    pd.concat(half_summaries, ignore_index=True).to_csv(OUT / "halfmonth_layer_summaries.csv", index=False, encoding="utf-8")

    make_price_object_example(full, endpoint_data)

    raw_all = {
        month: {k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in values.items()}
        for month, values in raw_counters.items()
    }
    total_raw_counters = {
        key: sum(int(values.get(key, 0)) for values in raw_counters.values())
        for key in ["pile_id_nonnull_rows", "rows_seen"]
    }
    facility = facility_audit(facility_maps, total_raw_counters)
    checks = {
        "schema_version": "module-a-v1",
        "status": "PASS",
        "runtime": {
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "pyarrow": package_version("pyarrow"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
        "original_membership": membership,
        "input_source_authority": {
            "config_stage2": str(CONFIG_PATH),
            "stage1_prices": str(STAGE1_PATH),
            "original_panel": str(ORIGINAL_PANEL_PATH),
            "orders": {month: str(path) for month, path in ORDERS.items()},
            "raw_columns_read": RAW_COLUMNS,
        },
        "raw_counters": raw_all,
        "raw_full_endpoint_authority_check": raw_authority_check,
        "calendar_examples": calendar_examples,
        "thresholds": {
            "fullmonth": {"transactions_each_side": 20, "tariff_dates_each_side": 5},
            "halfmonth": {"transactions_each_side": 10, "tariff_dates_each_side": 3},
            "electricity_qualification": 0.01,
            "total_retention": 0.03,
            "D30_band": 0.90,
        },
        "median_rule": {
            "pooled_cd": "ordinary transaction median with linear midpoint for even n",
            "date_weighted_dw": "equal tariff-date mass; equal conditional transaction mass within date",
            "exact_mass_half": "merge equal values; adjacent distinct-value midpoint within 1e-12",
            "daily_median_of_medians": False,
        },
        "counts": {
            "fullmonth_unit_rows": int(len(full)),
            "halfmonth_unit_rows": int(len(half)),
            "formation_A_f": int((half["period"].eq("formation") & half["original_support"]).sum()),
            "validation_A_v": int((half["period"].eq("validation") & half["original_support"]).sum()),
            "fullmonth_original_A": int(full["original_A"].sum()),
            "fullmonth_original_R": int(full["original_R"].sum()),
            "fullmonth_original_H": int(full["original_H"].sum()),
        },
        "covariance": {
            "rows": int(len(covariance)),
            "component_identity_failures": int((~covariance["cov_identity_pass"]).sum()),
            "variance_identity_failures": int((~covariance["variance_identity_pass"]).sum()),
            "max_component_identity_abs": float(covariance["cov_identity_residual"].abs().max()),
            "max_variance_identity_abs": float(covariance["variance_identity_residual"].abs().max()),
        },
        "fixed_member_scope": {
            "contract": "fullsample is configuration-specific; every common_original_* scope uses original membership plus original/CD/DW support intersection for all configurations",
            "counts": scope_audit.to_dict("records"),
            "expected_common_totals": {
                "common_original_A": 7554,
                "common_original_R": 7047,
                "common_original_H": 7003,
            },
        },
        "facility": facility,
        "sensitivity_outputs": {
            "facility_unit_rows": str((OUT / "facility_sensitivity_unit_rows.csv")) if (OUT / "facility_sensitivity_unit_rows.csv").exists() else None,
            "facility_layer_summary": str((OUT / "facility_sensitivity_layer_summary.csv")) if (OUT / "facility_sensitivity_layer_summary.csv").exists() else None,
            "chronological23_unit_rows": str((OUT / "chronological23_sensitivity_unit_rows.csv")) if (OUT / "chronological23_sensitivity_unit_rows.csv").exists() else None,
            "chronological23_layer_summary": str((OUT / "chronological23_sensitivity_layer_summary.csv")) if (OUT / "chronological23_sensitivity_layer_summary.csv").exists() else None,
            "formation_native8_support": str((OUT / "formation_native8_support.csv")) if (OUT / "formation_native8_support.csv").exists() else None,
            "audit": str((OUT / "facility_sensitivity_audit.json")) if (OUT / "facility_sensitivity_audit.json").exists() else None,
        },
        "output_schema": {
            "fullmonth_unit_rows": "one row per original A station-month-boundary unit; original/CD/DW endpoint E/S/T, support, R/H, D30, residual",
            "halfmonth_unit_rows": "one row per formation or validation supported pair; original/CD/DW endpoint prices and common-date n/date/kWh activity",
            "endpoint_aggregates": "one row per station-month-segment-period with raw and original endpoint statistics; CD/DW remain boundary-specific in unit rows",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(
        OUT / "config_used.json",
        {
            "module": "AE R3 Module A",
            "specification": "revision_records/ANALYSIS_SPEC.md",
            "author_instructions": "revision_records/AUTHOR_INSTRUCTIONS_ZH.md",
            "raw_source_hashes": {month: sha256_file(path) for month, path in ORDERS.items()},
            "input_hashes": {
                "config_stage2": sha256_file(CONFIG_PATH),
                "stage1_prices": sha256_file(STAGE1_PATH),
                "original_panel": sha256_file(ORIGINAL_PANEL_PATH),
            },
            "layers": layers.to_dict("records"),
            "support_thresholds": config["support_thresholds"],
            "native_8_segments": config["native_8_segments"],
            "median_rule": checks["median_rule"],
        },
    )
    write_json(OUT / "audit.json", checks)
    write_json(OUT / "facility_audit.json", facility)
    write_results_memo(full, half, covariance, checks)
    raw_diff_df.to_csv(OUT / "raw_full_endpoint_authority_diffs.csv", index=False, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": checks["status"],
                "elapsed_seconds": checks["elapsed_seconds"],
                "A_R_H": [
                    checks["counts"]["fullmonth_original_A"],
                    checks["counts"]["fullmonth_original_R"],
                    checks["counts"]["fullmonth_original_H"],
                ],
                "formation_A_f": checks["counts"]["formation_A_f"],
                "validation_A_v": checks["counts"]["validation_A_v"],
                "raw_full_endpoint_max_abs_difference": raw_authority_check["max_abs_difference"],
                "output_dir": str(OUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Raw fixed-support reconstruction audit. Use scripts/reproduce_prices.py for cache generation.")
    parser.add_argument('--package-root',type=Path,default=PACKAGE_ROOT)
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    ROOT=args.package_root.resolve();OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
    CONFIG_PATH=ROOT/'config/price_construction.json'
    STAGE1_PATH=args.cache_dir.resolve()/'native_prices.parquet'
    ORIGINAL_PANEL_PATH=ROOT/'data/derived/prices/frozen_original_panel.parquet'
    ORDERS={m:args.raw_dir.resolve()/f'orders_{m}_public.parquet' for m in ['2025-01','2025-07']}
    main()
