"""Re-aggregate frozen UrbanEV E2 draws with correct small-n reporting.

This is a reporting-only step.  It reads the 900 frozen draw CSVs, does not
read raw price files, and does not generate a mask.  The only changed summary
fields permitted by the audit are ``*_sd`` and ``*_mcse`` when exactly one
valid draw exists: both are undefined (NA), while the point values and valid
draw count remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DRAWS = PROJECT_ROOT / "outputs/09_price_architecture/urbanev_external_analysis_20260913/e2/draws"
DEFAULT_OLD_SUMMARY = PROJECT_ROOT / "outputs/09_price_architecture/urbanev_external_analysis_20260913/e2/scenario_summary.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs/09_price_architecture/focused_manuscript_revision_20260914/analysis_results/e2_reporting"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs/09_price_architecture/focused_manuscript_revision_20260914/working_source/tables"

SUMMARY_METRICS = (
    "n_reference_units",
    "n_survivors",
    "availability",
    "t_star_mae",
    "dabs_mae",
    "reference_all_mean_dabs",
    "reference_survivor_mean_dabs",
    "estimate_survivor_mean_dabs",
    "selection_bias_dabs",
    "recovery_bias_dabs",
    "total_bias_dabs",
    "eligibility_n_common",
    "eligibility_misclassification_rate",
    "survivor_days_mean",
    "actual_deleted_fraction_valid_nodes",
    "actual_deleted_fraction_valid_days",
)
SUMMARY_SUFFIXES = ("mean", "sd", "mcse", "p05", "p95", "n_valid_draws")
MECHANISM_ORDER = {"endpoint": 0, "continuous_days": 1, "price_period": 2}
METHOD_ORDER = {"local": 0, "complete": 1}
GROUP_ORDER = {"all": 0, "dynamic_ref": 1, "flat_ref": 2}
MECHANISM_CODES = {"endpoint": "EP", "continuous_days": "CD", "price_period": "PS"}
METHOD_CODES = {"local": "L", "complete": "C"}
GROUP_CODES = {"all": "all", "dynamic_ref": "dyn", "flat_ref": "flat"}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "n/a", "none"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return None if not math.isfinite(parsed) else parsed


def as_int(value: Any) -> int | None:
    parsed = as_float(value)
    return None if parsed is None else int(round(parsed))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def linear_percentile(values: list[float], percentile: float) -> float:
    """Match NumPy's default linear percentile interpolation."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summary_stats(values: Iterable[Any]) -> dict[str, float | int]:
    """Return valid-draw summaries, leaving SD and MCSE undefined for n=1."""
    numeric = []
    for value in values:
        parsed = as_float(value)
        if parsed is not None:
            numeric.append(parsed)
    n = len(numeric)
    result: dict[str, float | int] = {
        "mean": math.nan,
        "sd": math.nan,
        "mcse": math.nan,
        "p05": math.nan,
        "p95": math.nan,
        "n_valid_draws": n,
    }
    if n == 0:
        return result
    mean = sum(numeric) / n
    result.update({"mean": mean, "p05": linear_percentile(numeric, 5.0), "p95": linear_percentile(numeric, 95.0)})
    if n >= 2:
        variance = sum((value - mean) ** 2 for value in numeric) / (n - 1)
        sd = math.sqrt(variance)
        result.update({"sd": sd, "mcse": sd / math.sqrt(n)})
    return result


def summary_key(row: Mapping[str, Any]) -> tuple[str, float, str, str]:
    return (str(row["mechanism"]), float(row["p"]), str(row["method"]), str(row["group"]))


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, float, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[summary_key(row)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda item: (MECHANISM_ORDER[item[0]], item[1], METHOD_ORDER[item[2]], GROUP_ORDER[item[3]])):
        mechanism, p, method, group = key
        bucket = buckets[key]
        summary: dict[str, Any] = {
            "mechanism": mechanism,
            "p": p,
            "method": method,
            "group": group,
            "n_repetitions": len(bucket),
            "summary_note": "MCSE and p05/p95 describe Monte Carlo draws; they are not confidence intervals or coverage rates.",
        }
        for metric in SUMMARY_METRICS:
            stats = summary_stats(row.get(metric) for row in bucket)
            for suffix in SUMMARY_SUFFIXES:
                summary[f"{metric}_{suffix}"] = stats[suffix]
        output.append(summary)
    return output


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return str(value)


def value_equal(left: Any, right: Any) -> bool:
    left_float, right_float = as_float(left), as_float(right)
    if left_float is None or right_float is None:
        left_missing = left is None or str(left).strip().lower() in {"", "nan", "na", "n/a"}
        right_missing = right is None or str(right).strip().lower() in {"", "nan", "na", "n/a"}
        if not left_missing and not right_missing and left_float is None and right_float is None:
            return str(left) == str(right)
        return left_missing and right_missing
    return math.isclose(left_float, right_float, rel_tol=1e-10, abs_tol=1e-12)


def audit_against_old(old_rows: list[dict[str, str]], computed_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    old_map = {summary_key(row): row for row in old_rows}
    computed_map = {summary_key(row): row for row in computed_rows}
    if set(old_map) != set(computed_map):
        raise RuntimeError(f"summary key mismatch: old={len(old_map)} computed={len(computed_map)}")
    changed: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    corrected: list[dict[str, Any]] = []
    fields = list(old_rows[0])
    for key in sorted(old_map, key=lambda item: (MECHANISM_ORDER[item[0]], item[1], METHOD_ORDER[item[2]], GROUP_ORDER[item[3]])):
        old = old_map[key]
        computed = computed_map[key]
        row = dict(old)
        for field in fields:
            new_value = computed.get(field, "")
            if field.endswith(("_sd", "_mcse")):
                metric = field.rsplit("_", 1)[0]
                n_valid = as_int(computed.get(f"{metric}_n_valid_draws"))
                if n_valid == 1:
                    row[field] = "NaN"
                    if not value_equal(old.get(field), row[field]):
                        changed.append({"key": list(key), "field": field, "old": old.get(field), "new": "NaN", "reason": "n_valid=1; sample SD and MCSE undefined"})
                    continue
            if not value_equal(old.get(field), new_value):
                unexpected.append({"key": list(key), "field": field, "old": old.get(field), "computed": csv_value(new_value)})
        corrected.append(row)
    return corrected, changed, unexpected


def fmt_int(value: Any) -> str:
    parsed = as_int(value)
    return "NA" if parsed is None else f"{parsed:,}"


def fmt_pct(value: Any, digits: int = 4) -> str:
    parsed = as_float(value)
    return "NA" if parsed is None else f"{100.0 * parsed:.{digits}f}\\%"


def fmt_num(value: Any, digits: int = 6) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "NA"
    if parsed == 0:
        return "0"
    if abs(parsed) < 1e-4:
        exponent = int(math.floor(math.log10(abs(parsed))))
        mantissa = parsed / (10**exponent)
        return f"{mantissa:.3g}e{exponent:+d}"
    return f"{parsed:.{digits}f}".rstrip("0").rstrip(".")


def fmt_signed(value: Any) -> str:
    return fmt_num(value, digits=5)


def sorted_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: (MECHANISM_ORDER[row["mechanism"]], float(row["p"]), METHOD_ORDER[row["method"]], GROUP_ORDER[row["group"]]))


def table_key_cells(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        MECHANISM_CODES[str(row["mechanism"])],
        f"{100.0 * float(row['p']):.0f}\\%",
        METHOD_CODES[str(row["method"])],
        GROUP_CODES[str(row["group"])],
    )


def write_tables(rows: list[dict[str, str]], table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    availability_body = []
    offsets_body = []
    for row in sorted_summary_rows(rows):
        cells = table_key_cells(row)
        availability_body.append(
            " & ".join(cells + (fmt_pct(row["availability_mean"]), fmt_pct(row["availability_mcse"]), fmt_num(row["dabs_mae_mean"]), fmt_num(row["dabs_mae_mcse"]), fmt_int(row["dabs_mae_n_valid_draws"]))) + r" \\"
        )
        offsets_body.append(
            " & ".join(cells + (fmt_signed(row["selection_bias_dabs_mean"]), fmt_signed(row["recovery_bias_dabs_mean"]), fmt_signed(row["total_bias_dabs_mean"]))) + r" \\"
        )
    masking = r"""\begingroup
\footnotesize
\setlength{\LTleft}{\fill}
\setlength{\LTright}{\fill}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.04}
\begin{longtable}{@{}p{0.10\textwidth}r p{0.07\textwidth}p{0.10\textwidth}rrrrr@{}}
\caption{UrbanEV E2 artificial masking: availability and conditional amount error.}\label{tab:urban-e2-masking}\\
\toprule
Mechanism & $p$ & Rule & Group & Avail. (\%) & Avail. MCSE (\%) & $|D_{30}|$ MAE & MAE MCSE & $n_{\rm valid}$ \\
\midrule
\endfirsthead
\multicolumn{9}{c}{\tablename\ \thetable{} continued from previous page} \\
\toprule
Mechanism & $p$ & Rule & Group & Avail. (\%) & Avail. MCSE (\%) & $|D_{30}|$ MAE & MAE MCSE & $n_{\rm valid}$ \\
\midrule
\endhead
\midrule
\multicolumn{9}{r}{Continued on next page} \\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(availability_body) + r"""
\end{longtable}

\begin{minipage}{0.95\textwidth}\footnotesize
Codes: EP = independent endpoint deletion, CD = contiguous calendar-day
deletion, and PS = price/period-selective deletion; L = local support and C =
complete-grid support; all, dyn, and flat denote the all-reference,
dynamic-reference, and flat-reference groups. Every mechanism--rate scenario
has 100 masks. Availability is the mean survivor/reference ratio across all
masks; MCSE is shown in the next column. Amount MAE is conditional on a
surviving unit set, and $n_{\rm valid}$ is the number of masks contributing
that MAE. A zero availability is a real zero; an undefined survivor MAE is
\textit{NA}, not zero. When $n_{\rm valid}=1$, the point MAE is retained but
its sample SD and MCSE are undefined and shown as \textit{NA}. MCSE and the
fifth--ninety-fifth draw percentiles describe imposed masking randomness, not
confidence intervals, population
intervals, or coverage rates.
\end{minipage}
\endgroup
"""
    offsets = r"""\begingroup
\footnotesize
\setlength{\LTleft}{\fill}
\setlength{\LTright}{\fill}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.04}
\begin{longtable}{@{}p{0.10\textwidth}r p{0.07\textwidth}p{0.10\textwidth}rrr@{}}
\caption{UrbanEV E2 artificial masking: selection, recovery, and total mean offsets.}\label{tab:urban-e2-offsets}\\
\toprule
Mechanism & $p$ & Rule & Group & Selection & Recovery & Total \\
\midrule
\endfirsthead
\multicolumn{7}{c}{\tablename\ \thetable{} continued from previous page} \\
\toprule
Mechanism & $p$ & Rule & Group & Selection & Recovery & Total \\
\midrule
\endhead
\midrule
\multicolumn{7}{r}{Continued on next page} \\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(offsets_body) + r"""
\end{longtable}

\begin{minipage}{0.95\textwidth}\footnotesize
Notes: Selection, recovery, and total are means in RMB on the 30-kWh scale.
They satisfy the accounting identity defined in Equation~\eqref{eq:urban-offset};
totals use the two components before rounding. Offset values are \textit{NA}
when no valid survivor estimate exists. Local and complete rules use the same
paired-date estimator on common units; equality on common units is a design
identity, not recovery validation.
\end{minipage}
\endgroup
"""
    (table_dir / "urbanev_e2_masking.tex").write_text(masking, encoding="utf-8")
    (table_dir / "urbanev_e2_offsets.tex").write_text(offsets, encoding="utf-8")


def run(draws_dir: Path, old_summary: Path, out_dir: Path, table_dir: Path, expected_draws: int = 900) -> dict[str, Any]:
    draw_paths = sorted(draws_dir.glob("draw_*.csv"))
    if len(draw_paths) != expected_draws:
        raise RuntimeError(f"expected {expected_draws} frozen draw files, found {len(draw_paths)}")
    draw_rows: list[dict[str, str]] = []
    for draw_path in draw_paths:
        rows = read_csv(draw_path)
        if len(rows) != 6:
            raise RuntimeError(f"{draw_path.name}: expected 6 rows, found {len(rows)}")
        draw_rows.extend(rows)
    computed = aggregate(draw_rows)
    old_rows = read_csv(old_summary)
    corrected, changed, unexpected = audit_against_old(old_rows, computed)
    if unexpected:
        raise RuntimeError(f"unexpected reaggregation changes: {json.dumps(unexpected[:3], ensure_ascii=False)}")
    fieldnames = list(old_rows[0])
    write_csv(out_dir / "scenario_summary.csv", corrected, fieldnames)
    write_tables(corrected, table_dir)
    n_one = [
        {"key": list(summary_key(row)), "metric": metric}
        for row in computed
        for metric in SUMMARY_METRICS
        if as_int(row.get(f"{metric}_n_valid_draws")) == 1
    ]
    n_zero = [
        {"key": list(summary_key(row)), "metric": metric}
        for row in computed
        for metric in SUMMARY_METRICS
        if as_int(row.get(f"{metric}_n_valid_draws")) == 0
    ]
    comparison = {
        "schema": "urbanev_e2_reporting_fix_v1",
        "source_draw_dir": str(draws_dir),
        "old_summary": str(old_summary),
        "draw_files": len(draw_paths),
        "draw_rows": len(draw_rows),
        "summary_rows": len(computed),
        "changed_fields": changed,
        "unexpected_changes": unexpected,
        "only_expected_changes": not unexpected and all(item["field"].endswith(("_sd", "_mcse")) for item in changed),
        "n_valid_zero_cells": n_zero,
        "n_valid_one_cells": n_one,
        "n_valid_at_least_two_cells": sum(
            1
            for row in computed
            for metric in SUMMARY_METRICS
            if (as_int(row.get(f"{metric}_n_valid_draws")) or 0) >= 2
        ),
        "rule": "n=0: all summary statistics undefined; n=1: point mean and percentiles retained, sample SD and MCSE NA; n>=2: sample SD (ddof=1) and SD/sqrt(n).",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema": "urbanev_e2_reporting_manifest_v1",
        "status": "COMPLETE",
        "draws_sha256": {path.name: sha256(path) for path in draw_paths},
        "old_summary_sha256": sha256(old_summary),
        "corrected_summary_sha256": sha256(out_dir / "scenario_summary.csv"),
        "masking_table_sha256": sha256(table_dir / "urbanev_e2_masking.tex"),
        "offsets_table_sha256": sha256(table_dir / "urbanev_e2_offsets.tex"),
        "comparison": "comparison.json",
        "new_masks_generated": False,
        "raw_price_files_read": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws-dir", type=Path, default=DEFAULT_DRAWS)
    parser.add_argument("--old-summary", type=Path, default=DEFAULT_OLD_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--expected-draws", type=int, default=900)
    args = parser.parse_args()
    comparison = run(args.draws_dir, args.old_summary, args.out_dir, args.table_dir, args.expected_draws)
    print(json.dumps({"status": "COMPLETE", "changed_fields": len(comparison["changed_fields"]), "n_valid_zero_cells": len(comparison["n_valid_zero_cells"]), "n_valid_one_cells": len(comparison["n_valid_one_cells"]), "unexpected_changes": len(comparison["unexpected_changes"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
