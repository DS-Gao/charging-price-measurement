"""Build tables and vector figures from focused revision extension outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs" / "09_price_architecture"
DEFAULT_OUT = BASE / "focused_manuscript_revision_20260914"
EXT_REL = Path("analysis_results") / "extensions"
TABLE_REL = Path("working_source") / "tables"
FIGURE_REL = Path("working_source") / "figures"

BLUE = "#245778"
GOLD = "#BF8433"
GREEN = "#47866D"
GRAY = "#8A9299"
RED = "#B94A48"


def read_outputs(out: Path) -> dict[str, pd.DataFrame]:
    ext = out / EXT_REL
    return {
        "summary": pd.read_csv(ext / "summary_short.csv"),
        "strata": pd.read_csv(ext / "monthly_boundary_strata.csv"),
        "threshold": pd.read_csv(ext / "threshold_sensitivity.csv"),
        "audit3": pd.read_csv(ext / "input_audit_3x3.csv"),
        "auditc": pd.read_csv(ext / "input_audit_C.csv"),
        "transition": pd.read_csv(ext / "residual_classification_transition.csv"),
        "residual_summary": pd.read_csv(ext / "residual_classification_summary.csv"),
        "units": pd.read_csv(
            ext / "unit_level_signed.csv",
            dtype={"station_id": "string"},
            low_memory=False,
        ),
    }


def fmt_int(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "--"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    # Avoid rendering a rounded negative zero (for example -0.00).
    if round(value, digits) == 0:
        value = 0.0
    text = f"{value:.{digits}f}"
    return text


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    return f"{100 * value:.{digits}f}\\%"


def tex_escape(value: Any) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def write_tex(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_main_table(summary: pd.DataFrame) -> str:
    support_rows = []
    outcome_rows = []
    for row in summary.to_dict("records"):
        support_rows.append(
            " & ".join(
                [
                    tex_escape(row["system"]),
                    fmt_int(row["A_n"]),
                    fmt_int(row["R_n"]),
                    fmt_int(row["H_n"]),
                    fmt_int(row["C_n"]),
                    fmt_pct(row["R_over_A"], 1),
                    fmt_pct(row["H_over_R"], 1),
                    fmt_pct(row["C_over_A"], 1),
                ]
            )
            + r" \\"
        )
        outcome_rows.append(
            " & ".join(
                [
                    tex_escape(row["system"]),
                    fmt_int(row["R_retained_n"]),
                    fmt_int(row["R_neutralized_n"]),
                    fmt_int(row["R_reversed_n"]),
                    fmt_int(row["H_negative_n"]),
                    fmt_int(row["H_near_n"]),
                    fmt_int(row["H_positive_n"]),
                    fmt_num(row["H_signed_median_RMB"], 4),
                    fmt_num(row["H_absolute_median_RMB"], 2),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Support and signed total-price profiles in the two observation systems.}}
\label{{tab:revision-signed-main}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\textit{{Panel A: support sets}}\\[-2pt]
\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\toprule
System & $A$ & $R$ & $H$ & $C$ & $R/A$ & $H/R$ & $C/A$\\
\midrule
{chr(10).join(support_rows)}
\bottomrule
\end{{tabular}}

\vspace{{6pt}}
\textit{{Panel B: direction and signed amount}}\\[-2pt]
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
\toprule
System & \multicolumn{{3}}{{c}}{{Total outcome in $R$}} &
\multicolumn{{3}}{{c}}{{Signed $D_{{30}}$ in $H$}} &
\shortstack{{signed\\median}} & \shortstack{{absolute\\median}}\\
\cmidrule(lr){{2-4}}\cmidrule(lr){{5-7}}
 & Retained & Neutralized & Reversed & Negative & Near & Positive &
\multicolumn{{2}}{{c}}{{RMB}}\\
\midrule
{chr(10).join(outcome_rows)}
\bottomrule
\end{{tabular}}

\vspace{{3pt}}
\begin{{minipage}}{{0.97\textwidth}}\footnotesize
Notes: $A$ is the locally supported unit set; $R$ applies $E^*>0.01$
RMB/kWh; $H$ applies $T^*>0.03$ within $R$.  $C$ is the complete-menu
subset within $A$ (native menu for Beijing and complete nine-endpoint support
for UrbanEV).  $D_{{30}}=30(T^*-E^*)$ RMB.  The three signed cells use
$D_{{30}}<-0.90$, $|D_{{30}}|\leq0.90$, and $D_{{30}}>0.90$ RMB,
respectively.  Rows are equal-weighted units within each observation system.
Panel A reports set membership and Panel B reports the two conditional
category profiles; the panels use the same frozen units.
\end{{minipage}}
\end{{table}}
"""


def build_strata_table(strata: pd.DataFrame) -> str:
    support_rows = []
    amount_rows = []
    for row in strata.to_dict("records"):
        support_rows.append(
            " & ".join(
                [
                    tex_escape(row["system"]),
                    tex_escape(row["month"]),
                    tex_escape(row["boundary"]),
                    fmt_int(row["A_n"]),
                    fmt_int(row["R_n"]),
                    fmt_int(row["H_n"]),
                    fmt_int(row["C_n"]),
                    fmt_int(row["R_retained_n"]),
                    fmt_int(row["R_neutralized_n"]),
                    fmt_int(row["R_reversed_n"]),
                ]
            )
            + r" \\"
        )
        amount_rows.append(
            " & ".join(
                [
                    tex_escape(row["system"]),
                    tex_escape(row["month"]),
                    tex_escape(row["boundary"]),
                    fmt_int(row["H_negative_n"]),
                    fmt_int(row["H_near_n"]),
                    fmt_int(row["H_positive_n"]),
                    fmt_num(row["H_signed_median_RMB"], 2),
                    fmt_num(row["H_absolute_median_RMB"], 2),
                    fmt_int(row["H_abs_gt_amount_n"]),
                ]
            )
            + r" \\"
        )
    return rf"""
\begingroup
\footnotesize
\linespread{{1}}\selectfont
\setlength{{\tabcolsep}}{{3pt}}
\begin{{longtable}}{{@{{}}lllrrrrrrr@{{}}}}
\caption{{Complete month--boundary support and direction stratification for the signed extension.}}
\label{{tab:revision-signed-strata}}\\
\toprule
System & Month & Boundary & $A$ & $R$ & $H$ & $C$ &
Ret. & Neu. & Rev.\\
\midrule
\endfirsthead
\toprule
System & Month & Boundary & $A$ & $R$ & $H$ & $C$ &
Ret. & Neu. & Rev.\\
\midrule
\endhead
{chr(10).join(support_rows)}
\bottomrule
\end{{longtable}}
\noindent\parbox{{\textwidth}}{{\raggedright Notes: Ret./Neu./Rev. are
retained/neutralized/reversed total outcomes within $R$.  Boundary labels are
local clock times; the underlying Beijing hour or UrbanEV minute is retained
in the companion CSV.\par}}
\par

\begin{{longtable}}{{@{{}}lllrrrrrr@{{}}}}
\caption{{Complete month--boundary signed amount stratification for the signed extension.}}
\label{{tab:revision-signed-strata-amount}}\\
\toprule
System & Month & Boundary &
\multicolumn{{3}}{{c}}{{Signed $D_{{30}}$ in $H$}} &
\shortstack{{signed\\median}} & \shortstack{{absolute\\median}} &
\shortstack{{$|D_{{30}}|>0.90$\\count}}\\
\cmidrule(lr){{4-6}}
 & & & Negative & Near & Positive & \multicolumn{{2}}{{c}}{{RMB}} & $n$\\
\midrule
\endfirsthead
\toprule
System & Month & Boundary &
\multicolumn{{3}}{{c}}{{Signed $D_{{30}}$ in $H$}} &
\shortstack{{signed\\median}} & \shortstack{{absolute\\median}} &
\shortstack{{$|D_{{30}}|>0.90$\\count}}\\
\midrule
\endhead
{chr(10).join(amount_rows)}
\bottomrule
\end{{longtable}}
\noindent\parbox{{\textwidth}}{{\raggedright Notes: Negative, near and positive
use $D_{{30}}<-0.90$, $|D_{{30}}|\leq0.90$, and $D_{{30}}>0.90$ RMB.
The signed and absolute medians are conditional on $H$; zero-count cells
remain in the table.\par}}
\par
\endgroup
"""


def build_threshold_table(threshold: pd.DataFrame) -> str:
    rows = []
    factor_names = {
        "electricity_qualification": "Electricity qualification",
        "total_direction": "Total direction",
        "amount_threshold_fixed_primary_H": "Amount on fixed primary $H$",
    }
    for row in threshold.to_dict("records"):
        value = (
            f"{float(row['electricity_threshold_RMB_per_kWh']):g}"
            if row["factor"] == "electricity_qualification"
            else f"{float(row['total_threshold_RMB_per_kWh']):g}"
            if row["factor"] == "total_direction"
            else f"{float(row['amount_threshold_RMB']):g} RMB"
        )
        rows.append(
            " & ".join(
                [
                    factor_names.get(row["factor"], tex_escape(row["factor"])),
                    tex_escape(value),
                    fmt_int(row["A_n"]),
                    fmt_int(row["R_n"]),
                    fmt_int(row["H_n"]),
                    fmt_int(row["C_n"]),
                    fmt_int(row["R_retained_n"]),
                    fmt_int(row["R_neutralized_n"]),
                    fmt_int(row["R_reversed_n"]),
                    fmt_int(row["H_negative_n"]),
                    fmt_int(row["H_near_n"]),
                    fmt_int(row["H_positive_n"]),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[p]
\centering
\caption{{One-factor threshold sensitivity for UrbanEV.}}
\label{{tab:revision-threshold-sensitivity}}
\scriptsize
\setlength{{\tabcolsep}}{{2.8pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}llrrrrrrrrrr@{{}}}}
\toprule
Factor & Value & $A$ & $R$ & $H$ & $C$ &
\multicolumn{{3}}{{c}}{{Total outcome in $R$}} &
\multicolumn{{3}}{{c}}{{Signed $D_{{30}}$ in $H$}}\\
\cmidrule(lr){{7-9}}\cmidrule(lr){{10-12}}
 & & & & & & Ret. & Neu. & Rev. & Neg. & Near & Pos.\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}

\vspace{{3pt}}
\begin{{minipage}}{{0.97\textwidth}}\footnotesize
Notes: This is a post-review specified sensitivity: one factor changes at a time and all other primary settings remain
fixed.  Electricity and total thresholds are in RMB/kWh.  Amount rows use
RMB thresholds on the fixed primary $H$ ($0.60$, $0.90$, and $1.50$ RMB);
they do not reselect $H$.  Counts are equal-unit descriptive summaries.
\end{{minipage}}
\end{{table}}
"""


def build_audit_3x3(audit: pd.DataFrame) -> str:
    rows = []
    for row in audit.to_dict("records"):
        rows.append(
            " & ".join(
                [
                    tex_escape(row["system"]),
                    tex_escape(row["E_label"]),
                    tex_escape(row["T_label"]),
                    fmt_int(row["n"]),
                    fmt_pct(row["share_of_A"], 1),
                    fmt_num(row["D30_signed_median_RMB"], 2),
                    fmt_num(row["D30_absolute_median_RMB"], 2),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[p]
\centering
\caption{{Embedded price-input audit on each system's supported $A$ units.}}
\label{{tab:revision-input-audit}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.06}}
\begin{{tabular}}{{@{{}}lllr rrr@{{}}}}
\toprule
System & $E^*$ label & $T^*$ label & $n$ & Share of $A$ &
\shortstack{{signed $D_{{30}}$\\median}} &
\shortstack{{absolute $D_{{30}}$\\median}}\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}

\vspace{{3pt}}
\begin{{minipage}}{{0.97\textwidth}}\footnotesize
Notes: Both price objects use the same $\kappa=0.03$ RMB/kWh:
negative $<-\kappa$, neutral $[-\kappa,\kappa]$, and positive $>\kappa$.
The matrix is descriptive label correspondence, not a prediction or truth
comparison.  Medians are in reference RMB.
\end{{minipage}}
\end{{table}}
"""


def build_audit_c(auditc: pd.DataFrame) -> str:
    rows = []
    for row in auditc.loc[auditc["set_label"].isin(["A", "C"])].to_dict("records"):
        rows.append(
            " & ".join(
                [
                    tex_escape(row["system"]),
                    tex_escape(row["set_label"]),
                    fmt_int(row["n"]),
                    fmt_pct(row["share_of_A"], 1),
                    fmt_int(row["R_n_primary"]),
                    fmt_int(row["H_n_primary"]),
                    fmt_num(row["D30_signed_median_RMB"], 2),
                    fmt_num(row["D30_absolute_median_RMB"], 2),
                ]
            )
            + r" \\"
        )
    return rf"""
\begin{{table}}[t]
\centering
\caption{{Support-membership audit for the complete-menu subset.}}
\label{{tab:revision-input-audit-C}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\begin{{tabular}}{{@{{}}llrrrrrr@{{}}}}
\toprule
System & Set & $n$ & Share of $A$ & $R$ & $H$ &
\shortstack{{signed $D_{{30}}$\\median}} &
\shortstack{{absolute $D_{{30}}$\\median}}\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}

\vspace{{3pt}}
\begin{{minipage}}{{0.97\textwidth}}\footnotesize
Notes: $C$ changes membership only; the price objects and primary thresholds
are unchanged.  Beijing $C$ uses the native complete menu, whereas UrbanEV
$C$ uses complete support for all nine endpoints.
\end{{minipage}}
\end{{table}}
"""


def build_residual_table(transition: pd.DataFrame, residual_summary: pd.DataFrame) -> str:
    rows = []
    for row in transition.to_dict("records"):
        rows.append(
            " & ".join(
                [
                    tex_escape(row["system"]),
                    tex_escape(row["observed_signed_category"]),
                    tex_escape(row["no_residual_signed_category"]),
                    fmt_int(row["n"]),
                    fmt_pct(row["share_of_H"], 2),
                ]
            )
            + r" \\"
        )
    summary_text = "; ".join(
        f"{row['system']}: {fmt_int(row['changed_n'])} ({fmt_pct(row['changed_share'], 2)})"
        for row in residual_summary.to_dict("records")
    )
    return rf"""
\begin{{table}}[p]
\centering
\caption{{Signed-category changes after removing the aggregation residual.}}
\label{{tab:revision-residual-transition}}
\scriptsize
\setlength{{\tabcolsep}}{{5pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular}}{{@{{}}lllrr@{{}}}}
\toprule
System & Observed $D_{{30}}$ & Residual-free $30S^*$ & $n$ & Share of $H$\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}

\vspace{{3pt}}
\begin{{minipage}}{{0.97\textwidth}}\footnotesize
Notes: The residual-free quantity is $30S^*$; the observed quantity is
$D_{{30}}=30(S^*+r^*)$.  The primary $H$ membership is held fixed.
Changed-cell totals are {summary_text}.  Full zero cells are retained in the
companion CSV.
\end{{minipage}}
\end{{table}}
"""


def stacked_bar(ax: plt.Axes, summary: pd.DataFrame, prefix: str, title: str, colors: dict[str, str], xlabel: str) -> None:
    systems = summary["system"].tolist()
    y = np.arange(len(systems))
    denominator = summary[f"{prefix}_n"].to_numpy(float)
    left = np.zeros(len(systems), dtype=float)
    for category, color in colors.items():
        counts = summary[f"{prefix}_{category}_n"].to_numpy(float)
        shares = np.divide(
            100.0 * counts,
            denominator,
            out=np.zeros_like(counts),
            where=denominator > 0,
        )
        ax.barh(
            y,
            shares,
            left=left,
            color=color,
            height=0.56,
            label=category.capitalize(),
        )
        for i, (count, share) in enumerate(zip(counts, shares)):
            # Label only legible, large segments; the adjacent strata table
            # retains every count, including small segments.
            if share >= 5.0:
                ax.text(
                    left[i] + share / 2,
                    i,
                    f"{int(count):,}\n({share:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if color != GRAY else "#222222",
                    fontsize=11.0,
                )
        left += shares
    ax.set_yticks(y, systems)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", pad=10, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.grid(axis="x", color="#E6E6E6")
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, ncol=3, loc="lower left", bbox_to_anchor=(0, -0.40))
    for i, row in enumerate(summary.to_dict("records")):
        ax.text(
            1.02,
            i,
            f"n={fmt_int(row[f'{prefix}_n'])}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=11,
            clip_on=False,
        )


def build_figure(out: Path, frames: dict[str, pd.DataFrame]) -> list[str]:
    figdir = out / FIGURE_REL
    figdir.mkdir(parents=True, exist_ok=True)
    summary = frames["summary"].copy()
    units = frames["units"]
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.5), gridspec_kw={"hspace": 0.90})
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    stacked_bar(
        axes[0],
        summary,
        "R",
        "(a) Total-price direction within $R$",
        {"retained": BLUE, "neutralized": GRAY, "reversed": RED},
        "Share of R (%)",
    )
    stacked_bar(
        axes[1],
        summary,
        "H",
        "(b) Signed total-minus-electricity amount within $H$",
        {"negative": BLUE, "near": GRAY, "positive": GOLD},
        "Share of H (%)",
    )
    for ax in axes:
        ax.tick_params(labelsize=11)
    fig.subplots_adjust(bottom=0.14, top=0.95, left=0.14, right=0.86, hspace=0.90)
    paths = []
    for suffix in ("pdf", "svg"):
        path = figdir / f"Figure_revision_signed.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def build_report(out: Path, frames: dict[str, pd.DataFrame]) -> str:
    summary = frames["summary"]
    residual = frames["residual_summary"]
    threshold = frames["threshold"]
    lines = [
        "# Focused manuscript revision: scientific extension report",
        "",
        "This report is generated from frozen derived panels. It covers signed "
        "decomposition, external one-factor threshold sensitivity, and the "
        "embedded price-input audit. No raw orders or new simulations were read.",
        "",
        "## Primary results",
        "",
        "| system | A | R | H | C | R retained / neutralized / reversed | "
        "H negative / near / positive | H signed median (RMB) | H absolute median (RMB) |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['system']} | {fmt_int(row['A_n'])} | {fmt_int(row['R_n'])} | "
            f"{fmt_int(row['H_n'])} | {fmt_int(row['C_n'])} | "
            f"{fmt_int(row['R_retained_n'])} / {fmt_int(row['R_neutralized_n'])} / "
            f"{fmt_int(row['R_reversed_n'])} | "
            f"{fmt_int(row['H_negative_n'])} / {fmt_int(row['H_near_n'])} / "
            f"{fmt_int(row['H_positive_n'])} | "
            f"{fmt_num(row['H_signed_median_RMB'], 4)} | "
            f"{fmt_num(row['H_absolute_median_RMB'], 2)} |"
        )
    lines.extend(
        [
            "",
            "The amount threshold is 0.90 RMB, corresponding to 0.03 RMB/kWh "
            "at the 30 kWh reference. Signed-category closure and the identity "
            "D30 = 30(S* + r*) are checked in count_closure_checks.json.",
            "",
            "## Residual sensitivity",
            "",
        ]
    )
    for row in residual.to_dict("records"):
        lines.append(
            f"- {row['system']}: {fmt_int(row['changed_n'])} of {fmt_int(row['H_n'])} "
            f"H units ({fmt_pct(row['changed_share'], 2)}) change signed category "
            "when the aggregation residual is removed."
        )
    lines.extend(
        [
            "",
            "## Threshold sensitivity",
            "",
            f"UrbanEV has {len(threshold)} post-review specified one-factor rows: "
            "electricity qualification 0.005/0.01/0.02 RMB/kWh, total direction "
            "0.02/0.03/0.05 RMB/kWh, and amount 0.60/0.90/1.50 RMB on fixed "
            "primary H. Full counts are in threshold_sensitivity.csv.",
            "",
            "## Scope and risk",
            "",
            "The two observation systems are summarized separately with equal unit "
            "weights. Beijing C is obtained by a NATIVE_8 station-month "
            "many-to-one join; UrbanEV C is complete nine-endpoint support. "
            "These membership rules are system-specific and should not be "
            "interpreted as a common cross-city sampling design.",
        ]
    )
    path = out / EXT_REL / "extension_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def build(out: Path) -> dict[str, Any]:
    out = out.resolve()
    frames = read_outputs(out)
    table_dir = out / TABLE_REL
    table_dir.mkdir(parents=True, exist_ok=True)
    table_map = {
        "revision_signed_main.tex": build_main_table(frames["summary"]),
        "revision_signed_strata.tex": build_strata_table(frames["strata"]),
        "revision_threshold_sensitivity.tex": build_threshold_table(frames["threshold"]),
        "revision_input_audit.tex": build_audit_3x3(frames["audit3"]),
        "revision_input_audit_C.tex": build_audit_c(frames["auditc"]),
        "revision_residual_transition.tex": build_residual_table(
            frames["transition"], frames["residual_summary"]
        ),
    }
    for name, text in table_map.items():
        write_tex(table_dir / name, text)
    figure_paths = build_figure(out, frames)
    report_path = build_report(out, frames)
    checks = {
        "status": "PASS",
        "tables": [str(table_dir / name) for name in table_map],
        "figures": figure_paths,
        "strata_rows": int(len(frames["strata"])),
        "threshold_rows": int(len(frames["threshold"])),
        "input_audit_rows": int(len(frames["audit3"])),
        "figure_vector_formats": ["pdf", "svg"],
        "clearpage_inserted": False,
        "report": report_path,
        "source": str(out / EXT_REL),
    }
    (out / EXT_REL / "asset_build_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
