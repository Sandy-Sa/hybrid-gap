"""Render the three Precision/Recall Decomposition LaTeX sub-tables.

Reads ``outputs/revision/pr_decomp.csv`` (produced by
``python -m eval.harness pr-decomp``) and emits three table blocks suitable
for pasting into ``latex/acl_latex.tex`` under ``\\section{Precision/Recall
Decompositions}`` (``\\label{app:cost}``).

Sub-tables:
  - ``tab:pr-decomp-prf``  — micro/macro P, R, F1 with bootstrap CIs
  - ``tab:pr-decomp-topk`` — P@8, P@15, R@8, R@15, R-precision (SFT-only)
  - ``tab:pr-decomp-cal``  — EMR and mean codes-per-note (calibration)

Run from the repo root::

    cd /Users/full_of_bugs/repos/hybrid_coding
    uv run --project eval python -m eval.render.pr_decomp \\
        > outputs/revision/pr_decomp.tex
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from coding.eval.paths import EVAL_DIR, REPO_ROOT

DEFAULT_CSV = EVAL_DIR / "pr_decomp_scorable.csv"


def _fmt_n(x: str | float, *, digits: int = 3, drop_leading_zero: bool = True) -> str:
    """Format a number string/float to *digits* decimals; optionally strip the
    leading 0 for values in [0, 1) so that 0.572 → .572 (paper convention)."""
    if x == "" or x is None:
        return "---"
    v = float(x)
    s = f"{v:.{digits}f}"
    if drop_leading_zero:
        if s.startswith("0."):
            return s[1:]
        if s.startswith("-0."):
            return "-" + s[2:]
    return s


def _cell(point: str, lo: str, hi: str) -> str:
    """Render a `$p_{[lo,hi]}$` cell. If CI bounds are blank, show point only."""
    if point in ("", None):
        return "---"
    p = _fmt_n(point, digits=3, drop_leading_zero=False)
    if lo in ("", None) or hi in ("", None):
        return f"${p}$"
    return f"${p}_{{[{_fmt_n(lo)},{_fmt_n(hi)}]}}$"


def _cell_count(point: str, lo: str, hi: str) -> str:
    """Render a codes-per-note cell with 2-decimal precision (no leading-zero strip)."""
    if point in ("", None):
        return "---"
    p = f"{float(point):.2f}"
    if lo in ("", None) or hi in ("", None):
        return f"${p}$"
    return f"${p}_{{[{float(lo):.2f},{float(hi):.2f}]}}$"


# Display label for each (config, scope). Order matches paper rows.
# At the principled band: SFT-only at τ*_macro (=0.125) and τ*_micro (=0.311),
# stage rows composed off τ*_micro (=θ_high), and the principled hybrid.
ROW_ORDER: list[tuple[str, str, str, str]] = [
    # (config, scope, method_label, scope_label)
    ("sft_only_tau_macro", "in_dist", r"SFT only ($\tau{=}\tau^\star_\text{macro}$)", r"\textsc{InDist}"),
    ("sft_only_tau_macro", "full",    r"",                                            r"\textsc{Full}"),
    ("sft_only_tau_micro", "in_dist", r"SFT only ($\tau{=}\tau^\star_\text{micro}$, \textsc{sft\_high})", r"\textsc{InDist}"),
    ("sft_only_tau_micro", "full",    r"",                                            r"\textsc{Full}"),
    ("+band_verify_principled",      "full", r"\quad\textsc{+band\_verify}", r"\textsc{Full}"),
    ("+gap_fill_principled",         "full", r"\quad\textsc{+gap\_fill} (raw)", r"\textsc{Full}"),
    ("hybrid_principled",            "in_dist", r"\textbf{Hybrid (ours)}", r"\textsc{InDist}"),
    ("hybrid_principled",            "ood",     r"",                       r"\textsc{OOD}"),
    ("hybrid_principled",            "full",    r"",                       r"\textsc{Full}"),
]


def _index_csv(csv_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    """Return ``{(config, scope, metric): row}`` for fast lookup."""
    out: dict[tuple[str, str, str], dict[str, str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["config"], r["scope"], r["metric"])] = r
    return out


def render_prf_table(idx: dict, *, n: int, seed: int) -> str:
    rows: list[str] = []
    for config, scope, method_label, scope_label in ROW_ORDER:
        cells: list[str] = []
        for metric in (
            "precision_micro", "recall_micro", "f1_micro",
            "precision_macro", "recall_macro", "f1_macro",
        ):
            r = idx.get((config, scope, metric), {})
            cells.append(_cell(r.get("point", ""),
                               r.get("normal_ci_low", ""),
                               r.get("normal_ci_high", "")))
        rows.append(f"{method_label} & {scope_label} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    return rf"""\begin{{table*}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llcccccc}}
\toprule
& & \multicolumn{{3}}{{c}}{{Micro}} & \multicolumn{{3}}{{c}}{{Macro}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-8}}
Method & Scope & P & R & F1 & P & R & F1 \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\caption{{Precision, recall, and F1 (micro and macro) decomposition for every \textsc{{MIMIC-III}} row of Tables~\ref{{tab:main}} and~\ref{{tab:stages}}. Bracketed subscripts are $95\%$ bootstrap CIs ({n:,} admission-level resamples, seed~{seed}; normal-approximation interval $\hat{{\theta}}\pm 1.96\cdot\widehat{{\mathrm{{SE}}}}_{{\text{{boot}}}}$). \textsc{{OOD}} rows are omitted for SFT-only configurations because OOD codes are unreachable by construction (SFT vocabulary $|C|{{=}}3{{,}}681$). \textsc{{+gap\_fill}} is the raw retrieved-candidate union prior to verification, reported point-only as in Table~\ref{{tab:stages}}.}}
\label{{tab:pr-decomp-prf}}
\end{{table*}}"""


def render_topk_table(idx: dict, *, n: int, seed: int) -> str:
    """SFT-only top-k table. P@k/R@k/R-prec are threshold-free (computed from
    raw SFT logits), so the τ=0.15/0.35/0.5 rows yield identical values; we
    therefore report a single SFT row per scope, keeping τ=0.5 (the
    \\textsc{sft\\_high} anchor) as the labelled instance.
    """
    rows: list[str] = []
    for config, scope, scope_label in [
        ("sft_only_tau_micro", "in_dist", r"\textsc{InDist}"),
        ("sft_only_tau_micro", "full",    r"\textsc{Full}"),
    ]:
        cells: list[str] = []
        for metric in ("p_at_8", "p_at_15", "r_at_8", "r_at_15", "r_precision"):
            r = idx.get((config, scope, metric), {})
            cells.append(_cell(r.get("point", ""),
                               r.get("normal_ci_low", ""),
                               r.get("normal_ci_high", "")))
        rows.append(f"SFT (any $\\tau$) & {scope_label} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{llccccc}}
\toprule
Method & Scope & P@8 & P@15 & R@8 & R@15 & R-prec \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\caption{{Ranking metrics for the SFT classifier (Qwen3-4B-Base). Top-$k$ and R-precision are properties of the ranked logit output and are independent of the decision threshold $\tau$, so a single SFT row per scope suffices. Pipeline stages (\textsc{{+band\_verify}}, \textsc{{+gap\_fill}}, Hybrid) emit binary code sets without a native ranking and are omitted. CIs as in Table~\ref{{tab:pr-decomp-prf}} ({n:,} resamples, seed~{seed}).}}
\label{{tab:pr-decomp-topk}}
\end{{table}}"""


def render_cal_table(idx: dict, *, n: int, seed: int) -> str:
    rows: list[str] = []
    for config, scope, method_label, scope_label in ROW_ORDER:
        emr = idx.get((config, scope, "emr"), {})
        cpn = idx.get((config, scope, "codes_per_note"), {})
        emr_cell = _cell(emr.get("point", ""),
                         emr.get("normal_ci_low", ""),
                         emr.get("normal_ci_high", ""))
        cpn_cell = _cell_count(cpn.get("point", ""),
                               cpn.get("normal_ci_low", ""),
                               cpn.get("normal_ci_high", ""))
        rows.append(f"{method_label} & {scope_label} & {emr_cell} & {cpn_cell} \\\\")
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{6pt}}
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{llcc}}
\toprule
Method & Scope & EMR & Codes/note \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\caption{{Calibration check. EMR is the fraction of admissions whose predicted code set exactly equals the gold set (after intersecting both with the scope vocabulary). Codes/note is the mean predicted-set cardinality. Codes/note is invariant to scope for predictors whose output is a subset of \textsc{{InDist}} (SFT-only and \textsc{{+band\_verify}}); the \textsc{{OOD}} row for Hybrid reflects only the candidate-verifier additions to the OOD vocabulary. CIs as in Table~\ref{{tab:pr-decomp-prf}} ({n:,} resamples, seed~{seed}).}}
\label{{tab:pr-decomp-cal}}
\end{{table}}"""


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    csv_path = Path(argv[0]) if argv else DEFAULT_CSV
    if not csv_path.exists():
        print(f"error: {csv_path} not found; run `python -m eval.harness pr-decomp` first.",
              file=sys.stderr)
        return 2
    idx = _index_csv(csv_path)
    # Pull n / seed from any CI row.
    n = seed = 0
    for r in idx.values():
        if r.get("n") and r.get("seed"):
            n = int(r["n"])
            seed = int(r["seed"])
            break
    print("% Auto-generated by revision/render_pr_decomp.py — do not edit by hand.")
    print("% Source: " + str(csv_path.relative_to(REPO_ROOT)))
    print()
    print(render_prf_table(idx, n=n, seed=seed))
    print()
    print(render_topk_table(idx, n=n, seed=seed))
    print()
    print(render_cal_table(idx, n=n, seed=seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
