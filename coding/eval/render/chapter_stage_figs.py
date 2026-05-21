"""Render the per-chapter stage-delta figures used in the appendix.

Reads ``outputs/revision/error_analyses/<dataset>/chapter_heatmap.json`` (must
be produced with ``--bootstrap-chapter-deltas`` so the Full subset rows carry
``ci_normal`` for every transition) and writes two PNGs that
``latex/acl_latex.tex`` ``\\includegraphics``-es:

  - ``latex/figures/fig_chapter_stage_f1_full_diag.png``
  - ``latex/figures/fig_chapter_stage_f1_full_proc.png``

Each figure has four horizontal panels — one per transition (``D_in_band_verify``,
``D_gapfill``, ``D_verify_gap_fill``, ``D_total``) — with a row per kept chapter
(min-support gate: 30 gold events). Rows are sorted by ``D_total`` descending.
Points are marked, 95% CIs drawn as horizontal whiskers, and a red zero-line
is overlaid; each panel's mean ΔF1 is annotated bottom-right.

Run from the repo root::

    cd /Users/full_of_bugs/repos/hybrid_coding
    uv run --project eval python -m eval.render.chapter_stage_figs \\
        [--dataset=mimic_iii_clean]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402

from coding.eval.harness import OUT_DIR, REPO_ROOT  # noqa: E402


TRANSITIONS = [
    "D_in_band_verify",
    "D_gapfill",
    "D_verify_gap_fill",
    "D_total",
]
LATEX_FIG_DIR = OUT_DIR / "figures"


def _load_heatmap(dataset: str) -> dict:
    for candidate in ("chapter_heatmap_scorable.json", "chapter_heatmap.json"):
        path = OUT_DIR / "error_analyses" / dataset / candidate
        if path.exists():
            return json.loads(path.read_text())
    print(
        f"error: {OUT_DIR / 'error_analyses' / dataset} has no chapter_heatmap*.json; "
        "run `python -m coding.eval.decomposition.chapter "
        "--bootstrap-chapter-deltas` first.",
        file=sys.stderr,
    )
    sys.exit(2)


def _index_full_f1(heatmap: dict) -> dict[str, dict[str, dict]]:
    """Return ``{chapter_id: {transition: row}}`` for subset=full, metric=f1."""
    out: dict[str, dict[str, dict]] = {}
    for r in heatmap["chapter_deltas"]["rows"]:
        if r["subset"] != "full" or r["metric"] != "f1":
            continue
        if r["transition"] not in TRANSITIONS:
            continue
        out.setdefault(r["chapter_id"], {})[r["transition"]] = r
    return out


def _kept_chapters_full(heatmap: dict) -> list[str]:
    return list(heatmap["by_chapter"]["full"].get("kept_chapters", []))


def _is_proc(chapter_id: str) -> bool:
    return chapter_id.startswith("ICD9PCS-")


def _rows_sorted_by_d_total(
    chapters: list[str], indexed: dict[str, dict[str, dict]],
) -> list[tuple[str, str]]:
    """Return ``[(chapter_id, chapter_name), ...]`` sorted by D_total desc."""
    keyed = []
    for ch in chapters:
        rows = indexed.get(ch, {})
        d_total = rows.get("D_total", {}).get("value", 0.0)
        name = next(iter(rows.values()))["chapter_name"] if rows else ch
        keyed.append((ch, name, d_total))
    keyed.sort(key=lambda t: -t[2])
    return [(ch, name) for ch, name, _ in keyed]


def _render_one(
    *,
    title: str,
    chapter_rows: list[tuple[str, str]],
    indexed: dict[str, dict[str, dict]],
    out_path: Path,
) -> None:
    n_chapters = len(chapter_rows)
    if n_chapters == 0:
        print(f"  skip {out_path.name}: no chapters")
        return
    fig_height = max(3.0, 0.30 * n_chapters + 1.5)
    fig, axes = plt.subplots(
        nrows=1, ncols=len(TRANSITIONS),
        sharey=True,
        figsize=(13.0, fig_height),
        gridspec_kw={"wspace": 0.05},
    )
    fig.suptitle(title, fontsize=10)

    y_labels = [name for _, name in chapter_rows]
    y_positions = list(range(n_chapters))[::-1]  # top row = first sorted

    for ax, transition in zip(axes, TRANSITIONS, strict=True):
        points: list[float] = []
        for y, (ch, _name) in zip(y_positions, chapter_rows, strict=True):
            row = indexed.get(ch, {}).get(transition)
            if row is None:
                continue
            point = row["value"]
            ci = row.get("ci_normal")
            points.append(point)
            if ci is not None:
                lo, hi = ci
                ax.hlines(y=y, xmin=lo, xmax=hi, color="#1f77b4", linewidth=1.4)
            ax.plot(
                [point], [y], marker="s", markersize=4,
                color="#1f77b4", linestyle="",
            )
        ax.axvline(0.0, color="red", linewidth=0.8, alpha=0.8)
        ax.set_title(transition, fontsize=9)
        ax.set_xlabel("DF1", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        # Mean annotation bottom-right.
        if points:
            mean_val = sum(points) / len(points)
            ax.text(
                0.98, 0.02, f"mean={mean_val:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white", edgecolor="#888", alpha=0.85,
                ),
            )

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(y_labels, fontsize=7)
    axes[0].set_ylim(-0.5, n_chapters - 0.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}  ({n_chapters} chapters)")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    dataset = "mimic_iii_clean"
    out_dir: Path = LATEX_FIG_DIR
    for arg in argv:
        if arg.startswith("--dataset="):
            dataset = arg.split("=", 1)[1]
        elif arg.startswith("--out-dir="):
            out_dir = Path(arg.split("=", 1)[1])
        elif arg in {"-h", "--help"}:
            print(__doc__)
            return 0
        else:
            print(f"unknown argument: {arg!r}", file=sys.stderr)
            return 2

    heatmap = _load_heatmap(dataset)
    boot_meta = heatmap["chapter_deltas"].get("bootstrap_meta", {})
    applied = boot_meta.get("applied_to", [])
    has_full_cis = any(s.startswith("full/") for s in applied) if isinstance(applied, list) else False
    if not has_full_cis:
        print(
            "error: heatmap chapter_deltas.bootstrap_meta does not include "
            "Full-subset CIs. Re-run eval.decomposition.chapter with "
            "--bootstrap-chapter-deltas.",
            file=sys.stderr,
        )
        return 2

    indexed = _index_full_f1(heatmap)
    kept = _kept_chapters_full(heatmap)
    if not kept:
        # Fallback: use whichever chapters have Full subset rows.
        kept = list(indexed.keys())

    diag_chapters = [ch for ch in kept if not _is_proc(ch)]
    proc_chapters = [ch for ch in kept if _is_proc(ch)]

    diag_rows = _rows_sorted_by_d_total(diag_chapters, indexed)
    proc_rows = _rows_sorted_by_d_total(proc_chapters, indexed)

    min_support = heatmap["by_chapter"]["full"].get("min_support", 30)
    title_prefix = f"{dataset.replace('_', '-').upper()} chapter × stage DF1"
    common_suffix = f"subset = full  (gate = {min_support} gold events)"

    print(f"=== render_chapter_stage_figs: {dataset} ===")
    print(f"  diag chapters: {len(diag_rows)}, proc chapters: {len(proc_rows)}")

    _render_one(
        title=f"{title_prefix} — {common_suffix} (diag)",
        chapter_rows=diag_rows, indexed=indexed,
        out_path=out_dir / "fig_chapter_stage_f1_full_diag.png",
    )
    _render_one(
        title=f"{title_prefix} — {common_suffix} (proc)",
        chapter_rows=proc_rows, indexed=indexed,
        out_path=out_dir / "fig_chapter_stage_f1_full_proc.png",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
