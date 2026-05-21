"""DVC entry point for the eval stage.

Runs every producer in order so a single `dvc repro evaluate` regenerates all
the artifacts the notebook (`coding/eval/reproduce_paper.ipynb`) consumes.
"""

from __future__ import annotations

import sys
import traceback

from coding.eval import build_freq_bands
from coding.eval.decomposition import chapter as decomp_chapter
from coding.eval.decomposition import error_analysis as decomp_error
from coding.eval.decomposition import freq_band as decomp_freq
from coding.eval.render import chapter_stage_figs
from coding.eval.tables import (
    headline_paired,
    main_table,
    paired_macro,
    per_class_thresholds,
    stage2_cost,
    stages_table,
)
from coding.eval.validation import format_comparison, macro_recompute


def _run_step(label: str, fn, *args, **kwargs) -> bool:
    print(f"\n===== {label} =====")
    try:
        fn(*args, **kwargs)
        return True
    except Exception:  # noqa: BLE001
        print(f"!! {label} failed:")
        traceback.print_exc()
        return False


def main() -> int:
    ok = True
    # Producers run sequentially because later stages consume earlier outputs
    # (e.g. macro_recompute reads parquets written by main_table / headline_paired,
    # format_comparison reads what macro_recompute wrote, render/chapter_stage_figs
    # reads what decomp_chapter wrote).
    ok &= _run_step("build_freq_bands", build_freq_bands.main)
    ok &= _run_step("tables.main_table", main_table.main)
    ok &= _run_step(
        "tables.headline_paired", headline_paired.main,
    )
    ok &= _run_step("tables.paired_macro", paired_macro.main)
    ok &= _run_step(
        "tables.stages_table.full", stages_table.main,
    )
    # Re-run stages_table for the scorable denominator (notebook reads that).
    saved_argv = sys.argv[:]
    try:
        sys.argv = ["stages_table", "--macro-denominator=scorable"]
        ok &= _run_step("tables.stages_table.scorable", stages_table.main)
    finally:
        sys.argv = saved_argv
    ok &= _run_step("tables.stage2_cost", stage2_cost.main)
    ok &= _run_step("tables.per_class_thresholds", per_class_thresholds.main)
    ok &= _run_step("decomposition.error_analysis", decomp_error.main)
    # Chapter heatmap with bootstrap CIs (scorable denominator).
    ok &= _run_step(
        "decomposition.chapter",
        decomp_chapter.main,
        ["--bootstrap-chapter-deltas", "--macro-denominator=scorable"],
    )
    ok &= _run_step(
        "decomposition.freq_band",
        decomp_freq.main,
        ["--include-fp-by-band", "--macro-denominator=scorable"],
    )
    ok &= _run_step("validation.macro_recompute", macro_recompute.main)
    ok &= _run_step("validation.format_comparison", format_comparison.main)
    ok &= _run_step("render.chapter_stage_figs", chapter_stage_figs.main)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
