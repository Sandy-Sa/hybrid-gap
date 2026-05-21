"""Per-cell bootstrap CIs for every (system × test × scope) cell, under both
macro denominators. Feeds the bracketed subscripts in tab:main and tab:stages.

Writes ``outputs/revision/per_cell_cis_recomputed.parquet`` with columns:

    system, test_set, condition, macro_denominator,
    micro_f1, micro_ci_lo, micro_ci_hi,
    macro_f1, macro_ci_lo, macro_ci_hi,
    n_scorable, n_in_scope, n_resamples, seed

The 5 systems mirror ``eval.tables.headline_paired``:
``sft_only_tau_macro``, ``sft_only_tau_micro``, ``sft_band_verify``,
``gap_fill_intermediate``, ``full_pipeline``.

Run with: ``uv run python -m eval.tables.main_table``
"""
from __future__ import annotations

import sys

import polars as pl

from coding.eval.bootstrap import (
    bootstrap_f1_from_table,
    build_index_table,
)
from coding.eval.harness import (
    OUT_DIR,
    REPO_ROOT,
    SPEC,
    _load_dataset_context,
    _score_one_scope,
    _scope_vocabs,
    compose_predictions,
    load_gap_candidates,
)


N_RESAMPLES = 1000
SEED = 314

SCOPES = ("in_dist", "ood", "full")


def _build_systems(spec) -> tuple[dict, dict[str, dict[str, set[str]]]]:
    ctx = _load_dataset_context(
        spec,
        sft_thresholds=[spec.tau_macro, spec.tau_micro],
    )
    sft_macro = ctx["sft_lookups"][spec.tau_macro]
    sft_micro = ctx["sft_lookups"][spec.tau_micro]
    band_verify = ctx["band_verify"]
    candidate_verify = ctx["candidate_verify"]
    gap_candidates = load_gap_candidates(
        spec.stage2_dir / "gap_analysis_pass2_searched.parquet"
    )
    if ctx["mdace_leak_dropped"]:
        for eid in list(gap_candidates):
            if eid in ctx["mdace_leak_dropped"]:
                del gap_candidates[eid]
    systems = {
        "sft_only_tau_macro":    compose_predictions(sft_macro),
        "sft_only_tau_micro":    compose_predictions(sft_micro),
        "sft_band_verify":       compose_predictions(sft_micro, band_verify),
        "gap_fill_intermediate": compose_predictions(sft_micro, band_verify, gap_candidates),
        "full_pipeline":         compose_predictions(sft_micro, band_verify, candidate_verify),
    }
    return ctx, systems


def main() -> int:
    rows: list[dict] = []
    for spec in (SPEC,):
        print(f"\n=== per_cell_cis: {spec.name} ===")
        ctx, systems = _build_systems(spec)
        scope_vocabs = _scope_vocabs(ctx)
        for system, pred in systems.items():
            for scope in SCOPES:
                vocab = scope_vocabs[scope]
                if not vocab:
                    continue
                pr = _score_one_scope(
                    [pred.get(eid, set()) for eid in ctx["event_ids"]],
                    [ctx["gold_lookup"].get(eid, set()) for eid in ctx["event_ids"]],
                    vocab,
                )
                table = build_index_table(
                    pred, ctx["gold_lookup"], ctx["event_ids"], vocab,
                )
                for denom in ("full", "scorable"):
                    boot = bootstrap_f1_from_table(
                        table, n=N_RESAMPLES, seed=SEED,
                        label=f"{spec.name}/{system}/{scope}/{denom}",
                        macro_denominator=denom,
                    )
                    pr_denom = (
                        pr if denom == "full"
                        else _score_one_scope(
                            [pred.get(eid, set()) for eid in ctx["event_ids"]],
                            [ctx["gold_lookup"].get(eid, set()) for eid in ctx["event_ids"]],
                            vocab,
                            macro_denominator="scorable",
                        )
                    )
                    rows.append({
                        "system": system,
                        "test_set": spec.name,
                        "condition": scope,
                        "macro_denominator": denom,
                        "micro_f1": pr_denom["f1_micro"],
                        "micro_ci_lo": boot.micro_normal_ci_low,
                        "micro_ci_hi": boot.micro_normal_ci_high,
                        "macro_f1": pr_denom["f1_macro"],
                        "macro_ci_lo": boot.macro_normal_ci_low,
                        "macro_ci_hi": boot.macro_normal_ci_high,
                        "n_scorable": int(pr_denom["n_scorable"]),
                        "n_in_scope": int(pr_denom["n_in_scope"]),
                        "n_resamples": N_RESAMPLES,
                        "seed": SEED,
                    })
                    print(
                        f"  {system:<25s} | {scope:<8s} | {denom:8s} | "
                        f"micro={pr_denom['f1_micro']:.4f}"
                        f" [{boot.micro_normal_ci_low:.4f},{boot.micro_normal_ci_high:.4f}]"
                        f" | macro={pr_denom['f1_macro']:.4f}"
                        f" [{boot.macro_normal_ci_low:.4f},{boot.macro_normal_ci_high:.4f}]"
                    )

    df = pl.DataFrame(rows)
    out_path = OUT_DIR / "per_cell_cis_recomputed.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)} ({df.height} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
