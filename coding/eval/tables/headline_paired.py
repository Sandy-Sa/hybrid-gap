"""Recompute paired-bootstrap CIs/p-values for headline comparisons under both
macro denominators.

For each (comparison, dataset, scope) cell:
  * builds IndexTables for systems A and B over the requested vocab
  * runs paired bootstrap once with `macro_denominator="full"` and once with `"scorable"`
  * records delta_macro, delta_micro, normal-approx CI, two-sided bootstrap p

Comparisons reported (from playbook §Phase 3 #13):
  * Hybrid (full_pipeline) vs SFT-only @ τ*_macro — Full and InDist scopes
  * Stage deltas on Full: +band_verify vs sft_high, +verify_gap_fill vs +band_verify
  * Skip-band ablation: NO ablated pipeline cached today; surfaced as "missing".

Run with: ``uv run python revision/recompute_paired_bootstraps.py``
"""
from __future__ import annotations

import sys

import polars as pl

from coding.eval.bootstrap import (
    build_index_table,
    paired_bootstrap_test_from_tables,
)
from coding.eval.harness import (
    OUT_DIR,
    REPO_ROOT,
    SPEC,
    _load_dataset_context,
    _scope_vocabs,
    compose_predictions,
)


N_RESAMPLES = 1000
SEED = 314

# (a_label, b_label, scope) — labels resolve in build_systems() below.
HEADLINE_COMPARISONS = [
    ("full_pipeline", "sft_only_tau_macro", "full"),
    ("full_pipeline", "sft_only_tau_macro", "in_dist"),
    ("sft_band_verify", "sft_only_tau_micro", "full"),  # +band_verify vs sft_high
    ("full_pipeline", "sft_band_verify", "full"),        # +verify_gap_fill vs +band_verify
]


def build_systems(spec) -> tuple[dict, dict[str, dict[str, set[str]]]]:
    ctx = _load_dataset_context(
        spec,
        sft_thresholds=[spec.tau_macro, spec.tau_micro],
    )
    sft_macro = ctx["sft_lookups"][spec.tau_macro]
    sft_micro = ctx["sft_lookups"][spec.tau_micro]
    band_verify = ctx["band_verify"]
    candidate_verify = ctx["candidate_verify"]
    systems = {
        "sft_only_tau_macro": compose_predictions(sft_macro),
        "sft_only_tau_micro": compose_predictions(sft_micro),
        "sft_band_verify":    compose_predictions(sft_micro, band_verify),
        "full_pipeline":      compose_predictions(sft_micro, band_verify, candidate_verify),
    }
    return ctx, systems


def main() -> int:
    rows: list[dict] = []
    for spec in (SPEC,):
        print(f"\n=== paired_bootstraps: {spec.name} ===")
        ctx, systems = build_systems(spec)
        scope_vocabs = _scope_vocabs(ctx)
        for a_label, b_label, scope in HEADLINE_COMPARISONS:
            vocab = scope_vocabs[scope]
            if not vocab:
                continue
            print(f"  {a_label} vs {b_label} on {scope} (|vocab|={len(vocab)}) …")
            tab_a = build_index_table(systems[a_label], ctx["gold_lookup"], ctx["event_ids"], vocab)
            tab_b = build_index_table(systems[b_label], ctx["gold_lookup"], ctx["event_ids"], vocab)
            for denom in ("full", "scorable"):
                res = paired_bootstrap_test_from_tables(
                    tab_a, tab_b, n=N_RESAMPLES, seed=SEED,
                    label=f"{spec.name}/{a_label}_vs_{b_label}/{scope}/{denom}",
                    macro_denominator=denom,
                )
                rows.append({
                    "comparison": f"{a_label}_vs_{b_label}",
                    "test_set": spec.name,
                    "condition": scope,
                    "macro_denominator": denom,
                    "delta_micro": res.delta_micro,
                    "delta_macro": res.delta_macro,
                    "delta_micro_ci_lo": res.delta_micro_normal_ci_low,
                    "delta_micro_ci_hi": res.delta_micro_normal_ci_high,
                    "delta_macro_ci_lo": res.delta_macro_normal_ci_low,
                    "delta_macro_ci_hi": res.delta_macro_normal_ci_high,
                    "p_micro_two_sided": res.p_two_sided_micro,
                    "p_macro_two_sided": res.p_two_sided_macro,
                    "n_resamples": res.n_resamples,
                    "seed": res.seed,
                })
                print(
                    f"    {denom:8s} | Δμ={res.delta_micro:+.4f} "
                    f"[{res.delta_micro_normal_ci_low:+.4f},{res.delta_micro_normal_ci_high:+.4f}] "
                    f"p={res.p_two_sided_micro:.4f} | ΔM={res.delta_macro:+.4f} "
                    f"[{res.delta_macro_normal_ci_low:+.4f},{res.delta_macro_normal_ci_high:+.4f}] "
                    f"p={res.p_two_sided_macro:.4f}"
                )

    df = pl.DataFrame(rows)
    out_path = OUT_DIR / "bootstrap_recomputed.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)} ({df.height} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
