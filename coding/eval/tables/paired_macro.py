"""Paired bootstrap: full_pipeline vs sft_only_tau_macro, MIMIC-III and MIMIC-IV.

The eval-driver's saved paired_tests/*.json files omit this pair (they cover
full_pipeline vs sft_only_tau_micro and vs sft_band_verify only) and use
n_resamples=1000. This focused script fills both gaps without re-running the
existing per-config bootstraps.
"""

from __future__ import annotations

import json

from coding.eval.bootstrap import (
    build_index_table,
    paired_bootstrap_test_from_tables,
)
from coding.eval.harness import (
    OUT_DIR,
    REPO_ROOT,
    SPEC,
    _load_dataset_context,
    _round_floats,
    compose_predictions,
)

OUT_PATH = OUT_DIR / "results" / "paired_tests" / "full_pipeline_vs_sft_macro.json"

N_RESAMPLES = 1_000
SEED = 314


def run_one(spec) -> dict:
    print(f"\n=== {spec.name}: full_pipeline vs sft_only_tau_macro "
          f"(n={N_RESAMPLES}, seed={SEED}) ===")
    ctx = _load_dataset_context(
        spec, sft_thresholds=[spec.tau_macro, spec.tau_micro],
    )
    sft_macro = ctx["sft_lookups"][spec.tau_macro]
    sft_micro = ctx["sft_lookups"][spec.tau_micro]
    band_verify = ctx["band_verify"]
    candidate_verify = ctx["candidate_verify"]

    pred_sft_macro = compose_predictions(sft_macro)
    pred_full = compose_predictions(sft_micro, band_verify, candidate_verify)

    full_vocab = set(ctx["full_class_maps_global"][0])
    event_ids = ctx["event_ids"]
    gold = ctx["gold_lookup"]

    print(f"  building index tables (full scope, |vocab|={len(full_vocab)}, "
          f"n_admissions={len(event_ids)}) …")
    table_full = build_index_table(pred_full, gold, event_ids, full_vocab)
    table_macro = build_index_table(pred_sft_macro, gold, event_ids, full_vocab)

    print(f"  running paired bootstrap …")
    res = paired_bootstrap_test_from_tables(
        table_full, table_macro,
        n=N_RESAMPLES, seed=SEED,
        label=f"{spec.name}/full_pipeline_vs_sft_only_tau_macro",
    )
    return {
        "dataset": spec.name,
        "scope": "full",
        "a_config": "full_pipeline",
        "b_config": "sft_only_tau_macro",
        "n_admissions": len(event_ids),
        "vocab_size": len(full_vocab),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "delta_micro": res.delta_micro,
        "delta_macro": res.delta_macro,
        "delta_micro_normal_ci": [
            res.delta_micro_normal_ci_low,
            res.delta_micro_normal_ci_high,
        ],
        "delta_macro_normal_ci": [
            res.delta_macro_normal_ci_low,
            res.delta_macro_normal_ci_high,
        ],
        "delta_micro_pct_ci": [
            res.delta_micro_ci_low,
            res.delta_micro_ci_high,
        ],
        "delta_macro_pct_ci": [
            res.delta_macro_ci_low,
            res.delta_macro_ci_high,
        ],
        "delta_micro_bca_ci": [
            res.delta_micro_bca_ci_low,
            res.delta_micro_bca_ci_high,
        ],
        "delta_macro_bca_ci": [
            res.delta_macro_bca_ci_low,
            res.delta_macro_bca_ci_high,
        ],
        "p_two_sided_micro": res.p_two_sided_micro,
        "p_two_sided_macro": res.p_two_sided_macro,
    }


def main() -> None:
    results = [run_one(SPEC)]
    payload = {
        "comparison": "full_pipeline vs sft_only_tau_macro",
        "scope": "full",
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "results": _round_floats(results, ndigits=4),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)}")
    for r in results:
        print(f"\n  {r['dataset']}: "
              f"Δmacro = {r['delta_macro']:.4f} "
              f"(95% normal CI [{r['delta_macro_normal_ci'][0]:.4f}, "
              f"{r['delta_macro_normal_ci'][1]:.4f}], "
              f"p_two_sided = {r['p_two_sided_macro']:.4g})")


if __name__ == "__main__":
    main()
