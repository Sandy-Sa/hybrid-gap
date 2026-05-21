"""Bootstrap CIs for the +gap_fill intermediate row in tab:stages.

The intermediate `+gap_fill` row (sft_high ∪ band_verify_confirmed ∪ raw top-N
candidates, before candidate-verify) was previously reported point-only in
tab:stages. Stage 5.1.b of REVISION_PLAN_2026-05-11.md asks for normal-approx
95% CIs on this row for both datasets.

Outputs:
    outputs/revision/results/gap_fill_intermediate/mimic_iii_clean/raw_gap_fill[_scorable].json
    outputs/revision/results/gap_fill_intermediate/mimic_iv/raw_gap_fill[_scorable].json

Run from the repo root:

    cd /Users/full_of_bugs/repos/hybrid_coding
    uv run --project eval python -m eval.tables.stages_table [--macro-denominator=full|scorable]
"""

from __future__ import annotations

import json
import sys
from typing import Literal

from coding.eval.bootstrap import (
    bootstrap_f1_from_table,
    build_index_table,
)
from coding.eval.harness import (
    OUT_DIR as EVAL_DIR,
    REPO_ROOT,
    SPEC,
    _load_dataset_context,
    _round_floats,
    compose_predictions,
    load_gap_candidates,
)

MacroDenominator = Literal["full", "scorable"]

OUT_DIR = EVAL_DIR / "results" / "gap_fill_intermediate"

N_RESAMPLES = 1_000
SEED = 314


def run_one(spec, *, macro_denominator: MacroDenominator = "full") -> dict:
    print(f"\n=== {spec.name}: +gap_fill (sft_high ∪ band_verify ∪ raw top-N) "
          f"(n={N_RESAMPLES}, seed={SEED}, macro_denominator={macro_denominator}) ===")
    ctx = _load_dataset_context(spec, sft_thresholds=[spec.tau_micro])
    sft_high = ctx["sft_lookups"][spec.tau_micro]
    band_verify = ctx["band_verify"]

    gap_lookup = load_gap_candidates(
        spec.stage2_dir / "gap_analysis_pass2_searched.parquet"
    )
    # Mirror _load_dataset_context's MDACE-leak filter on the gap lookup.
    dropped = set(gap_lookup.keys()) - set(ctx["event_ids"])
    for eid in dropped:
        del gap_lookup[eid]
    print(f"  raw +gap_fill candidates loaded: {len(gap_lookup)} admissions "
          f"({len(dropped)} dropped to align with filtered test set)")

    pred_gap = compose_predictions(sft_high, band_verify, gap_lookup)

    full_vocab = set(ctx["full_class_maps_global"][0])
    event_ids = ctx["event_ids"]
    gold = ctx["gold_lookup"]

    print(f"  building index table (full scope, |vocab|={len(full_vocab)}, "
          f"n_admissions={len(event_ids)}) …")
    table = build_index_table(pred_gap, gold, event_ids, full_vocab)

    print(f"  running single-config bootstrap …")
    res = bootstrap_f1_from_table(
        table, n=N_RESAMPLES, seed=SEED,
        label=f"{spec.name}/+gap_fill",
        macro_denominator=macro_denominator,
    )
    return {
        "dataset": spec.name,
        "scope": "full",
        "config": "+gap_fill",
        "composition": "sft_high ∪ band_verify_confirmed ∪ raw_top_N_candidates",
        "macro_denominator": macro_denominator,
        "tau_high": spec.tau_micro,
        "n_admissions": len(event_ids),
        "vocab_size": len(full_vocab),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "f1_micro": res.point_micro,
        "f1_macro": res.point_macro,
        "f1_micro_normal_ci": [res.micro_normal_ci_low, res.micro_normal_ci_high],
        "f1_macro_normal_ci": [res.macro_normal_ci_low, res.macro_normal_ci_high],
        "f1_micro_pct_ci": [res.micro_ci_low, res.micro_ci_high],
        "f1_macro_pct_ci": [res.macro_ci_low, res.macro_ci_high],
        "f1_micro_bca_ci": [res.micro_bca_ci_low, res.micro_bca_ci_high],
        "f1_macro_bca_ci": [res.macro_bca_ci_low, res.macro_bca_ci_high],
        "f1_micro_boot_mean": res.micro_boot_mean,
        "f1_macro_boot_mean": res.macro_boot_mean,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    macro_denominator: MacroDenominator = "full"
    for arg in sys.argv[1:]:
        if arg.startswith("--macro-denominator="):
            choice = arg.split("=", 1)[1]
            if choice not in {"full", "scorable"}:
                print(f"unknown --macro-denominator value: {choice!r}",
                      file=sys.stderr)
                sys.exit(2)
            macro_denominator = choice  # type: ignore[assignment]
    suffix = "" if macro_denominator == "full" else "_scorable"
    results = []
    for spec, subdir in [(SPEC, SPEC.name)]:
        r = run_one(spec, macro_denominator=macro_denominator)
        results.append(r)
        out_path = OUT_DIR / subdir / f"raw_gap_fill{suffix}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(_round_floats(r, ndigits=4), f, indent=2)
        print(f"\n  wrote {out_path.relative_to(REPO_ROOT)}")
    print("\n=== summary ===")
    for r in results:
        print(
            f"  {r['dataset']}: micro {r['f1_micro']:.4f} "
            f"[{r['f1_micro_normal_ci'][0]:.4f}, {r['f1_micro_normal_ci'][1]:.4f}], "
            f"macro {r['f1_macro']:.4f} "
            f"[{r['f1_macro_normal_ci'][0]:.4f}, {r['f1_macro_normal_ci'][1]:.4f}]"
        )


if __name__ == "__main__":
    main()
