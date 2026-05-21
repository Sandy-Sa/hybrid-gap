"""Phase 6.1 + 6.2 error analyses at the principled band.

Emits two JSONs per dataset under ``outputs/revision/error_analyses/<dataset>/``:

  * ``oracle_waterfall.json`` — six rows (Phase 6.1):
        C0       = SFT @ tau_macro
        C0_high  = SFT @ tau_micro
        C0_5     = SFT @ tau_micro u band_verifier_confirmed (real verifier)
        C1       = SFT @ tau_micro u (band_input n gold)        — *oracle* band verifier
        C2       = C0_5 u candidate_verifier_confirmed (= full_pipeline, real)
        C3       = C1 u (candidate_pool n gold)                   — *oracle* gap-fill
    Each row reports micro/macro P/R/F1 across {in_dist, ood, full} via the
    same ``_scope_metrics_for_pred`` machinery used by re_eval. ``delta_full``
    is the micro/macro F1 lift on the full scope from the previous oracle row.

  * ``delta_attribution.json`` — codes added by each Stage-2 transition
    relative to the SFT @ tau_micro baseline (Phase 6.2):
        verifier_recovery       (TP added by band-verifier-confirmed)
        verifier_false_accept   (FP added by band-verifier-confirmed)
        gap_fill_recovery       (TP added by candidate-verifier-confirmed)
        gap_fill_hallucination  (FP added by candidate-verifier-confirmed)
    Aggregate counts plus FP:TP ratio per provenance.

Usage::

    cd /Users/full_of_bugs/repos/hybrid_coding
    uv run --project eval python -m eval.decomposition.error_analysis [--n=1000]

Reuses ``_load_dataset_context``, ``_scope_metrics_for_pred``,
``compose_predictions``, ``_round_floats``, and ``load_gap_candidates`` from
``revision.eval``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from coding.eval.harness import (
    OUT_DIR,
    REPO_ROOT,
    SPEC,
    DatasetSpec,
    _load_dataset_context,
    _round_floats,
    _scope_metrics_for_pred,
    compose_predictions,
    load_gap_candidates,
)


# Oracle ladder names used in REVISION_GUIDE.md / error_analysis.ipynb cell 34.
ORACLE_LADDER = ["C0", "C0_high", "C0_5", "C1", "C2", "C3"]


def _build_oracle_sets(
    *,
    sft_macro: dict[str, set[str]],
    sft_micro: dict[str, set[str]],
    band_verify: dict[str, set[str]],
    candidate_verify: dict[str, set[str]],
    gap_candidates: dict[str, set[str]],
    gold_lookup: dict[str, set[str]],
) -> dict[str, dict[str, set[str]]]:
    """Build per-event predicted-code sets for each oracle ladder rung."""
    eids = sorted(gold_lookup)
    oracle_sets: dict[str, dict[str, set[str]]] = {c: {} for c in ORACLE_LADDER}

    for eid in eids:
        gold = gold_lookup.get(eid, set())
        s_macro = sft_macro.get(eid, set())  # SFT >= tau_macro (band lower edge)
        s_micro = sft_micro.get(eid, set())  # SFT >= tau_micro (band upper edge)
        band_input = s_macro - s_micro       # codes in [tau_macro, tau_micro)
        bv = band_verify.get(eid, set())
        cv = candidate_verify.get(eid, set())
        cands = gap_candidates.get(eid, set())

        oracle_sets["C0"][eid] = set(s_macro)
        oracle_sets["C0_high"][eid] = set(s_micro)
        oracle_sets["C0_5"][eid] = s_micro | bv
        # C1: oracle band verifier = sft_micro plus band-input codes that are in gold.
        oracle_sets["C1"][eid] = s_micro | (band_input & gold)
        # C2: real candidate-verifier on top of C0_5.
        oracle_sets["C2"][eid] = (s_micro | bv) | cv
        # C3: oracle gap-fill = oracle band verifier plus retrieval candidates that are in gold.
        oracle_sets["C3"][eid] = (s_micro | (band_input & gold)) | (cands & gold)

    return oracle_sets


def _delta_attribution(
    *,
    sft_micro: dict[str, set[str]],
    band_verify: dict[str, set[str]],
    candidate_verify: dict[str, set[str]],
    gold_lookup: dict[str, set[str]],
) -> dict:
    """Counts per provenance of codes added vs SFT@tau_micro baseline."""
    counts = {
        "verifier_recovery": 0,
        "verifier_false_accept": 0,
        "gap_fill_recovery": 0,
        "gap_fill_hallucination": 0,
    }
    n_events_with_any = 0
    for eid, gold in gold_lookup.items():
        sm = sft_micro.get(eid, set())
        # Verifier transition: codes in band_verify but not already in sft_micro.
        verifier_added = band_verify.get(eid, set()) - sm
        # Gap-fill transition: candidate-verifier codes not already in sft_micro u verifier_added.
        verified_so_far = sm | verifier_added
        gap_added = candidate_verify.get(eid, set()) - verified_so_far
        any_added = bool(verifier_added) or bool(gap_added)
        if any_added:
            n_events_with_any += 1
        for c in verifier_added:
            counts["verifier_recovery" if c in gold else "verifier_false_accept"] += 1
        for c in gap_added:
            counts["gap_fill_recovery" if c in gold else "gap_fill_hallucination"] += 1

    def _ratio(fp: int, tp: int) -> float | None:
        return float(fp) / tp if tp > 0 else None

    return {
        "counts": counts,
        "verifier": {
            "tp": counts["verifier_recovery"],
            "fp": counts["verifier_false_accept"],
            "fp_to_tp_ratio": _ratio(counts["verifier_false_accept"], counts["verifier_recovery"]),
        },
        "gap_fill": {
            "tp": counts["gap_fill_recovery"],
            "fp": counts["gap_fill_hallucination"],
            "fp_to_tp_ratio": _ratio(counts["gap_fill_hallucination"], counts["gap_fill_recovery"]),
        },
        "combined": {
            "tp": counts["verifier_recovery"] + counts["gap_fill_recovery"],
            "fp": counts["verifier_false_accept"] + counts["gap_fill_hallucination"],
            "fp_to_tp_ratio": _ratio(
                counts["verifier_false_accept"] + counts["gap_fill_hallucination"],
                counts["verifier_recovery"] + counts["gap_fill_recovery"],
            ),
        },
        "n_events_with_any_added": n_events_with_any,
    }


def _waterfall_for_spec(
    spec: DatasetSpec,
    *,
    n: int,
    seed: int,
    filter_mdace_leak: bool | None,
) -> tuple[dict, dict]:
    """Compute oracle waterfall + delta attribution for one dataset spec."""
    print(f"\n=== error_analysis: {spec.name} ===")
    ctx = _load_dataset_context(
        spec,
        sft_thresholds=[spec.tau_macro, spec.tau_micro],
        filter_mdace_leak=filter_mdace_leak,
    )
    sft_macro = ctx["sft_lookups"][spec.tau_macro]
    sft_micro = ctx["sft_lookups"][spec.tau_micro]
    band_verify = ctx["band_verify"]
    candidate_verify = ctx["candidate_verify"]
    gold_lookup = ctx["gold_lookup"]
    n_adm = len(ctx["event_ids"])

    print(f"  loading gap candidate pool from {spec.stage2_dir.name}/gap_analysis_pass2_searched.parquet …")
    gap_candidates = load_gap_candidates(
        spec.stage2_dir / "gap_analysis_pass2_searched.parquet"
    )

    # Drop leaked event_ids from gap_candidates to match the rest of the context.
    if ctx["mdace_leak_dropped"]:
        for eid in list(gap_candidates):
            if eid in ctx["mdace_leak_dropped"]:
                del gap_candidates[eid]

    oracle_sets = _build_oracle_sets(
        sft_macro=sft_macro, sft_micro=sft_micro,
        band_verify=band_verify, candidate_verify=candidate_verify,
        gap_candidates=gap_candidates, gold_lookup=gold_lookup,
    )

    # Bootstrap each oracle config across scopes.
    rows: list[dict] = []
    full_micro_prev: float | None = None
    full_macro_prev: float | None = None
    for cond in ORACLE_LADDER:
        print(f"  bootstrapping {cond} (n={n}, seed={seed}) …")
        # compose_predictions just unions over all event_ids; here oracle_sets[cond]
        # is already a complete per-event lookup. Pass through unchanged.
        scope_metrics, _tables = _scope_metrics_for_pred(
            oracle_sets[cond], ctx, n=n, seed=seed,
            label=f"{spec.name}/oracle/{cond}",
        )
        full_micro = scope_metrics["full"]["f1_micro"]
        full_macro = scope_metrics["full"]["f1_macro"]
        delta_full = {
            "f1_micro": (full_micro - full_micro_prev) if full_micro_prev is not None else None,
            "f1_macro": (full_macro - full_macro_prev) if full_macro_prev is not None else None,
        }
        rows.append({
            "name": cond,
            "scope_metrics": scope_metrics,
            "delta_full_vs_previous": delta_full,
        })
        full_micro_prev, full_macro_prev = full_micro, full_macro

    waterfall_payload = {
        "dataset": spec.name,
        "thresholds": {"tau_macro": spec.tau_macro, "tau_micro": spec.tau_micro},
        "thresholds_source": str(spec.thresholds_source.relative_to(REPO_ROOT)),
        "n_admissions": n_adm,
        "n_resamples": n,
        "seed": seed,
        "filter_mdace_leak": ctx["filter_mdace_leak"],
        "ladder": ORACLE_LADDER,
        "definitions": {
            "C0":      "SFT >= tau_macro (band lower edge baseline)",
            "C0_high": "SFT >= tau_micro (band upper edge baseline = sft_only_tau_micro)",
            "C0_5":    "SFT >= tau_micro UNION band_verifier_confirmed (= sft_band_verify, real)",
            "C1":      "SFT >= tau_micro UNION (band_input INTERSECT gold)  -- oracle band verifier",
            "C2":      "C0_5 UNION candidate_verifier_confirmed (= full_pipeline, real)",
            "C3":      "C1 UNION (candidate_pool INTERSECT gold)  -- oracle gap-fill",
        },
        "rows": rows,
    }

    attribution_payload = {
        "dataset": spec.name,
        "thresholds": {"tau_macro": spec.tau_macro, "tau_micro": spec.tau_micro},
        "n_admissions": n_adm,
        "filter_mdace_leak": ctx["filter_mdace_leak"],
        "baseline": "sft_only_tau_micro",
        "transitions": ["verifier", "gap_fill"],
        **_delta_attribution(
            sft_micro=sft_micro, band_verify=band_verify,
            candidate_verify=candidate_verify, gold_lookup=gold_lookup,
        ),
    }

    return waterfall_payload, attribution_payload


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    n = 1000
    seed = 314
    filter_mdace_leak: bool | None = None
    datasets = [SPEC]
    for arg in argv:
        if arg.startswith("--n="):
            n = int(arg.split("=", 1)[1])
        elif arg.startswith("--seed="):
            seed = int(arg.split("=", 1)[1])
        elif arg == "--filter-mdace-leak":
            filter_mdace_leak = True
        elif arg == "--no-filter-mdace-leak":
            filter_mdace_leak = False
        elif arg in {"-h", "--help"}:
            print(__doc__)
            print("\nFlags: --n=<int> --seed=<int> "
                  "--filter-mdace-leak | --no-filter-mdace-leak")
            return 0
        else:
            print(f"unknown argument: {arg!r}", file=sys.stderr)
            return 2

    out_root = OUT_DIR / "error_analyses"
    for spec in datasets:
        waterfall, attribution = _waterfall_for_spec(
            spec, n=n, seed=seed, filter_mdace_leak=filter_mdace_leak,
        )
        ds_dir = out_root / spec.name
        ds_dir.mkdir(parents=True, exist_ok=True)
        wf_path = ds_dir / "oracle_waterfall.json"
        wf_path.write_text(json.dumps(_round_floats(waterfall), indent=2) + "\n")
        print(f"  wrote {wf_path.relative_to(REPO_ROOT)}")
        attr_path = ds_dir / "delta_attribution.json"
        attr_path.write_text(json.dumps(_round_floats(attribution), indent=2) + "\n")
        print(f"  wrote {attr_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
