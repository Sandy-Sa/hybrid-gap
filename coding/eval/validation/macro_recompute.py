"""Recompute macro-F1 on the scorable set, side-by-side with the legacy full-vocab macro.

Reads cached SFT posteriors + Stage-2 verdicts via `_load_dataset_context` —
no model re-inference. For every (system × test_set × condition) cell:

  * loads the same prediction lookup that `re_eval()` writes today
  * scores it once with `macro_denominator="full"` and once with `="scorable"`
  * asserts micro-F1 reproduces the value in the existing per-config JSON
    (the smoke-test for the change), and that macro_f1_full does too
  * appends a row to `outputs/revision/metrics_recomputed.parquet`

Run with: ``uv run python revision/recompute_all_macro.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from coding.eval.harness import (
    MIMIC_III_DIR,
    OUT_DIR,
    REPO_ROOT,
    SPEC,
    _load_dataset_context,
    _score_one_scope,
    _scope_vocabs,
    compose_predictions,
    load_gap_candidates,
)


SCOPE_TO_KEY = {"in_dist": "in_dist", "ood": "ood", "full": "full"}


def _existing_metrics(json_path: Path) -> dict[str, dict[str, float]]:
    """Load the existing per-config JSON and return scope_metrics dict."""
    if not json_path.exists():
        return {}
    with json_path.open() as f:
        payload = json.load(f)
    return payload.get("scope_metrics", {})


def _gap_candidates_path(spec) -> Path:
    """Path to gap_analysis_pass2_searched.parquet for the given spec."""
    return MIMIC_III_DIR / "gap_analysis_pass2_searched.parquet"


def _config_to_existing_json(out_dir: Path, spec, config: str) -> Path:
    """Map a config name to the JSON file produced by `re_eval()`."""
    base = out_dir / "results"
    if config == "sft_only_tau_macro":
        return base / "sft_only" / spec.name / "tau_macro.json"
    if config == "sft_only_tau_micro":
        return base / "sft_only" / spec.name / "tau_micro.json"
    if config == "sft_band_verify":
        return base / "sft_band_verify" / spec.name / f"band_{spec.tau_macro:.3f}-{spec.tau_micro:.3f}.json"
    if config == "full_pipeline":
        return base / "full_pipeline" / spec.name / f"full_band_{spec.tau_macro:.3f}-{spec.tau_micro:.3f}.json"
    # +gap_fill is an intermediate stage, persisted separately under gap_fill_intermediate/
    if config == "gap_fill_intermediate":
        return base / "gap_fill_intermediate" / spec.name / "raw_gap_fill.json"
    return Path()


def _round4_eq(a: float, b: float) -> bool:
    """Existing JSONs were rounded to 4 dp; check round-trip equality."""
    if a != a or b != b:  # NaN
        return False
    return round(a, 4) == round(b, 4)


def recompute_for_spec(spec, *, filter_mdace_leak: bool | None = None) -> list[dict]:
    print(f"\n=== recompute_all_macro: {spec.name} ===")
    ctx = _load_dataset_context(
        spec,
        sft_thresholds=[spec.tau_macro, spec.tau_micro],
        filter_mdace_leak=filter_mdace_leak,
    )
    sft_macro = ctx["sft_lookups"][spec.tau_macro]
    sft_micro = ctx["sft_lookups"][spec.tau_micro]
    band_verify = ctx["band_verify"]
    candidate_verify = ctx["candidate_verify"]

    # +gap_fill (raw candidates before LLM verification).
    gap_lookup = load_gap_candidates(_gap_candidates_path(spec))
    if ctx["filter_mdace_leak"] and ctx["mdace_leak_dropped"]:
        for eid in list(gap_lookup.keys()):
            if eid not in set(ctx["event_ids"]):
                del gap_lookup[eid]

    systems = {
        "sft_only_tau_macro":      compose_predictions(sft_macro),
        "sft_only_tau_micro":      compose_predictions(sft_micro),                # == sft_high stage
        "sft_band_verify":         compose_predictions(sft_micro, band_verify),   # == +band_verify stage
        "gap_fill_intermediate":   compose_predictions(sft_micro, band_verify, gap_lookup),   # == +gap_fill stage
        "full_pipeline":           compose_predictions(sft_micro, band_verify, candidate_verify),  # == +verify_gap_fill / hybrid
    }

    scope_vocabs = _scope_vocabs(ctx)
    event_ids = ctx["event_ids"]
    gold_lookup = ctx["gold_lookup"]

    rows: list[dict] = []
    for sys_name, pred_lookup in systems.items():
        existing = _existing_metrics(_config_to_existing_json(OUT_DIR, spec, sys_name))
        pred_sets = [pred_lookup.get(eid, set()) for eid in event_ids]
        gold_sets = [gold_lookup.get(eid, set()) for eid in event_ids]
        for scope, vocab in scope_vocabs.items():
            full_m = _score_one_scope(pred_sets, gold_sets, vocab, macro_denominator="full")
            scor_m = _score_one_scope(pred_sets, gold_sets, vocab, macro_denominator="scorable")
            # Reproducibility check vs the existing JSON (rounded to 4dp on disk).
            exp_scope = existing.get(SCOPE_TO_KEY[scope], {})
            mismatches: list[str] = []
            if exp_scope:
                if not _round4_eq(full_m["f1_micro"], exp_scope.get("f1_micro", float("nan"))):
                    mismatches.append(
                        f"micro {full_m['f1_micro']:.6f} vs cached {exp_scope.get('f1_micro')}"
                    )
                if not _round4_eq(full_m["f1_macro"], exp_scope.get("f1_macro", float("nan"))):
                    mismatches.append(
                        f"macro_full {full_m['f1_macro']:.6f} vs cached {exp_scope.get('f1_macro')}"
                    )
            status = "OK" if not mismatches else "MISMATCH: " + "; ".join(mismatches)
            macro_delta = (scor_m["f1_macro"] - full_m["f1_macro"]) if scor_m["f1_macro"] == scor_m["f1_macro"] else float("nan")
            print(
                f"  {sys_name:30s} | {scope:7s} "
                f"| micro={full_m['f1_micro']:.4f} "
                f"| macro_full={full_m['f1_macro']:.4f} "
                f"| macro_scor={scor_m['f1_macro'] if scor_m['f1_macro'] == scor_m['f1_macro'] else float('nan'):>7.4f} "
                f"| Δ={macro_delta:+.4f} "
                f"| N_scor={full_m['n_scorable']:5d}/{full_m['n_in_scope']:5d} "
                f"| {status}"
            )
            rows.append({
                "system": sys_name,
                "test_set": spec.name,
                "condition": scope,
                "micro_f1": full_m["f1_micro"],
                "macro_f1_full": full_m["f1_macro"],
                "macro_f1_scorable": scor_m["f1_macro"],
                "precision_macro_full": full_m["precision_macro"],
                "precision_macro_scorable": scor_m["precision_macro"],
                "recall_macro_full": full_m["recall_macro"],
                "recall_macro_scorable": scor_m["recall_macro"],
                "n_scorable": full_m["n_scorable"],
                "n_in_scope": full_m["n_in_scope"],
                "filter_mdace_leak": bool(ctx["filter_mdace_leak"]),
                "reproducibility_status": status,
            })
    return rows


def main() -> int:
    all_rows: list[dict] = []
    for spec in (SPEC,):
        all_rows.extend(recompute_for_spec(spec))

    df = pl.DataFrame(all_rows)
    out_path = OUT_DIR / "metrics_recomputed.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)} ({df.height} rows)")

    n_bad = df.filter(~pl.col("reproducibility_status").str.starts_with("OK")).height
    if n_bad:
        print(f"\n*** {n_bad} reproducibility mismatches — see status column ***")
        return 1
    print("All cells reproduce existing micro_f1 + macro_f1_full to 4dp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
