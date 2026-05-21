"""Phase 6.3 + 6.4 — chapter-x-stage heatmap and OOD/InDist decomposition.

For each dataset spec, emits
``outputs/revision/error_analyses/<dataset>/chapter_heatmap.json`` containing:

  * ``by_subset`` (Phase 6.4): scope-level micro/macro F1 per stage. Reuses
    ``_scope_metrics_for_pred``, so this matches the headline numbers in
    ``outputs/revision/results/{sft_only,sft_band_verify,full_pipeline}/``.

  * ``by_chapter`` (Phase 6.3): per-(scope, chapter, stage) binary P/R/F1
    (each event-chapter pair as a binary classification target), plus
    per-(scope, chapter, transition, metric) deltas.

Stages (mirroring chapter_subset_analysis_mimic_iii.ipynb at the new band):
  * ``sft_high``           = SFT scores >= tau_micro
  * ``sft_post_in_band``   = sft_high U band_verifier_confirmed
  * ``gapfill_pre_verify`` = sft_post_in_band U gap_search_top_N (raw retrieval)
  * ``final``              = sft_post_in_band U candidate_verifier_confirmed
                              (= full_pipeline)

Transitions: D_in_band_verify, D_gapfill, D_verify_gap_fill, D_total.

With ``--bootstrap-chapter-deltas`` the OOD D_total ΔF1 rows in
``chapter_deltas.rows`` get a ``ci_normal`` field (point ± 1.96·SE_boot from
admission-level resampling, seed 314), used by Appendix Tables
tab:ood-chapter-diag and tab:ood-chapter-proc.

Usage::

    cd /Users/full_of_bugs/repos/hybrid_coding
    uv run --project eval python -m eval.decomposition.chapter \\
        [--n=1000] [--bootstrap-chapter-deltas]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np

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

MacroDenominator = Literal["full", "scorable"]


# Stage names match chapter_subset_analysis_mimic_iii.ipynb cell 3.
STAGE_ORDER = ["sft_high", "sft_post_in_band", "gapfill_pre_verify", "final"]
TRANSITIONS = {
    "D_in_band_verify":  ("sft_post_in_band",   "sft_high"),
    "D_gapfill":         ("gapfill_pre_verify", "sft_post_in_band"),
    "D_verify_gap_fill": ("final",              "gapfill_pre_verify"),
    "D_total":           ("final",              "sft_high"),
}
SUBSETS = ["in_dist", "ood", "full"]
MIN_GOLD_PER_CHAPTER = 30
METRICS = ("precision", "recall", "f1")


from coding.utils import icd_hierarchy as ih  # noqa: E402


def _build_stage_lookups(
    *,
    sft_micro: dict[str, set[str]],
    band_verify: dict[str, set[str]],
    candidate_verify: dict[str, set[str]],
    gap_candidates: dict[str, set[str]],
    event_ids: list[str],
) -> dict[str, dict[str, set[str]]]:
    """Per-stage per-event predicted code sets."""
    out = {st: {} for st in STAGE_ORDER}
    for eid in event_ids:
        sh = sft_micro.get(eid, set())
        bv = band_verify.get(eid, set())
        sn = gap_candidates.get(eid, set())
        cv = candidate_verify.get(eid, set())
        sft_post = sh | bv
        out["sft_high"][eid] = set(sh)
        out["sft_post_in_band"][eid] = sft_post
        out["gapfill_pre_verify"][eid] = sft_post | sn
        out["final"][eid] = sft_post | cv
    return out


def _scope_vocabs_for_chapters(ctx: dict) -> dict[str, set[str]]:
    """Same vocab map as ``_scope_vocabs`` but exposed for chapter masking."""
    in_combined = ctx["in_class_maps"][0]
    ood_combined = ctx["ood_class_maps_global"][0]
    full_combined = ctx["full_class_maps_global"][0]
    return {
        "in_dist": set(in_combined),
        "ood": set(ood_combined),
        "full": set(full_combined),
    }


def _binary_prf1(pred_vec: np.ndarray, gold_vec: np.ndarray) -> tuple[float, float, float, int]:
    tp = int(((pred_vec == 1) & (gold_vec == 1)).sum())
    fp = int(((pred_vec == 1) & (gold_vec == 0)).sum())
    fn = int(((pred_vec == 0) & (gold_vec == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return p, r, f, tp + fn


def _chapter_metrics(
    *,
    spec: DatasetSpec,
    stage_lookups: dict[str, dict[str, set[str]]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    scope_vocabs: dict[str, set[str]],
) -> tuple[dict, dict, dict]:
    """Compute per-(scope, chapter, stage) binary P/R/F1 and per-transition deltas.

    Returns ``(by_chapter_payload, delta_payload, status)``. When the chapter
    mapping cannot be built (missing ICD-10 reference XML), payloads are empty
    and ``status['skipped'] = True`` with a reason.
    """
    code_type = spec.code_type

    # Roll-up cache: code -> chapter id (None if unresolved).
    chap_of: dict[str, str | None] = {}

    def chapter(code: str) -> str | None:
        if code not in chap_of:
            try:
                chap_of[code] = ih.code_to_chapter(code, code_type)
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"chapter mapping unavailable for {code_type}: {e}"
                ) from e
        return chap_of[code]

    # Per-event chapter sets, masked by scope vocab.
    n = len(event_ids)
    eid_index = {eid: i for i, eid in enumerate(event_ids)}

    chapter_metrics: dict[tuple[str, str], dict[str, dict]] = {}
    delta_rows: list[dict] = []

    for subset, vocab in scope_vocabs.items():
        # Build per-event chapter-set for gold within this subset.
        gold_chap: dict[str, set[str]] = {}
        for eid in event_ids:
            chaps: set[str] = set()
            for c in gold_lookup.get(eid, set()) & vocab:
                ch = chapter(c)
                if ch is not None:
                    chaps.add(ch)
            gold_chap[eid] = chaps

        # Support per chapter = number of events with at least one gold code in that chapter.
        support: dict[str, int] = {}
        for chaps in gold_chap.values():
            for ch in chaps:
                support[ch] = support.get(ch, 0) + 1
        kept_chapters = [ch for ch, s in support.items() if s >= MIN_GOLD_PER_CHAPTER]

        # Build pred chapter-sets per stage.
        pred_chap_per_stage: dict[str, dict[str, set[str]]] = {}
        for st in STAGE_ORDER:
            pred_chap: dict[str, set[str]] = {}
            for eid in event_ids:
                chaps: set[str] = set()
                for c in stage_lookups[st].get(eid, set()) & vocab:
                    ch = chapter(c)
                    if ch is not None:
                        chaps.add(ch)
                pred_chap[eid] = chaps
            pred_chap_per_stage[st] = pred_chap

        # Per-chapter per-stage binary P/R/F1.
        per_chapter_stage: dict[str, dict[str, dict]] = {}
        for ch in kept_chapters:
            gold_vec = np.zeros(n, dtype=np.uint8)
            for eid, chaps in gold_chap.items():
                if ch in chaps:
                    gold_vec[eid_index[eid]] = 1

            stage_metrics: dict[str, dict] = {}
            for st in STAGE_ORDER:
                pred_vec = np.zeros(n, dtype=np.uint8)
                for eid, chaps in pred_chap_per_stage[st].items():
                    if ch in chaps:
                        pred_vec[eid_index[eid]] = 1
                p, r, f, sup = _binary_prf1(pred_vec, gold_vec)
                stage_metrics[st] = {
                    "precision": p, "recall": r, "f1": f, "support": sup,
                }
            per_chapter_stage[ch] = stage_metrics

            # Per-transition metric deltas.
            chapter_name = ih.chapter_name(ch) or ch
            for trans, (b_st, a_st) in TRANSITIONS.items():
                for metric in METRICS:
                    delta_rows.append({
                        "subset": subset,
                        "chapter_id": ch,
                        "chapter_name": chapter_name,
                        "transition": trans,
                        "metric": metric,
                        "value": stage_metrics[b_st][metric] - stage_metrics[a_st][metric],
                    })

        chapter_metrics[(subset, "chapter")] = {
            "kept_chapters": kept_chapters,
            "min_support": MIN_GOLD_PER_CHAPTER,
            "support": support,
            "by_chapter": per_chapter_stage,
        }

    by_chapter_payload = {
        subset: {
            "kept_chapters": chapter_metrics[(subset, "chapter")]["kept_chapters"],
            "min_support": MIN_GOLD_PER_CHAPTER,
            "support": chapter_metrics[(subset, "chapter")]["support"],
            "by_chapter": chapter_metrics[(subset, "chapter")]["by_chapter"],
        }
        for subset in SUBSETS
    }
    delta_payload = {
        "transitions": list(TRANSITIONS),
        "metrics": list(METRICS),
        "rows": delta_rows,
    }
    return by_chapter_payload, delta_payload, {"skipped": False}


def _bootstrap_chapter_deltas(
    *,
    spec: DatasetSpec,
    stage_lookups: dict[str, dict[str, set[str]]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    subset_vocabs: dict[str, set[str]],
    deltas: dict,
    transitions: dict[str, tuple[str, str]],
    n: int,
    seed: int,
) -> None:
    """Add normal-approx 95% CIs to ΔF1 rows of *deltas* in place.

    Resamples admissions with replacement *n* times (seed) and, for each
    (subset, chapter, transition) cell, computes the CI of the F1 delta from
    the per-resample stage F1 differences. Updates only ``metric == "f1"``
    rows that match a (subset, transition) the caller asked for.

    *transitions* maps transition name → (after_stage, before_stage). All
    stages referenced must be keys of ``stage_lookups``.
    *subset_vocabs* maps subset name → vocab; only rows with these subsets
    are touched.
    """
    rows = deltas["rows"]
    target_rows = [
        r for r in rows
        if r["subset"] in subset_vocabs
        and r["transition"] in transitions
        and r["metric"] == "f1"
    ]
    if not target_rows:
        print("  bootstrap_chapter_deltas: no eligible rows; skipping")
        return

    chap_of: dict[str, str | None] = {}

    def chapter(code: str) -> str | None:
        if code not in chap_of:
            try:
                chap_of[code] = ih.code_to_chapter(code, spec.code_type)
            except FileNotFoundError as e:
                raise RuntimeError(f"chapter mapping unavailable: {e}") from e
        return chap_of[code]

    n_adm = len(event_ids)
    eid_index = {eid: i for i, eid in enumerate(event_ids)}
    rng = np.random.default_rng(seed)
    z = 1.959963984540054  # Φ⁻¹(0.975)
    stage_keys_needed: set[str] = set()
    for after, before in transitions.values():
        stage_keys_needed.update([after, before])
    stage_keys = sorted(stage_keys_needed)
    n_subsets_done = 0

    # Group target_rows by (subset, transition) for quick lookup later.
    rows_by_key: dict[tuple[str, str, str], dict] = {
        (r["subset"], r["chapter_id"], r["transition"]): r for r in target_rows
    }

    for subset, vocab in subset_vocabs.items():
        # Collect chapters present in this subset's target rows.
        chapters = sorted({
            r["chapter_id"] for r in target_rows if r["subset"] == subset
        })
        if not chapters:
            continue
        n_chap = len(chapters)
        chap_to_col = {ch: j for j, ch in enumerate(chapters)}

        # Gold and per-stage chapter-presence indicators for this subset.
        gold_mat = np.zeros((n_adm, n_chap), dtype=np.uint8)
        pred_mats: dict[str, np.ndarray] = {
            st: np.zeros((n_adm, n_chap), dtype=np.uint8) for st in stage_keys
        }
        for eid in event_ids:
            i = eid_index[eid]
            for c in gold_lookup.get(eid, set()) & vocab:
                ch = chapter(c)
                if ch is not None and ch in chap_to_col:
                    gold_mat[i, chap_to_col[ch]] = 1
            for st in stage_keys:
                for c in stage_lookups[st].get(eid, set()) & vocab:
                    ch = chapter(c)
                    if ch is not None and ch in chap_to_col:
                        pred_mats[st][i, chap_to_col[ch]] = 1

        print(
            f"  bootstrap_chapter_deltas[{subset}]: {n} resamples × "
            f"{n_chap} chapters × {len(transitions)} transitions …"
        )
        # For each transition, accumulate delta across resamples.
        deltas_boot = {
            t: np.empty((n, n_chap), dtype=np.float64) for t in transitions
        }
        sub_rng = np.random.default_rng(seed)  # deterministic per-subset
        for b in range(n):
            sample = sub_rng.integers(0, n_adm, n_adm)
            gs = gold_mat[sample]
            f1_per_stage: dict[str, np.ndarray] = {}
            for st in stage_keys:
                ps = pred_mats[st][sample]
                tp = np.sum((ps == 1) & (gs == 1), axis=0)
                fp = np.sum((ps == 1) & (gs == 0), axis=0)
                fn = np.sum((ps == 0) & (gs == 1), axis=0)
                denom = 2 * tp + fp + fn
                f1_per_stage[st] = np.where(
                    denom > 0, 2 * tp / np.maximum(denom, 1), 0.0
                )
            for t, (after, before) in transitions.items():
                deltas_boot[t][b] = f1_per_stage[after] - f1_per_stage[before]

        for t in transitions:
            se_per_chapter = deltas_boot[t].std(axis=0, ddof=1)
            boot_mean_per_chapter = deltas_boot[t].mean(axis=0)
            for ch in chapters:
                key = (subset, ch, t)
                r = rows_by_key.get(key)
                if r is None:
                    continue
                point = r["value"]
                se = float(se_per_chapter[chap_to_col[ch]])
                r["ci_normal"] = [float(point - z * se), float(point + z * se)]
                r["se_boot"] = se
                r["boot_mean"] = float(boot_mean_per_chapter[chap_to_col[ch]])
        n_subsets_done += 1

    deltas["bootstrap_meta"] = {
        "applied_to": [
            f"{s}/{t}/f1" for s in subset_vocabs for t in transitions
        ],
        "n_resamples": n,
        "seed": seed,
        "ci_kind": "normal_approx",
    }


def _bootstrap_chapter_deltas_ood(
    *,
    spec: DatasetSpec,
    stage_lookups: dict[str, dict[str, set[str]]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    ood_vocab: set[str],
    deltas: dict,
    n: int,
    seed: int,
) -> None:
    """Back-compat shim — bootstrap CIs for OOD/D_total/f1 only."""
    _bootstrap_chapter_deltas(
        spec=spec, stage_lookups=stage_lookups, gold_lookup=gold_lookup,
        event_ids=event_ids,
        subset_vocabs={"ood": ood_vocab},
        deltas=deltas,
        transitions={"D_total": TRANSITIONS["D_total"]},
        n=n, seed=seed,
    )


def _by_subset_metrics(
    spec: DatasetSpec,
    *,
    ctx: dict,
    stage_lookups: dict[str, dict[str, set[str]]],
    n: int,
    seed: int,
    macro_denominator: MacroDenominator = "full",
) -> dict:
    """Bootstrap micro/macro F1 per stage across {in_dist, ood, full}.

    Uses the same machinery as ``re_eval`` so numbers reconcile against the
    Phase 3/4/5 results JSONs.
    """
    out: dict[str, dict] = {}
    for st in STAGE_ORDER:
        pred_lookup = stage_lookups[st]
        scope_metrics, _tables = _scope_metrics_for_pred(
            pred_lookup, ctx, n=n, seed=seed,
            label=f"{spec.name}/by_subset/{st}",
            macro_denominator=macro_denominator,
        )
        out[st] = scope_metrics
    return out


def _decompose_for_spec(
    spec: DatasetSpec,
    *,
    n: int,
    seed: int,
    filter_mdace_leak: bool | None,
    bootstrap_chapter_deltas: bool = False,
    macro_denominator: MacroDenominator = "full",
) -> dict:
    print(f"\n=== chapter_decomposition: {spec.name} ===")
    ctx = _load_dataset_context(
        spec,
        sft_thresholds=[spec.tau_macro, spec.tau_micro],
        filter_mdace_leak=filter_mdace_leak,
    )
    sft_micro = ctx["sft_lookups"][spec.tau_micro]
    band_verify = ctx["band_verify"]
    candidate_verify = ctx["candidate_verify"]
    n_adm = len(ctx["event_ids"])

    print(f"  loading gap candidate pool from {spec.stage2_dir.name}/gap_analysis_pass2_searched.parquet …")
    gap_candidates = load_gap_candidates(
        spec.stage2_dir / "gap_analysis_pass2_searched.parquet"
    )
    if ctx["mdace_leak_dropped"]:
        for eid in list(gap_candidates):
            if eid in ctx["mdace_leak_dropped"]:
                del gap_candidates[eid]

    stage_lookups = _build_stage_lookups(
        sft_micro=sft_micro, band_verify=band_verify,
        candidate_verify=candidate_verify, gap_candidates=gap_candidates,
        event_ids=ctx["event_ids"],
    )

    print(f"  computing by-subset metrics across stages (macro_denominator={macro_denominator}) …")
    by_subset = _by_subset_metrics(
        spec, ctx=ctx, stage_lookups=stage_lookups, n=n, seed=seed,
        macro_denominator=macro_denominator,
    )

    print("  computing per-chapter metrics and deltas …")
    scope_vocabs = _scope_vocabs_for_chapters(ctx)
    try:
        by_chapter, deltas, chapter_status = _chapter_metrics(
            spec=spec, stage_lookups=stage_lookups,
            gold_lookup=ctx["gold_lookup"], event_ids=ctx["event_ids"],
            scope_vocabs=scope_vocabs,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"  per-chapter analysis SKIPPED for {spec.name}: {exc}")
        by_chapter = {subset: {} for subset in SUBSETS}
        deltas = {"transitions": list(TRANSITIONS), "metrics": list(METRICS), "rows": []}
        chapter_status = {
            "skipped": True,
            "reason": str(exc),
            "remediation": (
                "ICD-10-CM tabular XML missing under "
                "prompt-icd/data/reference/icd10cm_table_2022/. Place the "
                "reference XML there or run on Gadi where the file is present."
            ),
        }

    if bootstrap_chapter_deltas and not chapter_status.get("skipped", False):
        try:
            # OOD/D_total CIs feed Appendix Tables tab:ood-chapter-{diag,proc};
            # Full subset CIs across all 4 transitions feed the per-chapter
            # stage-deltas figures (Figs.~tab:chapter-stage-{diag,proc}).
            _bootstrap_chapter_deltas(
                spec=spec, stage_lookups=stage_lookups,
                gold_lookup=ctx["gold_lookup"], event_ids=ctx["event_ids"],
                subset_vocabs={
                    "ood": set(ctx["ood_class_maps_global"][0]),
                    "full": set(ctx["full_class_maps_global"][0]),
                },
                deltas=deltas,
                transitions=TRANSITIONS,
                n=n, seed=seed,
            )
        except RuntimeError as exc:
            print(f"  bootstrap_chapter_deltas SKIPPED: {exc}")

    return {
        "dataset": spec.name,
        "thresholds": {"tau_macro": spec.tau_macro, "tau_micro": spec.tau_micro},
        "thresholds_source": str(spec.thresholds_source.relative_to(REPO_ROOT)),
        "n_admissions": n_adm,
        "n_resamples": n,
        "seed": seed,
        "filter_mdace_leak": ctx["filter_mdace_leak"],
        "macro_denominator": macro_denominator,
        "stage_order": STAGE_ORDER,
        "transitions": TRANSITIONS,
        "by_subset": by_subset,
        "by_chapter": by_chapter,
        "chapter_deltas": deltas,
        "chapter_status": chapter_status,
    }


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    n = 1000
    seed = 314
    filter_mdace_leak: bool | None = None
    bootstrap_chapter_deltas = False
    macro_denominator: MacroDenominator = "full"
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
        elif arg == "--bootstrap-chapter-deltas":
            bootstrap_chapter_deltas = True
        elif arg.startswith("--macro-denominator="):
            choice = arg.split("=", 1)[1]
            if choice not in {"full", "scorable"}:
                print(
                    f"unknown --macro-denominator value: {choice!r} "
                    "(expected 'full' or 'scorable')",
                    file=sys.stderr,
                )
                return 2
            macro_denominator = choice  # type: ignore[assignment]
        elif arg in {"-h", "--help"}:
            print(__doc__)
            print("\nFlags: --n=<int> --seed=<int> "
                  "--filter-mdace-leak | --no-filter-mdace-leak "
                  "--bootstrap-chapter-deltas "
                  "--macro-denominator=full|scorable")
            return 0
        else:
            print(f"unknown argument: {arg!r}", file=sys.stderr)
            return 2

    out_root = OUT_DIR / "error_analyses"
    out_filename = (
        "chapter_heatmap.json" if macro_denominator == "full"
        else "chapter_heatmap_scorable.json"
    )
    for spec in datasets:
        payload = _decompose_for_spec(
            spec, n=n, seed=seed, filter_mdace_leak=filter_mdace_leak,
            bootstrap_chapter_deltas=bootstrap_chapter_deltas,
            macro_denominator=macro_denominator,
        )
        ds_dir = out_root / spec.name
        ds_dir.mkdir(parents=True, exist_ok=True)
        out_path = ds_dir / out_filename
        out_path.write_text(json.dumps(_round_floats(payload), indent=2) + "\n")
        print(f"  wrote {out_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
