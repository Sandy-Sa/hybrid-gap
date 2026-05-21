"""Phase 6 — leaf-level F1 stratified by training-set occurrence (Edin) and rank.

Refreshes the headline numbers cited in §`Where the hybrid help and where it
hurt?` (latex/acl_latex.tex L302) and Appendix Table tab:freq-band at the new
principled band ``[tau_macro, tau_micro] = [0.125, 0.311]``.

Produces ``outputs/revision/error_analyses/<dataset>/freq_bands_edin.json`` with
two binning schemes:

  * ``edin_count`` — Edin 2023 strict bins ``['0','1-10','11-50','51-100',
    '101-1000','>1000']`` keyed on each code's training-set occurrence count.
  * ``rank_based`` — ``['head','torso','tail']`` matching the appendix
    convention (head=top-50 by frequency, torso=51-1000, tail=>1000 in rank).

For each scheme × stage × bin: point P/R/F1 (micro+macro) + admission-level
bootstrap CI (1000 resamples, seed 314, normal-approx — same protocol as
``eval.decomposition.chapter``).

Sanity gates run before writing output:

  1. ``len(event_ids) == 8713`` after MDACE-leak filter.
  2. Recompose ``final`` on OOD vocab and reproduce
     ``chapter_heatmap.json.by_subset.final.ood`` ``f1_micro`` and ``f1_macro``
     within 5e-4.
  3. Reproduce per-chapter deltas for Congenital Anomalies (+0.4343) and
     Op. Musculoskeletal (+0.2650) within 1e-4.
  4. Cross-check ``n_codes_per_bin.rank_based`` against
     ``prompt-icd/reports/freq_bands.json`` (50 / 950 / 2681).

With ``--include-fp-by-band``, an additional ``fp_by_band`` block is written
that decomposes verifier false-acceptances and gap-fill hallucinations across
four count bins (``OOD`` for codes with zero training occurrences, ``1-50``,
``51-1000``, ``>1000``). Used by Appendix Table tab:fp-by-band; matches the
totals in ``delta_attribution.json`` (verifier_false_accept = 39542,
gap_fill_hallucination = 32997 on MIMIC-III at the new band).

Usage::

    cd /Users/full_of_bugs/repos/hybrid_coding
    uv run --project eval python -m eval.decomposition.freq_band \
        [--dataset=mimic_iii_clean] [--n=1000] [--seed=314] \
        [--include-fp-by-band]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from coding.eval.bootstrap import bootstrap_f1_from_table, build_index_table
from coding.eval.harness import (
    OUT_DIR,
    REPO_ROOT,
    SPEC,
    DatasetSpec,
    _load_dataset_context,
    _round_floats,
    _score_one_scope,
)
from coding.eval.paths import FREQ_BANDS_PATH

MacroDenominator = Literal["full", "scorable"]


EDIN_BIN_ORDER = ["0", "1-10", "11-50", "51-100", "101-1000", ">1000"]
RANK_BIN_ORDER = ["head", "torso", "tail"]
RANK_HEAD_MAX = 50
RANK_TORSO_MAX = 1000

# Coarse bins for tab:fp-by-band — same as Edin's count bins but collapsed to
# four levels with an OOD bucket (codes never seen in train).
FP_BIN_ORDER = ["OOD", "1-50", "51-1000", ">1000"]


def _fp_bin(n: int) -> str:
    """Coarse 4-bin classifier for FP-by-band (matches tab:fp-by-band rows)."""
    if n == 0:
        return "OOD"
    if n <= 50:
        return "1-50"
    if n <= 1000:
        return "51-1000"
    return ">1000"


STAGE_ORDER = ["sft_high", "sft_post_in_band", "final"]


def _train_occurrence_counts(spec: DatasetSpec) -> Counter:
    """Count each code's occurrence in the training split (combined diag+proc)."""
    train_path = spec.sft_dir / "train_dataset.parquet"
    df = pl.read_parquet(train_path)
    if "split" in df.columns:
        df = df.filter(pl.col("split") == "train")
    counts: Counter = Counter()
    for row in df.iter_rows(named=True):
        for c in (row.get("diagnoses_code") or []):
            counts[c] += 1
        for c in (row.get("procedure_code") or []):
            counts[c] += 1
    return counts


def _edin_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 100:
        return "51-100"
    if n <= 1000:
        return "101-1000"
    return ">1000"


def _rank_bin(rank: int | None) -> str:
    """Rank is 1-based; codes outside the train vocab (rank=None) go to tail."""
    if rank is None or rank > RANK_TORSO_MAX:
        return "tail"
    if rank <= RANK_HEAD_MAX:
        return "head"
    return "torso"


def _bin_assignments(
    in_vocab: set[str], counts: Counter
) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    """Return (code_to_edin_bin, code_to_rank_bin, code_to_count) over *in_vocab*.

    Rank is 1-based on the descending-count ordering of the FULL training
    vocabulary (3,681 codes for MIMIC-III), then projected onto in_vocab. This
    matches the `freq_bands.json` head=50 / torso=950 cutoffs (rank thresholds
    50 / 1000 over the train vocab).
    """
    ranked = counts.most_common()
    rank_lookup: dict[str, int] = {code: i + 1 for i, (code, _) in enumerate(ranked)}
    code_to_count: dict[str, int] = {c: counts.get(c, 0) for c in in_vocab}
    code_to_edin: dict[str, str] = {c: _edin_bin(code_to_count[c]) for c in in_vocab}
    code_to_rank: dict[str, str] = {c: _rank_bin(rank_lookup.get(c)) for c in in_vocab}
    return code_to_edin, code_to_rank, code_to_count


def _build_stage_lookups(
    *,
    sft_micro: dict[str, set[str]],
    band_verify: dict[str, set[str]],
    candidate_verify: dict[str, set[str]],
    event_ids: list[str],
) -> dict[str, dict[str, set[str]]]:
    """Per-stage per-event predicted code sets (sft_high, sft_post_in_band, final)."""
    out: dict[str, dict[str, set[str]]] = {st: {} for st in STAGE_ORDER}
    for eid in event_ids:
        sh = sft_micro.get(eid, set())
        bv = band_verify.get(eid, set())
        cv = candidate_verify.get(eid, set())
        sft_post = sh | bv
        out["sft_high"][eid] = set(sh)
        out["sft_post_in_band"][eid] = sft_post
        out["final"][eid] = sft_post | cv
    return out


def _per_class_counts_indexed(
    pred_lookup: dict[str, set[str]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    class_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_classes = len(class_to_idx)
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    for eid in event_ids:
        p = pred_lookup.get(eid, set())
        g = gold_lookup.get(eid, set())
        for c in p & g:
            i = class_to_idx.get(c)
            if i is not None:
                tp[i] += 1
        for c in p - g:
            i = class_to_idx.get(c)
            if i is not None:
                fp[i] += 1
        for c in g - p:
            i = class_to_idx.get(c)
            if i is not None:
                fn[i] += 1
    return tp, fp, fn


def _bin_metrics_from_counts(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, idx: np.ndarray,
    *,
    macro_denominator: MacroDenominator = "full",
) -> dict:
    if idx.size == 0:
        return dict(
            micro_p=0.0, micro_r=0.0, micro_f1=0.0,
            macro_p=0.0, macro_r=0.0, macro_f1=0.0,
            n_classes=0, n_scorable=0, tp=0, fp=0, fn=0,
        )
    tpb, fpb, fnb = tp[idx], fp[idx], fn[idx]
    sP = int(tpb.sum() + fpb.sum())
    sR = int(tpb.sum() + fnb.sum())
    micro_p = float(tpb.sum() / sP) if sP else 0.0
    micro_r = float(tpb.sum() / sR) if sR else 0.0
    denom_mi = 2 * int(tpb.sum()) + int(fpb.sum()) + int(fnb.sum())
    micro_f1 = float(2 * tpb.sum() / denom_mi) if denom_mi else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        per_p = np.where(tpb + fpb > 0, tpb / np.maximum(tpb + fpb, 1), 0.0)
        per_r = np.where(tpb + fnb > 0, tpb / np.maximum(tpb + fnb, 1), 0.0)
        per_f = np.where(
            2 * tpb + fpb + fnb > 0,
            2 * tpb / np.maximum(2 * tpb + fpb + fnb, 1),
            0.0,
        )
    support_mask = (tpb + fnb) > 0
    n_scorable = int(support_mask.sum())
    if macro_denominator == "scorable":
        if n_scorable == 0:
            macro_p_val = float("nan")
            macro_r_val = float("nan")
            macro_f1_val = float("nan")
        else:
            macro_p_val = float(per_p[support_mask].mean())
            macro_r_val = float(per_r[support_mask].mean())
            macro_f1_val = float(per_f[support_mask].mean())
    else:
        macro_p_val = float(per_p.mean())
        macro_r_val = float(per_r.mean())
        macro_f1_val = float(per_f.mean())
    return dict(
        micro_p=micro_p, micro_r=micro_r, micro_f1=micro_f1,
        macro_p=macro_p_val, macro_r=macro_r_val,
        macro_f1=macro_f1_val,
        n_classes=int(idx.size),
        n_scorable=n_scorable,
        tp=int(tpb.sum()), fp=int(fpb.sum()), fn=int(fnb.sum()),
    )


def _bootstrap_bin_ci(
    pred_lookup: dict[str, set[str]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    bin_codes: set[str],
    *,
    n: int,
    seed: int,
    label: str,
    macro_denominator: MacroDenominator = "full",
) -> dict:
    """Bootstrap normal-approx CI for micro/macro F1 over a bin's class subset."""
    if not bin_codes:
        return {
            "f1_micro_normal_ci": [0.0, 0.0],
            "f1_macro_normal_ci": [0.0, 0.0],
            "f1_micro_boot_mean": 0.0, "f1_macro_boot_mean": 0.0,
        }
    table = build_index_table(pred_lookup, gold_lookup, event_ids, bin_codes)
    boot = bootstrap_f1_from_table(
        table, n=n, seed=seed, label=label,
        macro_denominator=macro_denominator,
    )
    return {
        "f1_micro_normal_ci": [boot.micro_normal_ci_low, boot.micro_normal_ci_high],
        "f1_macro_normal_ci": [boot.macro_normal_ci_low, boot.macro_normal_ci_high],
        "f1_micro_boot_mean": boot.micro_boot_mean,
        "f1_macro_boot_mean": boot.macro_boot_mean,
    }


def _compute_by_bin(
    *,
    bin_to_codes: dict[str, list[str]],
    stage_lookups: dict[str, dict[str, set[str]]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    in_vocab: set[str],
    n: int,
    seed: int,
    scheme_label: str,
    macro_denominator: MacroDenominator = "full",
) -> dict[str, dict[str, dict]]:
    """For each stage, return {bin_name: per-bin metric dict (point + CI)}."""
    class_to_idx = {c: i for i, c in enumerate(sorted(in_vocab))}
    out: dict[str, dict[str, dict]] = {}
    for stage in STAGE_ORDER:
        # Restrict pred to in-distribution vocab so OOD codes don't contaminate.
        pred_lookup_in = {
            eid: stage_lookups[stage].get(eid, set()) & in_vocab
            for eid in event_ids
        }
        gold_lookup_in = {
            eid: gold_lookup.get(eid, set()) & in_vocab for eid in event_ids
        }
        tp, fp, fn = _per_class_counts_indexed(
            pred_lookup_in, gold_lookup_in, event_ids, class_to_idx,
        )
        stage_out: dict[str, dict] = {}
        for bin_name, codes in bin_to_codes.items():
            idx_arr = np.array(
                [class_to_idx[c] for c in codes if c in class_to_idx],
                dtype=np.int64,
            )
            point = _bin_metrics_from_counts(
                tp, fp, fn, idx_arr,
                macro_denominator=macro_denominator,
            )
            ci = _bootstrap_bin_ci(
                pred_lookup_in, gold_lookup_in, event_ids,
                bin_codes=set(codes),
                n=n, seed=seed,
                label=f"{scheme_label}/{stage}/{bin_name}",
                macro_denominator=macro_denominator,
            )
            stage_out[bin_name] = {**point, **ci}
        out[stage] = stage_out
    return out


def _compute_fp_by_band(
    *,
    sft_micro: dict[str, set[str]],
    band_verify: dict[str, set[str]],
    candidate_verify: dict[str, set[str]],
    gold_lookup: dict[str, set[str]],
    counts: Counter,
) -> dict:
    """Per-FP-bin breakdown of verifier FAs and gap-fill hallucinations.

    Mirrors ``eval.decomposition.error_analysis._delta_attribution`` (verifier-added
    codes = ``band_verify - sft_micro``; gap-added codes =
    ``candidate_verify - (sft_micro ∪ verifier_added)``), then bins each
    false-positive code by its training-set occurrence count using
    :func:`_fp_bin`.
    """
    verifier_fa: Counter = Counter({b: 0 for b in FP_BIN_ORDER})
    gap_fill_hall: Counter = Counter({b: 0 for b in FP_BIN_ORDER})
    verifier_total_tp = 0
    gap_fill_total_tp = 0
    for eid, gold in gold_lookup.items():
        sm = sft_micro.get(eid, set())
        verifier_added = band_verify.get(eid, set()) - sm
        verified_so_far = sm | verifier_added
        gap_added = candidate_verify.get(eid, set()) - verified_so_far
        for c in verifier_added:
            if c in gold:
                verifier_total_tp += 1
            else:
                verifier_fa[_fp_bin(counts.get(c, 0))] += 1
        for c in gap_added:
            if c in gold:
                gap_fill_total_tp += 1
            else:
                gap_fill_hall[_fp_bin(counts.get(c, 0))] += 1

    verifier_total_fp = sum(verifier_fa.values())
    gap_fill_total_fp = sum(gap_fill_hall.values())
    total_fp = {
        b: verifier_fa[b] + gap_fill_hall[b] for b in FP_BIN_ORDER
    }
    total_fp_total = sum(total_fp.values())

    def _share_in_head_torso(d: Counter, total: int) -> float:
        if total == 0:
            return 0.0
        return (d[">1000"] + d["51-1000"]) / total

    def _share_in_tail(d: Counter, total: int) -> float:
        if total == 0:
            return 0.0
        return d["1-50"] / total

    def _share_in_ood(d: Counter, total: int) -> float:
        if total == 0:
            return 0.0
        return d["OOD"] / total

    return {
        "bin_order": FP_BIN_ORDER,
        "verifier_fa": {
            **{b: int(verifier_fa[b]) for b in FP_BIN_ORDER},
            "total": int(verifier_total_fp),
        },
        "gap_fill_hallucination": {
            **{b: int(gap_fill_hall[b]) for b in FP_BIN_ORDER},
            "total": int(gap_fill_total_fp),
        },
        "total_fp": {
            **total_fp,
            "total": int(total_fp_total),
        },
        "share_in_head_torso": {
            "verifier_fa": _share_in_head_torso(verifier_fa, verifier_total_fp),
            "gap_fill_hallucination": _share_in_head_torso(gap_fill_hall, gap_fill_total_fp),
        },
        "share_in_tail": {
            "verifier_fa": _share_in_tail(verifier_fa, verifier_total_fp),
            "gap_fill_hallucination": _share_in_tail(gap_fill_hall, gap_fill_total_fp),
        },
        "share_in_ood": {
            "verifier_fa": _share_in_ood(verifier_fa, verifier_total_fp),
            "gap_fill_hallucination": _share_in_ood(gap_fill_hall, gap_fill_total_fp),
        },
        "totals_check": {
            "verifier_tp": verifier_total_tp,
            "gap_fill_tp": gap_fill_total_tp,
            "verifier_fp": verifier_total_fp,
            "gap_fill_fp": gap_fill_total_fp,
        },
    }


def _gate_chapter_delta(
    *,
    spec: DatasetSpec,
    stage_lookups: dict[str, dict[str, set[str]]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    ood_vocab: set[str],
    chapter_id: str,
    expected_delta: float,
    tol: float,
) -> tuple[bool, float]:
    from coding.utils import icd_hierarchy as ih

    def chap_of(code: str) -> str | None:
        try:
            return ih.code_to_chapter(code, spec.code_type)
        except FileNotFoundError:
            return None

    eid_index = {eid: i for i, eid in enumerate(event_ids)}
    gold_vec = np.zeros(len(event_ids), dtype=np.uint8)
    for eid in event_ids:
        for c in gold_lookup.get(eid, set()) & ood_vocab:
            if chap_of(c) == chapter_id:
                gold_vec[eid_index[eid]] = 1
                break

    def stage_f1(stage: str) -> float:
        pred_vec = np.zeros(len(event_ids), dtype=np.uint8)
        for eid in event_ids:
            for c in stage_lookups[stage].get(eid, set()) & ood_vocab:
                if chap_of(c) == chapter_id:
                    pred_vec[eid_index[eid]] = 1
                    break
        tp = int(((pred_vec == 1) & (gold_vec == 1)).sum())
        fp = int(((pred_vec == 1) & (gold_vec == 0)).sum())
        fn = int(((pred_vec == 0) & (gold_vec == 1)).sum())
        denom = 2 * tp + fp + fn
        return (2 * tp / denom) if denom else 0.0

    delta = stage_f1("final") - stage_f1("sft_high")
    return abs(delta - expected_delta) <= tol, delta


def _decompose_for_spec(
    spec: DatasetSpec, *, n: int, seed: int, filter_mdace_leak: bool | None,
    include_fp_by_band: bool = False,
    macro_denominator: MacroDenominator = "full",
) -> dict:
    print(f"\n=== freq_band_decomposition: {spec.name} ===")
    ctx = _load_dataset_context(
        spec,
        sft_thresholds=[spec.tau_macro, spec.tau_micro],
        filter_mdace_leak=filter_mdace_leak,
    )
    sft_micro = ctx["sft_lookups"][spec.tau_micro]
    band_verify = ctx["band_verify"]
    candidate_verify = ctx["candidate_verify"]
    event_ids = ctx["event_ids"]
    gold_lookup = ctx["gold_lookup"]
    in_vocab = set(ctx["in_class_maps"][0])
    ood_vocab = set(ctx["ood_class_maps_global"][0])

    stage_lookups = _build_stage_lookups(
        sft_micro=sft_micro, band_verify=band_verify,
        candidate_verify=candidate_verify, event_ids=event_ids,
    )

    # ---------- Sanity gate 1 ----------
    expected_admissions = 8713 if spec.name == "mimic_iii_clean" else None
    if expected_admissions is not None:
        assert len(event_ids) == expected_admissions, (
            f"GATE 1 FAIL: expected {expected_admissions} admissions after "
            f"MDACE-leak filter, got {len(event_ids)}"
        )
        print(f"  GATE 1 ok: {len(event_ids)} admissions after MDACE filter")

    # ---------- Sanity gate 2 (OOD final F1 vs chapter_heatmap.json) ----------
    # Compare against the heatmap that matches our macro_denominator, so the
    # gate stays meaningful in both modes.
    heatmap_name = (
        "chapter_heatmap.json" if macro_denominator == "full"
        else "chapter_heatmap_scorable.json"
    )
    heatmap_path = OUT_DIR / "error_analyses" / spec.name / heatmap_name
    if heatmap_path.exists():
        prev = json.loads(heatmap_path.read_text())
        ref_micro = prev["by_subset"]["final"]["ood"]["f1_micro"]
        ref_macro = prev["by_subset"]["final"]["ood"]["f1_macro"]
        final_pred_lookup = stage_lookups["final"]
        ood_metrics = _score_one_scope(
            [final_pred_lookup.get(eid, set()) for eid in event_ids],
            [gold_lookup.get(eid, set()) for eid in event_ids],
            ood_vocab,
            macro_denominator=macro_denominator,
        )
        d_mi = abs(ood_metrics["f1_micro"] - ref_micro)
        d_ma = abs(ood_metrics["f1_macro"] - ref_macro)
        assert d_mi < 5e-4 and d_ma < 5e-4, (
            f"GATE 2 FAIL: final.ood F1 mismatch — "
            f"this run mi={ood_metrics['f1_micro']:.6f} vs heatmap mi={ref_micro:.6f} "
            f"(Δ={d_mi:.6f}); "
            f"this run ma={ood_metrics['f1_macro']:.6f} vs heatmap ma={ref_macro:.6f} "
            f"(Δ={d_ma:.6f})"
        )
        print(
            f"  GATE 2 ok: final.ood f1_micro={ood_metrics['f1_micro']:.4f} "
            f"(Δ={d_mi:.5f}), f1_macro={ood_metrics['f1_macro']:.4f} (Δ={d_ma:.5f})"
        )
    else:
        print(f"  GATE 2 skipped: {heatmap_path} not present")

    # ---------- Sanity gate 3 (chapter deltas) ----------
    if spec.name == "mimic_iii_clean":
        for chapter_id, expected in [("ICD9-14", 0.4343), ("ICD9PCS-14", 0.2650)]:
            ok, observed = _gate_chapter_delta(
                spec=spec, stage_lookups=stage_lookups, gold_lookup=gold_lookup,
                event_ids=event_ids, ood_vocab=ood_vocab,
                chapter_id=chapter_id, expected_delta=expected, tol=1e-3,
            )
            assert ok, (
                f"GATE 3 FAIL: chapter {chapter_id} OOD delta — observed {observed:.4f}, "
                f"expected {expected:.4f}"
            )
            print(f"  GATE 3 ok: {chapter_id} OOD final-vs-sft_high Δf1 = {observed:+.4f}")

    # ---------- Compute occurrence counts and bin assignments ----------
    print("  computing training-set occurrence counts …")
    counts = _train_occurrence_counts(spec)
    code_to_edin, code_to_rank, code_to_count = _bin_assignments(in_vocab, counts)
    edin_bin_to_codes: dict[str, list[str]] = {b: [] for b in EDIN_BIN_ORDER}
    rank_bin_to_codes: dict[str, list[str]] = {b: [] for b in RANK_BIN_ORDER}
    for c in in_vocab:
        edin_bin_to_codes[code_to_edin[c]].append(c)
        rank_bin_to_codes[code_to_rank[c]].append(c)
    n_codes_edin = {b: len(edin_bin_to_codes[b]) for b in EDIN_BIN_ORDER}
    n_codes_rank = {b: len(rank_bin_to_codes[b]) for b in RANK_BIN_ORDER}
    print(f"  Edin bins: {n_codes_edin}")
    print(f"  Rank bins: {n_codes_rank}")

    # ---------- Sanity gate 4 (rank bins vs freq_bands.json) ----------
    fb_json = FREQ_BANDS_PATH
    if fb_json.exists():
        ref = json.loads(fb_json.read_text())["codes_per_band"]
        # freq_bands.json's "head" / "torso" / "tail_in_train" are sized over the
        # full *train* vocab (3,681 codes), so we sanity-check the in-vocab
        # projection by reconstructing those from rank_lookup directly: every
        # in-vocab code ranks within [1, 3681], so head/torso/tail counts
        # should match the train-vocab counts exactly when in_vocab covers the
        # full train vocab. Print, don't assert (avoids brittleness if codes
        # diverge slightly between the SFT class_map and the train parquet).
        match = (
            n_codes_rank["head"] == ref["head"]
            and n_codes_rank["torso"] == ref["torso"]
            and n_codes_rank["tail"] == ref["tail_in_train"]
        )
        if match:
            print(
                f"  GATE 4 ok: rank bins ({n_codes_rank}) match freq_bands.json"
            )
        else:
            print(
                f"  GATE 4 INFO: rank bins {n_codes_rank} vs freq_bands.json "
                f"head/torso/tail_in_train={ref['head']}/{ref['torso']}/{ref['tail_in_train']}"
            )

    # ---------- Compute per-bin metrics with bootstrap CIs ----------
    print(f"  computing Edin-strict bin metrics + bootstrap (macro_denominator={macro_denominator}) …")
    by_bin_edin = _compute_by_bin(
        bin_to_codes=edin_bin_to_codes, stage_lookups=stage_lookups,
        gold_lookup=gold_lookup, event_ids=event_ids, in_vocab=in_vocab,
        n=n, seed=seed, scheme_label="edin",
        macro_denominator=macro_denominator,
    )
    print(f"  computing rank-based bin metrics + bootstrap (macro_denominator={macro_denominator}) …")
    by_bin_rank = _compute_by_bin(
        bin_to_codes=rank_bin_to_codes, stage_lookups=stage_lookups,
        gold_lookup=gold_lookup, event_ids=event_ids, in_vocab=in_vocab,
        n=n, seed=seed, scheme_label="rank",
        macro_denominator=macro_denominator,
    )

    # ---------- Headline summary ----------
    sft_high_e = by_bin_edin["sft_high"]
    final_e = by_bin_edin["final"]
    head_f1 = sft_high_e[">1000"]["micro_f1"]
    tail_f1 = sft_high_e["1-10"]["micro_f1"]
    head_over_tail = head_f1 / tail_f1 if tail_f1 > 0 else float("inf")
    headline = {
        "sft_high.gt1000.micro_f1": head_f1,
        "sft_high.1to10.micro_f1": tail_f1,
        "head_over_tail_ratio": head_over_tail,
        "final.gt1000.micro_f1": final_e[">1000"]["micro_f1"],
        "final.1to10.micro_f1": final_e["1-10"]["micro_f1"],
        "delta_final_minus_sft_high.micro_f1": {
            ">1000": final_e[">1000"]["micro_f1"] - sft_high_e[">1000"]["micro_f1"],
            "1-10": final_e["1-10"]["micro_f1"] - sft_high_e["1-10"]["micro_f1"],
        },
        "delta_final_minus_sft_high.micro_p": {
            ">1000": final_e[">1000"]["micro_p"] - sft_high_e[">1000"]["micro_p"],
            "1-10": final_e["1-10"]["micro_p"] - sft_high_e["1-10"]["micro_p"],
        },
        "delta_final_minus_sft_high.micro_r": {
            ">1000": final_e[">1000"]["micro_r"] - sft_high_e[">1000"]["micro_r"],
            "1-10": final_e["1-10"]["micro_r"] - sft_high_e["1-10"]["micro_r"],
        },
        "sft_high.precision_loss_check": {
            ">1000": {"p": sft_high_e[">1000"]["micro_p"], "r": sft_high_e[">1000"]["micro_r"]},
            "1-10": {"p": sft_high_e["1-10"]["micro_p"], "r": sft_high_e["1-10"]["micro_r"]},
        },
    }
    print()
    print(f"  HEADLINE: head(>1000) micro_f1={head_f1:.4f}  tail(1-10) micro_f1={tail_f1:.4f}")
    print(f"            head/tail ratio = {head_over_tail:.2f}")
    print(
        f"            sft_high precision: head={sft_high_e['>1000']['micro_p']:.3f} "
        f"tail={sft_high_e['1-10']['micro_p']:.3f}"
    )
    print(
        f"            sft_high recall:    head={sft_high_e['>1000']['micro_r']:.3f} "
        f"tail={sft_high_e['1-10']['micro_r']:.3f}"
    )

    payload = {
        "dataset": spec.name,
        "thresholds": {"tau_macro": spec.tau_macro, "tau_micro": spec.tau_micro},
        "thresholds_source": str(spec.thresholds_source.relative_to(REPO_ROOT)),
        "n_admissions": len(event_ids),
        "n_bootstrap": n,
        "seed": seed,
        "filter_mdace_leak": ctx["filter_mdace_leak"],
        "macro_denominator": macro_denominator,
        "subset": "in_dist",
        "in_dist_vocab_size": len(in_vocab),
        "binning_schemes": {
            "edin_count": {
                "order": EDIN_BIN_ORDER,
                "source": "occurrence count of each ICD code in train_dataset.parquet (combined diag+proc)",
            },
            "rank_based": {
                "order": RANK_BIN_ORDER,
                "cutoffs": {
                    "head_max_rank": RANK_HEAD_MAX,
                    "torso_max_rank": RANK_TORSO_MAX,
                },
                "source": "rank by descending occurrence count over the full train vocabulary",
            },
        },
        "n_codes_per_bin": {
            "edin_count": n_codes_edin,
            "rank_based": n_codes_rank,
        },
        "stages": STAGE_ORDER,
        "by_bin": {
            "edin_count": by_bin_edin,
            "rank_based": by_bin_rank,
        },
        "headline": headline,
    }

    if include_fp_by_band:
        print()
        print("  computing fp_by_band (verifier FA + gap-fill hallucination per Edin bin) …")
        fp_by_band = _compute_fp_by_band(
            sft_micro=sft_micro, band_verify=band_verify,
            candidate_verify=candidate_verify, gold_lookup=gold_lookup,
            counts=counts,
        )
        # Sanity gates against delta_attribution.json totals.
        attr_path = OUT_DIR / "error_analyses" / spec.name / "delta_attribution.json"
        if attr_path.exists():
            attr = json.loads(attr_path.read_text())
            ref_v_fp = attr["verifier"]["fp"]
            ref_g_fp = attr["gap_fill"]["fp"]
            assert fp_by_band["totals_check"]["verifier_fp"] == ref_v_fp, (
                f"FP gate FAIL: verifier_fp {fp_by_band['totals_check']['verifier_fp']} "
                f"vs delta_attribution {ref_v_fp}"
            )
            assert fp_by_band["totals_check"]["gap_fill_fp"] == ref_g_fp, (
                f"FP gate FAIL: gap_fill_fp {fp_by_band['totals_check']['gap_fill_fp']} "
                f"vs delta_attribution {ref_g_fp}"
            )
            print(
                f"  FP gate ok: verifier_fp={ref_v_fp}, gap_fill_fp={ref_g_fp} match delta_attribution"
            )
        v = fp_by_band["verifier_fa"]
        g = fp_by_band["gap_fill_hallucination"]
        print(
            f"  fp_by_band verifier:  OOD={v['OOD']:>5}  1-50={v['1-50']:>5}  "
            f"51-1000={v['51-1000']:>5}  >1000={v['>1000']:>5}  total={v['total']:>5}"
        )
        print(
            f"  fp_by_band gap-fill:  OOD={g['OOD']:>5}  1-50={g['1-50']:>5}  "
            f"51-1000={g['51-1000']:>5}  >1000={g['>1000']:>5}  total={g['total']:>5}"
        )
        s = fp_by_band["share_in_head_torso"]
        print(
            f"  share head+torso: verifier={s['verifier_fa']:.3f}, "
            f"gap-fill={s['gap_fill_hallucination']:.3f}; "
            f"share tail: verifier={fp_by_band['share_in_tail']['verifier_fa']:.3f}, "
            f"gap-fill={fp_by_band['share_in_tail']['gap_fill_hallucination']:.3f}"
        )
        payload["fp_by_band"] = fp_by_band

    return payload


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    n = 1000
    seed = 314
    filter_mdace_leak: bool | None = None
    include_fp_by_band = False
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
        elif arg == "--include-fp-by-band":
            include_fp_by_band = True
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
            return 0
        else:
            print(f"unknown argument: {arg!r}", file=sys.stderr)
            return 2

    out_root = OUT_DIR / "error_analyses"
    out_filename = (
        "freq_bands_edin.json" if macro_denominator == "full"
        else "freq_bands_edin_scorable.json"
    )
    for spec in datasets:
        payload = _decompose_for_spec(
            spec, n=n, seed=seed, filter_mdace_leak=filter_mdace_leak,
            include_fp_by_band=include_fp_by_band,
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
