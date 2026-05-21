"""Re-evaluation harness for the ICD-coding paper results.

Composes predicted code sets from saved SFT posteriors and saved Stage-2 LLM
decisions, evaluates against gold labels in INDIST/OOD/FULL scopes, and (via
the `validate` CLI) prints a diff against the saved metrics files and the
paper's published numbers.

Single-dataset: the active pipeline is configured in `params.yaml`
(`dataset_name`, `dataset_path`, `output_dir`, `code_type`). Run from the
repo root:

    python -m coding.eval.harness validate
"""

from __future__ import annotations

import csv
import functools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from coding.eval import paths
from coding.eval.bootstrap import _pr_f1_from_counts
from coding.eval.loaders import (
    build_class_maps_from_parquet as _build_class_maps_from_parquet,
    build_ood_and_full_class_maps as _build_ood_and_full_class_maps,
    load_labels as _load_labels,  # noqa: F401  (re-exported for downstream)
    load_sft_codes as _load_sft_codes,
    load_verified_codes as _load_verified_codes,
)
from coding.pipeline.io import load_predictions
from coding.utils.helpers import (
    cast_id_columns_to_str,
    is_procedure_code,
)

REPO_ROOT = paths.REPO_ROOT
SFT_DIR = paths.DATASET_DIR
MIMIC_III_DIR = paths.OUTPUT_DIR
MIMIC_III_LABELS = paths.LABELS_PATH
OUT_DIR = paths.EVAL_DIR
LOG_PATH = OUT_DIR / "revision_log.md"


# --- Dataset specs --------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Bundle the per-dataset paths, label conventions, and principled thresholds.

    Two specs are pre-built (`MIMIC_III_SPEC`, `MIMIC_IV_SPEC`) and consumed by
    `_load_dataset_context` + `re_eval`. Old MIMIC-III callers (validate,
    bootstrap, pr_decomp) keep using `_load_mimic_iii_context` which is now a
    thin wrapper that loads the OLD thresholds (0.15 / 0.35 / 0.50) for
    byte-identical reproduction of the published draft.
    """

    name: str
    sft_dir: Path
    stage2_dir: Path
    label_path: Path
    code_type: str
    label_config: dict[str, str]
    tau_macro: float
    tau_micro: float
    mdace_filter_applies: bool
    label_code_type_filter: str | None
    thresholds_source: Path


_LABEL_CONFIG_DEFAULT = {
    "diag_column": "diagnoses_code",
    "proc_column": "procedure_code",
}


@functools.cache
def _build_spec() -> DatasetSpec:
    """Construct the active `DatasetSpec`, resolving thresholds on first call.

    Lazy by design: `paths.resolve_thresholds()` triggers the
    `optimal_thresholds.json` read, so importing this module before
    `evaluate_sft` has run is still safe — failure only surfaces here, on the
    first attempt to *use* the spec.
    """
    tau_macro, tau_micro = paths.resolve_thresholds()
    return DatasetSpec(
        name="mimic_iii_clean",
        sft_dir=SFT_DIR,
        stage2_dir=MIMIC_III_DIR,
        label_path=MIMIC_III_LABELS,
        code_type=paths.CODE_TYPE,
        label_config=_LABEL_CONFIG_DEFAULT,
        tau_macro=tau_macro,
        tau_micro=tau_micro,
        mdace_filter_applies=paths.MDACE_LEAK_DIR.exists(),
        label_code_type_filter=None,
        thresholds_source=SFT_DIR / "optimal_thresholds.json",
    )


def __getattr__(name: str):
    """Lazy module attributes: `SPEC`, `MIMIC_III_SPEC`.

    Defers the threshold file read until the spec is actually requested, so
    `from coding.eval.harness import _score_one_scope` (or any other helper)
    succeeds even when `optimal_thresholds.json` has not been produced yet.
    """
    if name in {"SPEC", "MIMIC_III_SPEC"}:
        return _build_spec()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Loaders --------------------------------------------------------------


def load_sft_lookup(
    sft_dir: Path, threshold: float
) -> tuple[dict[str, set[str]], pl.DataFrame]:
    """Return ({event_id: set[code]}, class_map_df) for SFT codes >= threshold."""
    _id_col, lookup, class_map = _load_sft_codes(sft_dir, threshold)
    return lookup, class_map


def load_llm_lookup(jsonl_path: Path) -> dict[str, set[str]]:
    """Return {event_id: set[code]} of correct=True verdicts from a Stage-2 jsonl."""
    _id_col, acc = _load_verified_codes(jsonl_path)
    return acc


def load_llm_decisions_full(jsonl_path: Path) -> dict[str, list[dict]]:
    """Return {event_id: list[verdict]} of ALL verdicts (accepted and rejected).

    Unlike `load_llm_lookup`, this preserves the full verdict record so the
    caller can compute FA/FR rates, oracle compositions, and evidence-grounding
    statistics. Each verdict is a dict with at least `{code, correct}`; band
    verifier verdicts also carry `decision`, `evidence_quote`, `contradiction_quote`.
    """
    acc: dict[str, list[dict]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            content = r["content"]
            if isinstance(content, str):
                content = json.loads(content)
            eid = str(r["event_id"])
            acc.setdefault(eid, []).extend(content.get("verdicts", []))
    return acc


def _write_mdace_filter_manifest(
    dropped: set[str], leak: frozenset[str], original_test_eids: set[str],
) -> None:
    """Emit `outputs/revision/mdace_filter_manifest.txt` with dropped event_ids."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "mdace_filter_manifest.txt"
    sorted_dropped = sorted(dropped, key=lambda s: int(s) if s.isdigit() else s)
    not_in_test = sorted(leak - original_test_eids)
    lines = [
        "# MDACE prompt-tuning leak filter manifest",
        "# Generated by eval.harness._load_mimic_iii_context",
        "",
        f"prompt_tuning_total\t{len(leak)}",
        f"original_test_admissions\t{len(original_test_eids)}",
        f"dropped_from_test\t{len(dropped)}",
        f"prompt_tuning_not_in_test\t{len(not_in_test)}",
        "",
        "# event_ids dropped from MIMIC-III test (one per line):",
        *sorted_dropped,
    ]
    if not_in_test:
        lines += [
            "",
            "# event_ids in prompt-tuning artifacts but not in MIMIC-III test set:",
            *not_in_test,
        ]
    path.write_text("\n".join(lines) + "\n")


@functools.cache
def load_mdace_prompt_tuning_eids() -> frozenset[str]:
    """Return the set of MDACE event_ids used in Stage-2 LLM prompt tuning.

    Recovered as the union of `event_id` keys across all jsonl artifacts under
    ``data/mdace_prompt_tuning/*.jsonl`` (gap-analysis, in-band-verify, top-N
    candidate verify across every backbone). These are the admissions whose
    LLM outputs informed prompt iteration; any of them appearing in the
    MIMIC-III-CLEAN test split represents a tuning leak and must be excluded
    from MIMIC-III headline metrics.

    Returns an empty set (and logs one warning) when the optional
    ``data/mdace_prompt_tuning/`` directory is absent — leak filtering is a
    no-op in that case.
    """
    if not paths.MDACE_LEAK_DIR.exists():
        print(
            f"  NOTE: MDACE prompt-tuning leak filter inactive — "
            f"{paths.MDACE_LEAK_DIR.relative_to(REPO_ROOT)} not present."
        )
        return frozenset()
    eids: set[str] = set()
    for path in paths.MDACE_LEAK_DIR.glob("**/*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eid = r.get("event_id")
                if eid is not None:
                    eids.add(str(eid))
    return frozenset(eids)


def load_gap_candidates(parquet_path: Path) -> dict[str, set[str]]:
    """Return {event_id: set[code]} of all top-N retrieved candidate codes.

    Loads `gap_analysis_pass2_searched.parquet` and flattens the per-term
    nested `diag_codes` / `proc_codes` lists into a single per-admission set.
    These are the raw retrieval candidates BEFORE candidate-verify filtering —
    matches the paper's Table 3 `+gap_fill` row.
    """
    df = pl.read_parquet(parquet_path)
    df = cast_id_columns_to_str(df)
    out: dict[str, set[str]] = {}
    for row in df.iter_rows(named=True):
        eid = str(row["event_id"])
        codes: set[str] = set()
        for sub in (row.get("diag_codes") or []):
            codes.update(sub or [])
        for sub in (row.get("proc_codes") or []):
            codes.update(sub or [])
        out[eid] = out.get(eid, set()) | codes
    return out


# --- Composition ----------------------------------------------------------


def compose_predictions(
    sft_lookup: dict[str, set[str]],
    *llm_lookups: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Union SFT codes with any provided LLM-verified code sets.

    Mirrors the composition in `prompt-icd.evaluate.run_evaluation_combined_ood`:
    keys are the union of all event_ids across inputs; values are set unions.
    """
    all_ids: set[str] = set(sft_lookup)
    for d in llm_lookups:
        all_ids |= set(d)
    out: dict[str, set[str]] = {}
    for eid in all_ids:
        codes = set(sft_lookup.get(eid, set()))
        for d in llm_lookups:
            codes |= d.get(eid, set())
        out[eid] = codes
    return out


# --- Evaluation -----------------------------------------------------------


@dataclass
class ScopeMetrics:
    """Flat container for the six F1 numbers reported per scope."""

    micro: float
    macro: float
    micro_diag: float
    macro_diag: float
    micro_proc: float
    macro_proc: float


@dataclass
class CompositionMetrics:
    in_dist: ScopeMetrics
    ood: ScopeMetrics
    full: ScopeMetrics


def _scope_metrics(combined: dict, diag: dict, proc: dict) -> ScopeMetrics:
    return ScopeMetrics(
        micro=combined["f1_micro"],
        macro=combined["f1_macro"],
        micro_diag=diag["f1_micro"],
        macro_diag=diag["f1_macro"],
        micro_proc=proc["f1_micro"],
        macro_proc=proc["f1_macro"],
    )


def _score_one_scope(
    pred_sets: list[set[str]], gold_sets: list[set[str]], vocab: set[str],
    *,
    macro_denominator: str = "full",
) -> dict[str, float]:
    """Compute micro+macro precision/recall/F1 by direct TP/FP/FN counting.

    Codes outside *vocab* are discarded from both pred and gold (matches the
    `c for c in codes if c in class_map` filtering in prompt-icd's
    `_eval_predictions`). Delegates the actual P/R/F1 reduction to
    `eval.bootstrap._pr_f1_from_counts` so the harness and the bootstrap
    CIs share a single source of truth.

    *macro_denominator*: ``"full"`` averages per-class P/R/F1 over every code in
    *vocab* (legacy behaviour). ``"scorable"`` restricts the macro denominator
    to codes with at least one gold positive within *vocab* (Edin et al. 2023
    convention). Adds ``n_scorable`` and ``n_in_scope`` keys to the returned
    dict so the caller can attach support sizes without re-deriving them.
    """
    if not vocab:
        # torchmetrics' multilabel functions reject num_labels < 2; preserve a
        # well-defined zero-vocab result instead of crashing the harness.
        return {"precision_micro": 0.0, "recall_micro": 0.0, "f1_micro": 0.0,
                "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0,
                "n_scorable": 0, "n_in_scope": 0}
    class_to_idx = {c: i for i, c in enumerate(sorted(vocab))}
    n_classes = len(class_to_idx)
    tp = np.zeros(n_classes, dtype=np.int64)
    fp = np.zeros(n_classes, dtype=np.int64)
    fn = np.zeros(n_classes, dtype=np.int64)
    for p, g in zip(pred_sets, gold_sets, strict=True):
        p_in = p & vocab
        g_in = g & vocab
        for c in p_in & g_in:
            tp[class_to_idx[c]] += 1
        for c in p_in - g_in:
            fp[class_to_idx[c]] += 1
        for c in g_in - p_in:
            fn[class_to_idx[c]] += 1
    p_mi, r_mi, f_mi, p_ma, r_ma, f_ma = _pr_f1_from_counts(
        tp, fp, fn, macro_denominator=macro_denominator,
    )
    n_scorable = int(((tp + fn) > 0).sum())
    return {
        "precision_micro": p_mi, "recall_micro": r_mi, "f1_micro": f_mi,
        "precision_macro": p_ma, "recall_macro": r_ma, "f1_macro": f_ma,
        "n_scorable": n_scorable, "n_in_scope": n_classes,
    }


def evaluate_composition(
    pred_lookup: dict[str, set[str]],
    labels: pl.DataFrame,
    label_config: dict,
    in_class_maps: tuple[dict, dict, dict],
    code_type: str,
    *,
    label_id_col: str = "event_id",
    ood_class_maps: tuple[dict, dict, dict] | None = None,
    full_class_maps: tuple[dict, dict, dict] | None = None,
) -> CompositionMetrics:
    """Evaluate a predicted-code lookup against gold labels in INDIST/OOD/FULL scopes.

    *ood_class_maps* and *full_class_maps* may be passed in to ensure the macro-F1
    denominator matches the originally-published vocabulary. If omitted, both are
    derived from *labels* — pass the GLOBAL labels-derived class maps when
    *labels* itself is restricted to a test split, or you will under-count the
    OOD/full vocabularies and inflate macro F1.
    """
    in_combined_map, in_diag_map, in_proc_map = in_class_maps
    if ood_class_maps is None or full_class_maps is None:
        ood_class_maps, full_class_maps = _build_ood_and_full_class_maps(
            labels, label_config, in_class_maps
        )
    ood_combined, ood_diag_map, ood_proc_map = ood_class_maps
    full_combined, full_diag_map, full_proc_map = full_class_maps

    in_combined_vocab = set(in_combined_map)
    in_diag_vocab = set(in_diag_map)
    in_proc_vocab = set(in_proc_map)
    ood_combined_vocab = set(ood_combined)
    ood_diag_vocab = set(ood_diag_map)
    ood_proc_vocab = set(ood_proc_map)
    full_combined_vocab = set(full_combined)
    full_diag_vocab = set(full_diag_map)
    full_proc_vocab = set(full_proc_map)

    # Build per-admission predicted code sets, split by code type.
    label_ids = labels[label_id_col].to_list()
    diag_codes_col = labels[label_config["diag_column"]].to_list()
    proc_codes_col = labels[label_config["proc_column"]].to_list()

    pred_combined: list[set[str]] = []
    pred_diag: list[set[str]] = []
    pred_proc: list[set[str]] = []
    gold_combined: list[set[str]] = []
    gold_diag: list[set[str]] = []
    gold_proc: list[set[str]] = []

    for eid, gd, gp in zip(label_ids, diag_codes_col, proc_codes_col, strict=True):
        codes = pred_lookup.get(str(eid), set())
        diag = {c for c in codes if not is_procedure_code(c, code_type)}
        proc = {c for c in codes if is_procedure_code(c, code_type)}
        pred_combined.append(codes)
        pred_diag.append(diag)
        pred_proc.append(proc)
        gd_set = set(gd) if gd is not None else set()
        gp_set = set(gp) if gp is not None else set()
        gold_diag.append(gd_set)
        gold_proc.append(gp_set)
        gold_combined.append(gd_set | gp_set)

    def _scope(combined_v, diag_v, proc_v) -> ScopeMetrics:
        c = _score_one_scope(pred_combined, gold_combined, combined_v)
        d = _score_one_scope(pred_diag, gold_diag, diag_v)
        p = _score_one_scope(pred_proc, gold_proc, proc_v)
        return _scope_metrics(c, d, p)

    return CompositionMetrics(
        in_dist=_scope(in_combined_vocab, in_diag_vocab, in_proc_vocab),
        ood=_scope(ood_combined_vocab, ood_diag_vocab, ood_proc_vocab),
        full=_scope(full_combined_vocab, full_diag_vocab, full_proc_vocab),
    )


# --- Validation gate ------------------------------------------------------


# Paper Table 2 (MIMIC-III) micro/macro F1 cells we need to reproduce.
# Columns: in_dist_micro, in_dist_macro, ood_micro, ood_macro, full_micro, full_macro.
# A `None` cell means "by construction ---" (SFT cannot reach OOD).
_PAPER_TABLE_2 = {
    "sft_only_tau_0.15": (0.575, 0.333, None, None, 0.570, 0.137),
    "sft_only_tau_0.35": (0.612, 0.283, None, None, 0.606, 0.117),
    "sft_only_tau_0.50": (0.585, 0.229, None, None, 0.579, 0.095),
    "hybrid":            (0.554, 0.325, 0.087, 0.042, 0.542, 0.159),
}

# Paper Table 3 cumulative-stage micro/macro F1 cells (MIMIC-III FULL).
# Columns: full_micro, full_macro.
_PAPER_TABLE_3 = {
    "sft_high":          (0.579, 0.095),
    "+band_verify":      (0.592, 0.103),
    "+gap_fill":         (0.413, 0.142),
    "+verify_gap_fill":  (0.542, 0.159),
}


def _saved_metrics_to_table(saved: dict) -> tuple:
    """Extract (in_dist, ood, full) micro/macro from a saved metrics_*.json blob."""
    return (
        saved["in_dist"]["f1_micro"], saved["in_dist"]["f1_macro"],
        saved["ood"]["f1_micro"], saved["ood"]["f1_macro"],
        saved["full"]["f1_micro"], saved["full"]["f1_macro"],
    )


def _diff(
    label: str, harness: tuple, paper: tuple, saved: tuple | None,
    *, paper_gate_relaxed: bool = False,
) -> list[dict]:
    rows = []
    cols = ["in_dist_micro", "in_dist_macro",
            "ood_micro", "ood_macro",
            "full_micro", "full_macro"]
    paper_pass = "INFO_FILTERED" if paper_gate_relaxed else "PASS"
    paper_fail = "INFO_FILTERED" if paper_gate_relaxed else "FAIL"
    for i, col in enumerate(cols):
        h = harness[i]
        p = paper[i]
        s = saved[i] if saved is not None else None
        delta_paper = (h - p) if (h is not None and p is not None) else None
        delta_saved = (h - s) if (h is not None and s is not None) else None
        rows.append({
            "config": label, "metric": col,
            "harness": "" if h is None else f"{h:.4f}",
            "paper": "" if p is None else f"{p:.3f}",
            "saved": "" if s is None else f"{s:.4f}",
            "delta_vs_paper": "" if delta_paper is None else f"{delta_paper:+.4f}",
            "delta_vs_saved": "" if delta_saved is None else f"{delta_saved:+.4f}",
            "gate_pass_paper": (
                "" if delta_paper is None
                else paper_pass if abs(delta_paper) <= 0.001 else paper_fail
            ),
            "gate_pass_saved": (
                "" if delta_saved is None
                else "PASS" if abs(delta_saved) <= 0.001 else "FAIL"
            ),
        })
    return rows


def _comp_to_t2_tuple(c: CompositionMetrics) -> tuple:
    return (
        c.in_dist.micro, c.in_dist.macro,
        c.ood.micro, c.ood.macro,
        c.full.micro, c.full.macro,
    )


def _comp_to_t3_tuple(c: CompositionMetrics) -> tuple:
    """Table 3 only reports FULL micro / macro."""
    return (None, None, None, None, c.full.micro, c.full.macro)


def validate_mimic_iii(out_csv: Path, *, filter_mdace_leak: bool = False) -> dict:
    """Reproduce Table 2 (4 rows) and Table 3 (4 rows) from saved artifacts.

    With *filter_mdace_leak* false (default) this reproduces the published
    paper numbers; with it true the MDACE prompt-tuning admissions are dropped
    from the MIMIC-III test set first and the paper-comparison gate is
    relaxed (delta > 0.001 is logged as ``INFO_FILTERED`` rather than
    ``FAIL``, since divergence from the published numbers is expected and
    intended after filtering).

    Returns a summary dict; writes the full diff to *out_csv*.
    """
    label_suffix = "[filtered]" if filter_mdace_leak else ""
    print(f"=== validate_mimic_iii {label_suffix} ===")

    sft_dir = SFT_DIR
    mimic_dir = MIMIC_III_DIR
    code_type = "ICD-9"
    label_config = {"diag_column": "diagnoses_code", "proc_column": "procedure_code"}

    print("Loading SFT predictions and class map …")
    sft_015, class_map_df = load_sft_lookup(sft_dir, threshold=0.15)
    sft_035, _ = load_sft_lookup(sft_dir, threshold=0.35)
    sft_050, _ = load_sft_lookup(sft_dir, threshold=0.50)

    print("Loading Stage-2 LLM decisions …")
    band_verify = load_llm_lookup(mimic_dir / "in_band_verification.jsonl")
    candidate_verify = load_llm_lookup(mimic_dir / "searched_topn_verification.jsonl")

    print("Loading gold labels (full unfiltered MIMIC-III) and filtering to test split …")
    # The SFT-side test_dataset.parquet has labels filtered to in-distribution
    # codes only (matches class_map exactly), so it contains zero OOD codes.
    # Use the unfiltered processed_mimic_iii_full.parquet from prompt-icd/data/.
    #
    # IMPORTANT: OOD/Full class maps must be derived from the WHOLE label
    # parquet (matching the original eval, where the saved metrics file records
    # vocab_sizes.full.combined=8929). Restricting OOD derivation to the test
    # split alone underestimates the macro-F1 denominator and inflates the
    # metric by a factor of ~1.6×.
    test_event_ids_int = (
        {int(eid) for eid in sft_015.keys()}
        | {int(eid) for eid in sft_035.keys()}
        | {int(eid) for eid in sft_050.keys()}
    )
    full_labels = pl.read_parquet(MIMIC_III_LABELS)
    full_labels = cast_id_columns_to_str(full_labels).with_columns(
        pl.concat_list([
            pl.col(label_config["diag_column"]).fill_null([]),
            pl.col(label_config["proc_column"]).fill_null([]),
        ]).alias("target")
    )
    in_class_maps = _build_class_maps_from_parquet(class_map_df, code_type=code_type)
    # OOD/Full derived from ALL labels (matches original eval).
    ood_class_maps_global, full_class_maps_global = _build_ood_and_full_class_maps(
        full_labels, label_config, in_class_maps
    )
    print(
        f"  vocab sizes — in_dist: {len(in_class_maps[0])}, "
        f"ood: {len(ood_class_maps_global[0])}, full: {len(full_class_maps_global[0])}"
    )

    # For per-admission scoring, restrict to test event_ids.
    if "split" in full_labels.columns:
        labels = full_labels.filter(pl.col("split") == "test")
    else:
        labels = full_labels.filter(
            pl.col("event_id").cast(pl.Int64).is_in(list(test_event_ids_int))
        )
    print(f"  test labels: {labels.height} rows (vs {len(test_event_ids_int)} SFT-predicted ids)")

    if filter_mdace_leak:
        leak = load_mdace_prompt_tuning_eids()
        original_test_eids = set(labels["event_id"].to_list())
        dropped = {e for e in leak if e in original_test_eids}
        print(
            f"  MDACE leak filter: {len(leak)} prompt-tuning ids; "
            f"{len(dropped)} dropped from MIMIC-III test"
        )
        if dropped:
            keep_ids = list(original_test_eids - dropped)
            labels = labels.filter(pl.col("event_id").is_in(keep_ids))
            for d in (sft_015, sft_035, sft_050, band_verify, candidate_verify):
                for eid in list(d.keys()):
                    if eid in dropped:
                        del d[eid]
            _write_mdace_filter_manifest(dropped, leak, original_test_eids)

    rows: list[dict] = []

    # --- Table 2 rows ---
    configs = [
        ("sft_only_tau_0.15", compose_predictions(sft_015)),
        ("sft_only_tau_0.35", compose_predictions(sft_035)),
        ("sft_only_tau_0.50", compose_predictions(sft_050)),
        ("hybrid", compose_predictions(sft_050, band_verify, candidate_verify)),
    ]
    table2_metrics: dict[str, CompositionMetrics] = {}
    for label, preds in configs:
        print(f"\n[Table 2] evaluating {label} …")
        m = evaluate_composition(
            preds, labels, label_config, in_class_maps, code_type=code_type,
            ood_class_maps=ood_class_maps_global,
            full_class_maps=full_class_maps_global,
        )
        table2_metrics[label] = m
        harness_t = _comp_to_t2_tuple(m)
        paper_t = _PAPER_TABLE_2[label]
        saved_t = None
        if label == "hybrid" and not filter_mdace_leak:
            saved = json.loads((mimic_dir / "metrics_verified_combined_ood.json").read_text())
            saved_t = _saved_metrics_to_table(saved)
        rows.extend(_diff(
            f"table2:{label}", harness_t, paper_t, saved_t,
            paper_gate_relaxed=filter_mdace_leak,
        ))

    # --- Table 3 stage-wise rows ---
    print("\n[Table 3] evaluating stage-wise compositions …")
    stage_configs = [
        ("sft_high", compose_predictions(sft_050)),
        ("+band_verify", compose_predictions(sft_050, band_verify)),
        # `+gap_fill` row is the SFT-high + band-verify + UNVERIFIED top-N gap candidates.
        # Saved artifacts only retain the verified candidate-verify decisions; the raw
        # top-N candidate set is reconstructible from gap_analysis_pass2_searched.parquet
        # but is not currently loaded — flagged below in the diff so the harness clearly
        # reports it as "deferred" rather than silently mis-reproducing.
        ("+verify_gap_fill",
         compose_predictions(sft_050, band_verify, candidate_verify)),
    ]
    for label, preds in stage_configs:
        m = evaluate_composition(
            preds, labels, label_config, in_class_maps, code_type=code_type,
            ood_class_maps=ood_class_maps_global,
            full_class_maps=full_class_maps_global,
        )
        harness_t = _comp_to_t3_tuple(m)
        paper_t = (None, None, None, None, *_PAPER_TABLE_3[label])
        rows.extend(_diff(
            f"table3:{label}", harness_t, paper_t, None,
            paper_gate_relaxed=filter_mdace_leak,
        ))

    # +gap_fill row is intentionally deferred (no raw top-N artifact loaded yet).
    rows.append({
        "config": "table3:+gap_fill",
        "metric": "deferred",
        "harness": "",
        "paper": f"{_PAPER_TABLE_3['+gap_fill'][0]:.3f}/{_PAPER_TABLE_3['+gap_fill'][1]:.3f}",
        "saved": "",
        "delta_vs_paper": "",
        "delta_vs_saved": "",
        "gate_pass_paper": "DEFERRED",
        "gate_pass_saved": "",
    })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config", "metric", "harness", "paper", "saved",
                  "delta_vs_paper", "delta_vs_saved",
                  "gate_pass_paper", "gate_pass_saved"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    # Summary verdict
    fail_paper = sum(1 for r in rows if r["gate_pass_paper"] == "FAIL")
    pass_paper = sum(1 for r in rows if r["gate_pass_paper"] == "PASS")
    info_filtered = sum(1 for r in rows if r["gate_pass_paper"] == "INFO_FILTERED")
    fail_saved = sum(1 for r in rows if r["gate_pass_saved"] == "FAIL")
    pass_saved = sum(1 for r in rows if r["gate_pass_saved"] == "PASS")
    return {
        "rows": rows,
        "pass_paper": pass_paper, "fail_paper": fail_paper,
        "info_filtered": info_filtered,
        "pass_saved": pass_saved, "fail_saved": fail_saved,
        "filtered": filter_mdace_leak,
    }


def _append_validation_log(summary: dict, csv_path: Path) -> None:
    """Append a Phase 0.2 section to revision_log.md (idempotent on the marker)."""
    marker = (
        "## Phase 1.3 — Harness validation [MDACE-leak filtered] (auto-generated)"
        if summary.get("filtered")
        else "## Phase 0.2 — Harness validation gate (auto-generated)"
    )
    lines = ["", marker, ""]
    lines.append(f"- Diff CSV: `{csv_path.relative_to(REPO_ROOT)}`")
    if summary.get("filtered"):
        lines.append(
            "- MDACE prompt-tuning admissions filtered out of the MIMIC-III "
            "test set; deltas vs the published-paper numbers are EXPECTED "
            "(reported as INFO_FILTERED rather than FAIL)."
        )
        total = summary["pass_paper"] + summary["fail_paper"] + summary["info_filtered"]
        lines.append(f"- Cells reproduced: {total} (vs paper)")
        lines.append(
            f"- INFO_FILTERED vs paper: **{summary['info_filtered']}**, "
            f"unchanged-from-paper PASS: {summary['pass_paper']}"
        )
    else:
        lines.append(f"- Cells reproduced: {summary['pass_paper'] + summary['fail_paper']} (vs paper)")
        lines.append(f"- PASS (≤0.001 F1) vs paper: **{summary['pass_paper']}**, FAIL: **{summary['fail_paper']}**")
        lines.append(f"- PASS vs saved metrics_*.json: {summary['pass_saved']}, FAIL: {summary['fail_saved']}")
    lines.append("")
    lines.append("| config | metric | harness | paper | Δ vs paper | gate |")
    lines.append("|---|---|---|---|---|---|")
    for r in summary["rows"]:
        if r["metric"] == "deferred":
            lines.append(f"| {r['config']} | deferred | — | {r['paper']} | — | DEFERRED |")
            continue
        if not r["harness"]:
            continue
        lines.append(
            f"| {r['config']} | {r['metric']} | {r['harness']} | {r['paper']} "
            f"| {r['delta_vs_paper']} | {r['gate_pass_paper']} |"
        )
    lines.append("")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        existing = LOG_PATH.read_text()
        if marker in existing:
            existing = existing.split(marker)[0].rstrip() + "\n"
        LOG_PATH.write_text(existing + "\n".join(lines))
    else:
        LOG_PATH.write_text("# Revision log\n" + "\n".join(lines))


def validate(out_dir: Path = OUT_DIR, *, filter_mdace_leak: bool = False) -> int:
    out_csv = out_dir / (
        "validation_diff_filtered.csv" if filter_mdace_leak else "validation_diff.csv"
    )
    summary = validate_mimic_iii(out_csv, filter_mdace_leak=filter_mdace_leak)
    _append_validation_log(summary, out_csv)
    print("\n=== Gate verdict ===")
    if filter_mdace_leak:
        print(f"  INFO_FILTERED vs paper:  {summary['info_filtered']}")
        print(f"  Unchanged from paper:    {summary['pass_paper']}")
    else:
        print(f"  PASS vs paper:  {summary['pass_paper']}")
        print(f"  FAIL vs paper:  {summary['fail_paper']}")
        print(f"  PASS vs saved:  {summary['pass_saved']}")
        print(f"  FAIL vs saved:  {summary['fail_saved']}")
        if summary["fail_paper"] > 0:
            print("  → Validation gate has UNRESOLVED diffs vs paper. See diff CSV and revision_log.md.")
    return 0 if summary["fail_paper"] == 0 else 2


def _load_dataset_context(
    spec: DatasetSpec,
    *,
    sft_thresholds: list[float],
    filter_mdace_leak: bool | None = None,
) -> dict:
    """Generic context loader keyed by `DatasetSpec`.

    Loads SFT predictions at each requested threshold, Stage-2 verifier and
    candidate-verifier lookups, gold labels (deduped, optionally code-type
    filtered), and class maps. `_load_mimic_iii_context` is now a thin wrapper
    that calls this with the OLD MIMIC-III draft thresholds for byte-identical
    reproduction of the published numbers.

    Returned dict keys:
      - spec — the input DatasetSpec
      - sft_lookups — dict[float, dict[event_id, set[code]]] keyed by threshold
      - band_verify, candidate_verify — dict[event_id, set[code]] of correct-True verdicts
      - labels — pl.DataFrame restricted to test admissions, deduped on event_id
      - in_class_maps — (combined, diag, proc) for the SFT in-distribution vocab
      - ood_class_maps_global, full_class_maps_global — derived from labels
      - event_ids — ordered list[str] of test admissions
      - gold_lookup — dict[event_id, set[code]] of gold codes (combined)
      - code_type, label_config — copied from spec
      - filter_mdace_leak, mdace_leak_dropped — book-keeping for the filter
    """
    if filter_mdace_leak is None:
        filter_mdace_leak = spec.mdace_filter_applies
    elif filter_mdace_leak and not spec.mdace_filter_applies:
        print(
            f"  WARN: MDACE-leak filter requested for {spec.name}, but "
            f"spec.mdace_filter_applies=False. Honouring the override."
        )

    print(
        f"Loading SFT predictions for {spec.name} at thresholds "
        f"{[round(t, 4) for t in sft_thresholds]} …"
    )
    sft_lookups: dict[float, dict[str, set[str]]] = {}
    class_map_df: pl.DataFrame | None = None
    for tau in sft_thresholds:
        lookup, class_map_df = load_sft_lookup(spec.sft_dir, threshold=tau)
        sft_lookups[tau] = lookup
    assert class_map_df is not None, "sft_thresholds must be non-empty"

    print(f"Loading Stage-2 LLM decisions from {spec.stage2_dir.name} …")
    band_verify = load_llm_lookup(spec.stage2_dir / "in_band_verification.jsonl")
    candidate_verify = load_llm_lookup(
        spec.stage2_dir / "searched_topn_verification.jsonl"
    )

    print(f"Loading labels from {spec.label_path.name} …")
    full_labels = pl.read_parquet(spec.label_path)
    full_labels = cast_id_columns_to_str(full_labels).with_columns(
        pl.concat_list([
            pl.col(spec.label_config["diag_column"]).fill_null([]),
            pl.col(spec.label_config["proc_column"]).fill_null([]),
        ]).alias("target")
    )

    if spec.label_code_type_filter is not None:
        n_before = full_labels.height
        full_labels = full_labels.filter(
            pl.col("diagnoses_code_type") == spec.label_code_type_filter
        )
        print(
            f"  code_type filter: kept {full_labels.height} / {n_before} rows "
            f"(diagnoses_code_type == {spec.label_code_type_filter!r})"
        )

    in_class_maps = _build_class_maps_from_parquet(class_map_df, code_type=spec.code_type)
    ood_class_maps_global, full_class_maps_global = _build_ood_and_full_class_maps(
        full_labels, spec.label_config, in_class_maps
    )

    test_event_ids_str: set[str] = set()
    for lookup in sft_lookups.values():
        test_event_ids_str |= set(lookup)
    if "split" in full_labels.columns:
        labels_test = full_labels.filter(pl.col("split") == "test")
    else:
        labels_test = full_labels.filter(
            pl.col("event_id").is_in(list(test_event_ids_str))
        )

    n_before_dedup = labels_test.height
    labels_test = labels_test.unique(subset=["event_id"], keep="first")
    if labels_test.height != n_before_dedup:
        print(
            f"  dedup: kept {labels_test.height} / {n_before_dedup} rows "
            f"after unique(event_id, keep='first')"
        )

    mdace_leak_dropped: set[str] = set()
    if filter_mdace_leak:
        leak = load_mdace_prompt_tuning_eids()
        original_test_eids = set(labels_test["event_id"].to_list())
        mdace_leak_dropped = {e for e in leak if e in original_test_eids}
        print(
            f"  MDACE prompt-tuning leak filter: {len(leak)} prompt-tuning ids, "
            f"{len(mdace_leak_dropped)} present in {spec.name} test (dropped)"
        )
        if mdace_leak_dropped:
            keep_ids = list(original_test_eids - mdace_leak_dropped)
            labels_test = labels_test.filter(pl.col("event_id").is_in(keep_ids))
            for d in [*sft_lookups.values(), band_verify, candidate_verify]:
                for eid in list(d.keys()):
                    if eid in mdace_leak_dropped:
                        del d[eid]
            _write_mdace_filter_manifest(
                mdace_leak_dropped, leak, original_test_eids
            )

    event_ids = labels_test["event_id"].to_list()
    gold_combined: list[set[str]] = []
    diag_col = labels_test[spec.label_config["diag_column"]].to_list()
    proc_col = labels_test[spec.label_config["proc_column"]].to_list()
    for d, p in zip(diag_col, proc_col, strict=True):
        gold_combined.append((set(d) if d else set()) | (set(p) if p else set()))
    gold_lookup = {
        eid: codes for eid, codes in zip(event_ids, gold_combined, strict=True)
    }
    print(f"  context loaded: {len(event_ids)} test admissions for {spec.name}")

    return {
        "spec": spec,
        "sft_lookups": sft_lookups,
        "band_verify": band_verify,
        "candidate_verify": candidate_verify,
        "labels": labels_test,
        "in_class_maps": in_class_maps,
        "ood_class_maps_global": ood_class_maps_global,
        "full_class_maps_global": full_class_maps_global,
        "event_ids": event_ids,
        "gold_lookup": gold_lookup,
        "code_type": spec.code_type,
        "label_config": spec.label_config,
        "filter_mdace_leak": filter_mdace_leak,
        "mdace_leak_dropped": mdace_leak_dropped,
    }


def _load_mimic_iii_context(*, filter_mdace_leak: bool = True) -> dict:
    """Back-compat wrapper for legacy callers (validate / bootstrap / pr_decomp).

    Loads SFT at both the legacy thresholds (0.15 / 0.35 / 0.50) AND the
    principled ones (τ*_macro=0.125 / τ*_micro=0.311), aliases all five under
    historical keys (`sft_015`, `sft_035`, `sft_050`, `sft_macro`, `sft_micro`)
    so the legacy rows in `pr_decomp.csv` reproduce byte-identically *and* the
    new-band rows are available via the same context object. New callers
    (re_eval) should use `_load_dataset_context` directly.
    """
    ctx = _load_dataset_context(
        MIMIC_III_SPEC,
        sft_thresholds=[
            0.15, 0.35, 0.50,
            MIMIC_III_SPEC.tau_macro, MIMIC_III_SPEC.tau_micro,
        ],
        filter_mdace_leak=filter_mdace_leak,
    )
    sft_lookups = ctx["sft_lookups"]
    ctx["sft_015"] = sft_lookups[0.15]
    ctx["sft_035"] = sft_lookups[0.35]
    ctx["sft_050"] = sft_lookups[0.50]
    ctx["sft_macro"] = sft_lookups[MIMIC_III_SPEC.tau_macro]
    ctx["sft_micro"] = sft_lookups[MIMIC_III_SPEC.tau_micro]
    return ctx


def load_sft_scores(
    sft_dir: Path,
) -> tuple[list[str], "np.ndarray", list[str | None]]:
    """Return ``(event_ids, scores, codes_by_index)`` for the raw SFT logits.

    `scores` has shape ``(n_admissions, n_classes)`` where ``n_classes`` matches
    `class_map.parquet`'s `value` column (the InDist vocabulary). `codes_by_index[j]`
    is the ICD code mapped to column ``j``. Used to compute threshold-free
    ranking metrics (P@k, R@k, R-precision) from saved artifacts without
    re-running inference.
    """
    import numpy as np  # local: keep top-of-file import set tight.

    df = load_predictions(sft_dir)
    if df.height == 1:
        df = df.explode(pl.all())
    if "ids" in df.columns:
        df = df.rename({"ids": "event_id"})
    df = cast_id_columns_to_str(df)
    event_ids = df["event_id"].to_list()
    scores = np.asarray(df["predictions"].to_list(), dtype=np.float32)
    cm = pl.read_parquet(sft_dir / "class_map.parquet")
    n_classes = int(cm["value"].max() or 0) + 1
    codes_by_idx: list[str | None] = [None] * n_classes
    for k, v in zip(cm["key"].to_list(), cm["value"].to_list(), strict=True):
        codes_by_idx[int(v)] = k
    return event_ids, scores, codes_by_idx


def _build_decomp_table(
    pred_lookup: dict[str, set[str]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    vocab: set[str],
    *,
    sft_event_ids: list[str] | None = None,
    sft_scores: "np.ndarray | None" = None,
    codes_by_idx: list[str | None] | None = None,
):
    """Build a `DecompTable` for one (system, scope) cell.

    When all three of *sft_event_ids*, *sft_scores*, *codes_by_idx* are
    provided, top-k arrays (`correct_at_8`, `correct_at_15`, `correct_at_R`)
    are computed from the SFT logits. Otherwise they are left at None and
    the bootstrap routine skips ranking metrics.
    """
    import numpy as np
    from coding.eval.bootstrap import DecompTable, build_index_table

    base = build_index_table(pred_lookup, gold_lookup, event_ids, vocab)
    n = len(event_ids)
    n_pred = np.zeros(n, dtype=np.int64)
    n_gold = np.zeros(n, dtype=np.int64)
    exact_match = np.zeros(n, dtype=bool)
    for i, eid in enumerate(event_ids):
        p_in = pred_lookup.get(str(eid), set()) & vocab
        g_in = gold_lookup.get(str(eid), set()) & vocab
        n_pred[i] = len(p_in)
        n_gold[i] = len(g_in)
        exact_match[i] = (p_in == g_in)

    correct_at_8 = correct_at_15 = correct_at_R = None
    if (
        sft_event_ids is not None
        and sft_scores is not None
        and codes_by_idx is not None
    ):
        correct_at_8 = np.zeros(n, dtype=np.int64)
        correct_at_15 = np.zeros(n, dtype=np.int64)
        correct_at_R = np.zeros(n, dtype=np.int64)
        sft_id_to_idx = {eid: i for i, eid in enumerate(sft_event_ids)}
        n_classes_sft = sft_scores.shape[1]
        # Pre-build bool mask of which SFT class indices fall in *vocab*; saves
        # n*K Python comparisons.
        in_vocab_mask = np.array(
            [(c is not None and c in vocab) for c in codes_by_idx], dtype=bool,
        )
        for i, eid in enumerate(event_ids):
            sft_idx = sft_id_to_idx.get(str(eid))
            if sft_idx is None:
                continue
            row = sft_scores[sft_idx]
            g_in = gold_lookup.get(str(eid), set()) & vocab
            R = int(n_gold[i])
            # Top-k indices of raw logits (descending). Top-k can include codes
            # outside *vocab* — we filter at the membership-check stage.
            for k, dst in ((8, correct_at_8), (15, correct_at_15)):
                kk = min(k, n_classes_sft)
                if kk <= 0:
                    continue
                topk = np.argpartition(-row, kk - 1)[:kk]
                # Codes for these indices that are in vocab AND in gold:
                hit = 0
                for j in topk:
                    code = codes_by_idx[j]
                    if code is not None and code in g_in:
                        hit += 1
                dst[i] = hit
            if R > 0 and R <= n_classes_sft:
                topR = np.argpartition(-row, R - 1)[:R]
                hit = 0
                for j in topR:
                    code = codes_by_idx[j]
                    if code is not None and code in g_in:
                        hit += 1
                correct_at_R[i] = hit

    return DecompTable(
        n_admissions=base.n_admissions, n_classes=base.n_classes,
        tp_idx=base.tp_idx, fp_idx=base.fp_idx, fn_idx=base.fn_idx,
        n_pred=n_pred, n_gold=n_gold, exact_match=exact_match,
        correct_at_8=correct_at_8, correct_at_15=correct_at_15,
        correct_at_R=correct_at_R,
    )


def _system_lookups(ctx: dict) -> dict[str, dict[str, set[str]]]:
    """Map system label → admission-level predicted code lookup.

    Includes both legacy [0.15, 0.35, 0.50] and principled [τ*_macro, τ*_micro]
    operating points so callers can switch between bands without re-loading.
    """
    out = {
        "sft_only_tau_0.15": compose_predictions(ctx["sft_015"]),
        "sft_only_tau_0.35": compose_predictions(ctx["sft_035"]),
        "sft_only_tau_0.50": compose_predictions(ctx["sft_050"]),
        "hybrid": compose_predictions(ctx["sft_050"], ctx["band_verify"], ctx["candidate_verify"]),
        # Legacy Table 3 stage rows; sft_high == sft_only_tau_0.50 by definition.
        "sft_high": compose_predictions(ctx["sft_050"]),
        "+band_verify": compose_predictions(ctx["sft_050"], ctx["band_verify"]),
        "+verify_gap_fill": compose_predictions(
            ctx["sft_050"], ctx["band_verify"], ctx["candidate_verify"]
        ),
    }
    if "sft_macro" in ctx and "sft_micro" in ctx:
        # Principled-band rows: sft_only at τ*_macro / τ*_micro and the hybrid
        # composed off τ*_micro (== θ_high under the principled framing).
        out["sft_only_tau_macro"] = compose_predictions(ctx["sft_macro"])
        out["sft_only_tau_micro"] = compose_predictions(ctx["sft_micro"])
        out["hybrid_principled"] = compose_predictions(
            ctx["sft_micro"], ctx["band_verify"], ctx["candidate_verify"],
        )
        out["sft_high_principled"] = compose_predictions(ctx["sft_micro"])
        out["+band_verify_principled"] = compose_predictions(
            ctx["sft_micro"], ctx["band_verify"],
        )
        out["+verify_gap_fill_principled"] = compose_predictions(
            ctx["sft_micro"], ctx["band_verify"], ctx["candidate_verify"],
        )
    return out


# Cells to bootstrap: (system_label, scope, vocab_source).
def _bootstrap_targets(ctx: dict) -> list[tuple[str, str, set[str]]]:
    in_v = set(ctx["in_class_maps"][0])
    ood_v = set(ctx["ood_class_maps_global"][0])
    full_v = set(ctx["full_class_maps_global"][0])
    out: list[tuple[str, str, set[str]]] = []
    for system in ["sft_only_tau_0.15", "sft_only_tau_0.35", "sft_only_tau_0.50", "hybrid"]:
        out.append((system, "in_dist", in_v))
        out.append((system, "ood", ood_v))
        out.append((system, "full", full_v))
    # Table 3 cells (FULL only).
    for stage in ["sft_high", "+band_verify", "+verify_gap_fill"]:
        out.append((stage, "full", full_v))
    return out


# Headline paired comparisons: (A_system, B_system, scope).
_PAIRED_TESTS: list[tuple[str, str, str]] = [
    ("hybrid", "sft_only_tau_0.15", "full"),  # Hybrid vs macro-optimal (headline)
    ("hybrid", "sft_only_tau_0.35", "full"),  # Hybrid vs micro-optimal (cost)
    ("hybrid", "sft_only_tau_0.50", "full"),  # Hybrid vs matched threshold
    ("+band_verify", "sft_high", "full"),     # Table 3 stage delta
    ("+verify_gap_fill", "+band_verify", "full"),  # Table 3 stage delta
]


def bootstrap(
    out_dir: Path = OUT_DIR, *, n: int = 1000, seed: int = 314,
    filter_mdace_leak: bool = True,
) -> int:
    """Compute CIs and paired tests for all Table 2/3 cells; write CSV + log."""
    from coding.eval.bootstrap import (
        bootstrap_f1_from_table,
        build_index_table,
        paired_bootstrap_test_from_tables,
    )

    ctx = _load_mimic_iii_context(filter_mdace_leak=filter_mdace_leak)
    systems = _system_lookups(ctx)

    # Pre-build per (system, scope) IndexTables so paired tests can reuse them.
    print("Building index tables …")
    tables: dict[tuple[str, str], object] = {}
    for system, scope, vocab in _bootstrap_targets(ctx):
        if (system, scope) in tables:
            continue
        tables[(system, scope)] = build_index_table(
            systems[system], ctx["gold_lookup"], ctx["event_ids"], vocab
        )

    # CIs per cell.
    rows: list[dict] = []
    print(f"Bootstrapping CIs (n={n}, seed={seed}) …")
    for system, scope, _vocab in _bootstrap_targets(ctx):
        table = tables[(system, scope)]
        res = bootstrap_f1_from_table(table, n=n, seed=seed, label=f"{system}/{scope}")
        rows.append({
            "kind": "ci", "config": system, "scope": scope, "metric": "micro",
            "point": f"{res.point_micro:.4f}",
            "boot_mean": f"{res.micro_boot_mean:.4f}",
            "normal_ci_low": f"{res.micro_normal_ci_low:.4f}",
            "normal_ci_high": f"{res.micro_normal_ci_high:.4f}",
            "pct_ci_low": f"{res.micro_ci_low:.4f}",
            "pct_ci_high": f"{res.micro_ci_high:.4f}",
            "basic_ci_low": f"{res.micro_basic_ci_low:.4f}",
            "basic_ci_high": f"{res.micro_basic_ci_high:.4f}",
            "bca_ci_low": f"{res.micro_bca_ci_low:.4f}",
            "bca_ci_high": f"{res.micro_bca_ci_high:.4f}",
            "delta": "", "delta_normal_ci_low": "", "delta_normal_ci_high": "",
            "delta_pct_ci_low": "", "delta_pct_ci_high": "",
            "delta_basic_ci_low": "", "delta_basic_ci_high": "",
            "delta_bca_ci_low": "", "delta_bca_ci_high": "",
            "delta_boot_mean": "", "p_two_sided": "",
            "n": n, "seed": seed,
        })
        rows.append({
            "kind": "ci", "config": system, "scope": scope, "metric": "macro",
            "point": f"{res.point_macro:.4f}",
            "boot_mean": f"{res.macro_boot_mean:.4f}",
            "normal_ci_low": f"{res.macro_normal_ci_low:.4f}",
            "normal_ci_high": f"{res.macro_normal_ci_high:.4f}",
            "pct_ci_low": f"{res.macro_ci_low:.4f}",
            "pct_ci_high": f"{res.macro_ci_high:.4f}",
            "basic_ci_low": f"{res.macro_basic_ci_low:.4f}",
            "basic_ci_high": f"{res.macro_basic_ci_high:.4f}",
            "bca_ci_low": f"{res.macro_bca_ci_low:.4f}",
            "bca_ci_high": f"{res.macro_bca_ci_high:.4f}",
            "delta": "", "delta_normal_ci_low": "", "delta_normal_ci_high": "",
            "delta_pct_ci_low": "", "delta_pct_ci_high": "",
            "delta_basic_ci_low": "", "delta_basic_ci_high": "",
            "delta_bca_ci_low": "", "delta_bca_ci_high": "",
            "delta_boot_mean": "", "p_two_sided": "",
            "n": n, "seed": seed,
        })

    # Paired tests.
    print("Running paired bootstrap tests …")
    for a_sys, b_sys, scope in _PAIRED_TESTS:
        a_table = tables[(a_sys, scope)]
        b_table = tables[(b_sys, scope)]
        res = paired_bootstrap_test_from_tables(
            a_table, b_table, n=n, seed=seed, label=f"{a_sys}_vs_{b_sys}/{scope}",
        )
        for metric_name, delta_attr, p_attr in [
            ("micro", "delta_micro", "p_two_sided_micro"),
            ("macro", "delta_macro", "p_two_sided_macro"),
        ]:
            delta = getattr(res, delta_attr)
            normal_lo = getattr(res, f"{delta_attr}_normal_ci_low")
            normal_hi = getattr(res, f"{delta_attr}_normal_ci_high")
            pct_lo = getattr(res, f"{delta_attr}_ci_low")
            pct_hi = getattr(res, f"{delta_attr}_ci_high")
            basic_lo = getattr(res, f"{delta_attr}_basic_ci_low")
            basic_hi = getattr(res, f"{delta_attr}_basic_ci_high")
            bca_lo = getattr(res, f"{delta_attr}_bca_ci_low")
            bca_hi = getattr(res, f"{delta_attr}_bca_ci_high")
            boot_mean = getattr(res, f"{delta_attr}_boot_mean")
            rows.append({
                "kind": "paired", "config": f"{a_sys}_vs_{b_sys}",
                "scope": scope, "metric": metric_name,
                "point": "", "boot_mean": "",
                "normal_ci_low": "", "normal_ci_high": "",
                "pct_ci_low": "", "pct_ci_high": "",
                "basic_ci_low": "", "basic_ci_high": "",
                "bca_ci_low": "", "bca_ci_high": "",
                "delta": f"{delta:+.4f}",
                "delta_normal_ci_low": f"{normal_lo:+.4f}",
                "delta_normal_ci_high": f"{normal_hi:+.4f}",
                "delta_pct_ci_low": f"{pct_lo:+.4f}",
                "delta_pct_ci_high": f"{pct_hi:+.4f}",
                "delta_basic_ci_low": f"{basic_lo:+.4f}",
                "delta_basic_ci_high": f"{basic_hi:+.4f}",
                "delta_bca_ci_low": f"{bca_lo:+.4f}",
                "delta_bca_ci_high": f"{bca_hi:+.4f}",
                "delta_boot_mean": f"{boot_mean:+.4f}",
                "p_two_sided": f"{getattr(res, p_attr):.4f}",
                "n": n, "seed": seed,
            })

    out_csv = out_dir / (
        "bootstrap_results.csv" if filter_mdace_leak
        else "bootstrap_results_unfiltered.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["kind", "config", "scope", "metric", "point", "boot_mean",
                  "normal_ci_low", "normal_ci_high",
                  "pct_ci_low", "pct_ci_high", "basic_ci_low", "basic_ci_high",
                  "bca_ci_low", "bca_ci_high",
                  "delta", "delta_normal_ci_low", "delta_normal_ci_high",
                  "delta_pct_ci_low", "delta_pct_ci_high",
                  "delta_basic_ci_low", "delta_basic_ci_high",
                  "delta_bca_ci_low", "delta_bca_ci_high",
                  "delta_boot_mean", "p_two_sided", "n", "seed"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    marker = (
        "## Phase 1.3 — Bootstrap CIs and paired tests (filtered + normal/BCa) (auto-generated)"
        if filter_mdace_leak
        else "## Phase 1.2 — Bootstrap CIs and paired tests (auto-generated)"
    )
    _append_bootstrap_log(rows, out_csv, n=n, seed=seed, marker=marker)
    _print_significance_summary(rows)
    return 0


# --- Precision/Recall Decompositions appendix (tab:pr-decomp-*) -----------


# (system_label, scope, vocab_attr, with_topk).
# `vocab_attr` indexes into ctx (in/ood/full class maps); `with_topk` requests
# computation of P@k/R@k/R-precision from raw SFT scores. Pipeline rows leave
# `with_topk=False` since they emit binary code sets with no native ranking.
def _decomp_targets() -> list[tuple[str, str, str, bool]]:
    sft_systems = [
        "sft_only_tau_0.15", "sft_only_tau_0.35", "sft_only_tau_0.50",
        # Principled-band SFT operating points.
        "sft_only_tau_macro", "sft_only_tau_micro",
    ]
    targets: list[tuple[str, str, str, bool]] = []
    for system in sft_systems:
        targets.append((system, "in_dist", "in", True))
        targets.append((system, "full", "full", True))
        # OOD scope is unreachable for SFT (vocab disjoint from output) — skip.
    # Legacy hybrid: SFT τ=0.5 ∪ band-verify ∪ candidate-verify accepts.
    targets.append(("hybrid", "in_dist", "in", False))
    targets.append(("hybrid", "ood", "ood", False))
    targets.append(("hybrid", "full", "full", False))
    # Legacy stage-wise rows from tab:stages (FULL only).
    targets.append(("+band_verify", "full", "full", False))
    targets.append(("+gap_fill", "full", "full", False))
    # Principled-band hybrid + stage rows (composed off τ*_micro).
    targets.append(("hybrid_principled", "in_dist", "in", False))
    targets.append(("hybrid_principled", "ood", "ood", False))
    targets.append(("hybrid_principled", "full", "full", False))
    targets.append(("+band_verify_principled", "full", "full", False))
    targets.append(("+gap_fill_principled", "full", "full", False))
    return targets


def _vocab_for_scope(ctx: dict, vocab_attr: str) -> set[str]:
    if vocab_attr == "in":
        return set(ctx["in_class_maps"][0])
    if vocab_attr == "ood":
        return set(ctx["ood_class_maps_global"][0])
    if vocab_attr == "full":
        return set(ctx["full_class_maps_global"][0])
    raise ValueError(f"unknown vocab_attr: {vocab_attr!r}")


def _decomp_systems(ctx: dict, gap_lookup: dict[str, set[str]]) -> dict[str, dict[str, set[str]]]:
    """Pred-set lookup per row label (extends `_system_lookups` with `+gap_fill`)."""
    base = _system_lookups(ctx)
    base["+gap_fill"] = compose_predictions(ctx["sft_050"], ctx["band_verify"], gap_lookup)
    if "sft_micro" in ctx:
        base["+gap_fill_principled"] = compose_predictions(
            ctx["sft_micro"], ctx["band_verify"], gap_lookup,
        )
    return base


def pr_decomp(
    out_dir: Path = OUT_DIR, *, n: int = 1000, seed: int = 314,
    filter_mdace_leak: bool = True,
    macro_denominator: str = "full",
) -> int:
    """Compute Group-A (P/R/F1 + EMR + codes/note) and Group-B (top-k) decompositions.

    Output: ``outputs/revision/pr_decomp.csv`` with one row per
    ``(config, scope, metric)`` triple, carrying the deterministic point
    estimate, bootstrap mean, and 95% normal-approximation CI bounds. The
    ``+gap_fill`` row is reported point-only (no bootstrap), matching the
    `tab:stages` convention for that intermediate stage.
    """
    from coding.eval.bootstrap import (
        bootstrap_decomp_metrics,
        deterministic_decomp,
    )

    ctx = _load_mimic_iii_context(filter_mdace_leak=filter_mdace_leak)

    print("Loading SFT raw scores for ranking metrics …")
    sft_event_ids, sft_scores, codes_by_idx = load_sft_scores(SFT_DIR)
    if filter_mdace_leak:
        # Drop the same admissions that `_load_mimic_iii_context` filtered
        # (so admission ordering between scores and gold stays aligned).
        ctx_eid_set = set(ctx["event_ids"])
        keep = [i for i, e in enumerate(sft_event_ids) if e in ctx_eid_set]
        sft_event_ids = [sft_event_ids[i] for i in keep]
        sft_scores = sft_scores[keep]
        print(f"  filtered SFT scores: {sft_scores.shape[0]} admissions kept")

    print("Loading raw +gap_fill candidates …")
    gap_lookup = load_gap_candidates(MIMIC_III_DIR / "gap_analysis_pass2_searched.parquet")
    if filter_mdace_leak:
        for eid in list(gap_lookup.keys()):
            if eid not in set(ctx["event_ids"]):
                del gap_lookup[eid]

    systems = _decomp_systems(ctx, gap_lookup)

    rows: list[dict] = []
    print(f"Bootstrapping decomposition metrics (n={n}, seed={seed}) …")
    for system, scope, vocab_attr, with_topk in _decomp_targets():
        vocab = _vocab_for_scope(ctx, vocab_attr)
        kwargs = {}
        if with_topk:
            kwargs = dict(
                sft_event_ids=sft_event_ids,
                sft_scores=sft_scores,
                codes_by_idx=codes_by_idx,
            )
        table = _build_decomp_table(
            systems[system], ctx["gold_lookup"], ctx["event_ids"], vocab,
            **kwargs,
        )
        if system in {"+gap_fill", "+gap_fill_principled"}:
            point = deterministic_decomp(
                table, macro_denominator=macro_denominator,
            )
            for metric_name, val in point.items():
                rows.append({
                    "config": system, "scope": scope, "metric": metric_name,
                    "point": f"{val:.4f}",
                    "boot_mean": "", "normal_ci_low": "", "normal_ci_high": "",
                    "n": "", "seed": "",
                })
            print(f"  [{system}/{scope}] point only (no bootstrap)")
            continue
        res = bootstrap_decomp_metrics(
            table, n=n, seed=seed, macro_denominator=macro_denominator,
        )
        for metric_name in res.point.keys():
            rows.append({
                "config": system, "scope": scope, "metric": metric_name,
                "point": f"{res.point[metric_name]:.4f}",
                "boot_mean": f"{res.boot_mean[metric_name]:.4f}",
                "normal_ci_low": f"{res.normal_ci_lo[metric_name]:.4f}",
                "normal_ci_high": f"{res.normal_ci_hi[metric_name]:.4f}",
                "n": n, "seed": seed,
            })
        print(f"  [{system}/{scope}] {len(res.point)} metrics ✓")

    csv_stem = "pr_decomp" if filter_mdace_leak else "pr_decomp_unfiltered"
    if macro_denominator == "scorable":
        csv_stem += "_scorable"
    out_csv = out_dir / f"{csv_stem}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config", "scope", "metric", "point", "boot_mean",
                  "normal_ci_low", "normal_ci_high", "n", "seed"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")
    return 0


def _append_bootstrap_log(
    rows: list[dict], csv_path: Path, *, n: int, seed: int,
    marker: str = "## Phase 1.2 — Bootstrap CIs and paired tests (auto-generated)",
) -> None:
    lines = ["", marker, ""]
    lines.append(f"- Results CSV: `{csv_path.relative_to(REPO_ROOT)}`")
    lines.append(f"- n_resamples = {n}, seed = {seed}, resampling unit = admission_id")
    lines.append("")
    lines.append("### Methodological note: normal vs BCa vs percentile vs basic CI")
    lines.append("")
    lines.append(
        "The bootstrap distribution of macro-F1 is biased downward relative to "
        "the deterministic point estimate because rare classes drop out of "
        "admission resamples and contribute 0 to the macro denominator. As a "
        "result: the **percentile** CI sits below the point on macro rows; "
        "the **basic/pivot** CI `[2·point − pct_high, 2·point − pct_low]` "
        "reflects the percentile through the point and lands above it; **BCa** "
        "(bias-corrected, jackknife-accelerated) absorbs moderate bias but "
        "collapses to the upper percentile when the point lies far above the "
        "bootstrap distribution (the case for hybrid FULL macro). The headline "
        "interval is therefore the **normal-approximation** CI "
        "`point ± 1.96 · SE_boot`, which centres on the point estimate and "
        "uses the bootstrap-derived standard error — it brackets the point by "
        "construction and is the standard interval reported across A/B "
        "testing and clinical-prediction literatures when the point estimator "
        "is unbiased and the bootstrap is used only for SE. Percentile, "
        "basic, and BCa are retained as diagnostics. The p-value is computed "
        "from the proportion of bootstrap deltas crossing zero and is robust "
        "to this bias."
    )
    lines.append("")
    lines.append("### Headline paired tests (FULL scope, MIMIC-III)")
    lines.append("")
    lines.append("| comparison | metric | Δ | 95% normal CI | 95% BCa CI | 95% pct CI | 95% basic CI | p (two-sided) | significant? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kind"] != "paired":
            continue
        sig = "**yes**" if float(r["p_two_sided"]) < 0.05 else "no"
        lines.append(
            f"| {r['config']} | {r['metric']} | {r['delta']} "
            f"| [{r['delta_normal_ci_low']}, {r['delta_normal_ci_high']}] "
            f"| [{r['delta_bca_ci_low']}, {r['delta_bca_ci_high']}] "
            f"| [{r['delta_pct_ci_low']}, {r['delta_pct_ci_high']}] "
            f"| [{r['delta_basic_ci_low']}, {r['delta_basic_ci_high']}] "
            f"| {r['p_two_sided']} | {sig} |"
        )
    lines.append("")
    lines.append("### Per-cell CIs (Table 2 + Table 3 stages, FULL only for stages)")
    lines.append("")
    lines.append("| config | scope | metric | point | 95% normal CI | 95% BCa CI | 95% pct CI | 95% basic CI |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["kind"] != "ci":
            continue
        lines.append(
            f"| {r['config']} | {r['scope']} | {r['metric']} | {r['point']} "
            f"| [{r['normal_ci_low']}, {r['normal_ci_high']}] "
            f"| [{r['bca_ci_low']}, {r['bca_ci_high']}] "
            f"| [{r['pct_ci_low']}, {r['pct_ci_high']}] "
            f"| [{r['basic_ci_low']}, {r['basic_ci_high']}] |"
        )
    lines.append("")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        existing = LOG_PATH.read_text()
        if marker in existing:
            existing = existing.split(marker)[0].rstrip() + "\n"
        LOG_PATH.write_text(existing + "\n".join(lines))
    else:
        LOG_PATH.write_text("# Revision log\n" + "\n".join(lines))


def _print_significance_summary(rows: list[dict]) -> None:
    print("\n=== Phase 1.2 significance summary (FULL scope, MIMIC-III) ===")
    headline_macro = next(
        r for r in rows
        if r["kind"] == "paired"
        and r["config"] == "hybrid_vs_sft_only_tau_0.15"
        and r["metric"] == "macro"
    )
    headline_micro = next(
        r for r in rows
        if r["kind"] == "paired"
        and r["config"] == "hybrid_vs_sft_only_tau_0.15"
        and r["metric"] == "micro"
    )
    p_mac = float(headline_macro["p_two_sided"])
    p_mi = float(headline_micro["p_two_sided"])
    print(f"  Hybrid vs SFT τ=0.15 FULL macro Δ {headline_macro['delta']} "
          f"(95% normal CI [{headline_macro['delta_normal_ci_low']}, {headline_macro['delta_normal_ci_high']}], "
          f"BCa [{headline_macro['delta_bca_ci_low']}, {headline_macro['delta_bca_ci_high']}], "
          f"pct [{headline_macro['delta_pct_ci_low']}, {headline_macro['delta_pct_ci_high']}]), "
          f"p={p_mac:.4f}")
    print(f"  Hybrid vs SFT τ=0.15 FULL micro Δ {headline_micro['delta']} "
          f"(95% normal CI [{headline_micro['delta_normal_ci_low']}, {headline_micro['delta_normal_ci_high']}]), "
          f"p={p_mi:.4f}")
    if p_mac >= 0.05:
        print("  🚦 HEADLINE GATE: Hybrid vs SFT τ=0.15 FULL macro is NOT significant at p<0.05.")
        print("     Per revision.md, surface this for framing decision before continuing.")
    else:
        print("  ✓ Headline macro comparison reaches p<0.05 (gate clears).")


# --- Phase 3 + 4 re-evaluation at principled thresholds -----------------


def _scope_vocabs(ctx: dict) -> dict[str, set[str]]:
    return {
        "in_dist": set(ctx["in_class_maps"][0]),
        "ood": set(ctx["ood_class_maps_global"][0]),
        "full": set(ctx["full_class_maps_global"][0]),
    }


def _scope_metrics_for_pred(
    pred_lookup: dict[str, set[str]],
    ctx: dict,
    *,
    n: int,
    seed: int,
    label: str,
    macro_denominator: str = "full",
):
    """Run point P/R/F1 + bootstrap CI for one prediction lookup across three scopes.

    Returns ``(scope_metrics_dict, scope_to_index_table)`` so callers can reuse
    the index tables for paired tests without rebuilding them.
    """
    from coding.eval.bootstrap import (
        bootstrap_f1_from_table,
        build_index_table,
    )

    scope_vocabs = _scope_vocabs(ctx)
    out: dict[str, dict] = {}
    tables: dict[str, "object"] = {}
    for scope, vocab in scope_vocabs.items():
        # Use the existing per-cell counter for deterministic P/R/F1.
        pr_dict = _score_one_scope(
            [pred_lookup.get(eid, set()) for eid in ctx["event_ids"]],
            [ctx["gold_lookup"].get(eid, set()) for eid in ctx["event_ids"]],
            vocab,
            macro_denominator=macro_denominator,
        )
        table = build_index_table(
            pred_lookup, ctx["gold_lookup"], ctx["event_ids"], vocab,
        )
        tables[scope] = table
        if not vocab:
            # Skip bootstrap on an empty vocab (would 0-divide); emit zeros.
            out[scope] = {
                "vocab_size": 0,
                "n_scorable": 0,
                "f1_micro": 0.0, "f1_macro": 0.0,
                "precision_micro": 0.0, "recall_micro": 0.0,
                "precision_macro": 0.0, "recall_macro": 0.0,
                "f1_micro_normal_ci": [0.0, 0.0],
                "f1_macro_normal_ci": [0.0, 0.0],
                "f1_micro_boot_mean": 0.0, "f1_macro_boot_mean": 0.0,
            }
            continue
        boot = bootstrap_f1_from_table(
            table, n=n, seed=seed, label=f"{label}/{scope}",
            macro_denominator=macro_denominator,
        )
        out[scope] = {
            "vocab_size": len(vocab),
            "n_scorable": int(pr_dict.get("n_scorable", 0)),
            "f1_micro": pr_dict["f1_micro"],
            "f1_macro": pr_dict["f1_macro"],
            "precision_micro": pr_dict["precision_micro"],
            "recall_micro": pr_dict["recall_micro"],
            "precision_macro": pr_dict["precision_macro"],
            "recall_macro": pr_dict["recall_macro"],
            "f1_micro_normal_ci": [
                boot.micro_normal_ci_low, boot.micro_normal_ci_high,
            ],
            "f1_macro_normal_ci": [
                boot.macro_normal_ci_low, boot.macro_normal_ci_high,
            ],
            "f1_micro_boot_mean": boot.micro_boot_mean,
            "f1_macro_boot_mean": boot.macro_boot_mean,
        }
    return out, tables


def _round_floats(obj, ndigits: int = 4):
    """Recursively round all floats to *ndigits* decimal places for JSON emit."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_round_floats(v, ndigits) for v in obj)
    return obj


def _write_config_json(
    out_path: Path,
    *,
    spec: DatasetSpec,
    config: str,
    operating_point: dict,
    scope_metrics: dict,
    n_admissions: int,
    n_resamples: int,
    seed: int,
    filter_mdace_leak: bool,
) -> None:
    payload = {
        "dataset": spec.name,
        "config": config,
        "operating_point": operating_point,
        "thresholds_source": str(
            spec.thresholds_source.relative_to(REPO_ROOT)
        ),
        "n_admissions": n_admissions,
        "n_resamples": n_resamples,
        "seed": seed,
        "filter_mdace_leak": filter_mdace_leak,
        "scope_metrics": scope_metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_round_floats(payload), indent=2) + "\n")


def _write_paired_tests_json(
    out_path: Path,
    *,
    spec: DatasetSpec,
    scope: str,
    paired_results: list[dict],
    n_resamples: int,
    seed: int,
) -> None:
    payload = {
        "dataset": spec.name,
        "scope": scope,
        "n_resamples": n_resamples,
        "seed": seed,
        "tests": paired_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_round_floats(payload), indent=2) + "\n")


def re_eval(
    out_dir: Path = OUT_DIR,
    *,
    n: int = 1000,
    seed: int = 314,
    filter_mdace_leak: bool | None = None,
) -> int:
    """Phase 3 + 4 + 5 re-evaluation at principled thresholds for both datasets.

    For each dataset spec:
      * Phase 3 — SFT-only at τ*_macro and τ*_micro (no τ=0.5 row per the
        Lipton reframe).
      * Phase 4 — SFT high-tier ∪ band-verifier-confirmed (band already
        pre-filtered to ``[τ*_macro, τ*_micro)`` in stage-2 outputs).
      * Phase 5 — Full pipeline: SFT high-tier ∪ band-verifier-confirmed ∪
        candidate-verifier-confirmed (gap-fill).

    Per-config JSONs are written to
    ``outputs/revision/results/{sft_only|sft_band_verify|full_pipeline}/{dataset}/<name>.json``.
    Paired bootstrap tests on Full scope (``+band_verify`` vs each SFT-only,
    plus ``full_pipeline`` vs sft_only_tau_micro and vs sft_band_verify) are
    written to ``outputs/revision/results/paired_tests/<dataset>.json``.

    Set *filter_mdace_leak* explicitly to ``True``/``False`` to override the
    spec's default. ``None`` defers to ``spec.mdace_filter_applies`` per dataset.
    """
    from coding.eval.bootstrap import paired_bootstrap_test_from_tables

    log_lines: list[str] = []
    for spec in (SPEC,):
        print(f"\n=== re-eval: {spec.name} ===")
        ctx = _load_dataset_context(
            spec,
            sft_thresholds=[spec.tau_macro, spec.tau_micro],
            filter_mdace_leak=filter_mdace_leak,
        )
        sft_macro = ctx["sft_lookups"][spec.tau_macro]
        sft_micro = ctx["sft_lookups"][spec.tau_micro]
        band_verify = ctx["band_verify"]
        candidate_verify = ctx["candidate_verify"]
        n_adm = len(ctx["event_ids"])

        # Build the four predicted-set lookups for this dataset.
        configs: dict[str, dict[str, set[str]]] = {
            "sft_only_tau_macro": compose_predictions(sft_macro),
            "sft_only_tau_micro": compose_predictions(sft_micro),
            "sft_band_verify": compose_predictions(sft_micro, band_verify),
            "full_pipeline": compose_predictions(
                sft_micro, band_verify, candidate_verify
            ),
        }

        # Phase 3 + Phase 4 metrics + index tables (latter reused for paired tests).
        all_scope_metrics: dict[str, dict] = {}
        all_tables: dict[str, dict] = {}
        for cfg_name, pred_lookup in configs.items():
            print(f"  bootstrapping {cfg_name} (n={n}, seed={seed}) …")
            scope_metrics, tables = _scope_metrics_for_pred(
                pred_lookup, ctx, n=n, seed=seed,
                label=f"{spec.name}/{cfg_name}",
            )
            all_scope_metrics[cfg_name] = scope_metrics
            all_tables[cfg_name] = tables

        # Emit per-config JSONs.
        op_macro = {"name": "tau_macro", "tau_low": spec.tau_macro, "tau_high": None}
        op_micro = {"name": "tau_micro", "tau_low": spec.tau_micro, "tau_high": None}
        op_band = {
            "name": "band", "tau_low": spec.tau_macro, "tau_high": spec.tau_micro,
        }
        op_full = {
            "name": "full_pipeline",
            "tau_low": spec.tau_macro,
            "tau_high": spec.tau_micro,
            "stages": ["sft_micro", "band_verify", "candidate_verify"],
        }
        emit_specs = [
            ("sft_only", "sft_only_tau_macro", "tau_macro.json", op_macro),
            ("sft_only", "sft_only_tau_micro", "tau_micro.json", op_micro),
            (
                "sft_band_verify",
                "sft_band_verify",
                f"band_{spec.tau_macro:.3f}-{spec.tau_micro:.3f}.json",
                op_band,
            ),
            (
                "full_pipeline",
                "full_pipeline",
                f"full_band_{spec.tau_macro:.3f}-{spec.tau_micro:.3f}.json",
                op_full,
            ),
        ]
        for stage_dir, cfg_name, fname, op in emit_specs:
            out_path = out_dir / "results" / stage_dir / spec.name / fname
            _write_config_json(
                out_path,
                spec=spec, config=cfg_name, operating_point=op,
                scope_metrics=all_scope_metrics[cfg_name],
                n_admissions=n_adm, n_resamples=n, seed=seed,
                filter_mdace_leak=ctx["filter_mdace_leak"],
            )
            print(f"  wrote {out_path.relative_to(REPO_ROOT)}")

        # Paired tests on Full scope:
        #   +band_verify vs each SFT-only config (Phase 4 lifts);
        #   full_pipeline vs sft_only_tau_micro and vs sft_band_verify (Phase 5 lifts).
        paired_pairs = [
            ("sft_band_verify", "sft_only_tau_macro"),
            ("sft_band_verify", "sft_only_tau_micro"),
            ("full_pipeline", "sft_only_tau_micro"),
            ("full_pipeline", "sft_band_verify"),
        ]
        paired_payload: list[dict] = []
        for a_cfg, b_cfg in paired_pairs:
            res = paired_bootstrap_test_from_tables(
                all_tables[a_cfg]["full"],
                all_tables[b_cfg]["full"],
                n=n, seed=seed,
                label=f"{spec.name}/{a_cfg}_vs_{b_cfg}",
            )
            paired_payload.append({
                "a_config": a_cfg,
                "b_config": b_cfg,
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
                "p_two_sided_micro": res.p_two_sided_micro,
                "p_two_sided_macro": res.p_two_sided_macro,
            })
        paired_path = (
            out_dir / "results" / "paired_tests" / f"{spec.name}.json"
        )
        _write_paired_tests_json(
            paired_path,
            spec=spec, scope="full", paired_results=paired_payload,
            n_resamples=n, seed=seed,
        )
        print(f"  wrote {paired_path.relative_to(REPO_ROOT)}")

        # Build a per-dataset summary block for the revision log.
        log_lines.append(f"\n### {spec.name}\n")
        log_lines.append(
            f"- thresholds: τ\\*_macro = {spec.tau_macro}, τ\\*_micro = {spec.tau_micro}; "
            f"band = `[{spec.tau_macro}, {spec.tau_micro})`."
        )
        log_lines.append(
            f"- n admissions = {n_adm}; "
            f"filter_mdace_leak = {ctx['filter_mdace_leak']}; "
            f"mdace_dropped = {len(ctx['mdace_leak_dropped'])}; "
            f"label_code_type_filter = {spec.label_code_type_filter!r}."
        )
        log_lines.append("")
        log_lines.append("| config | scope | F1_micro [95% normal CI] | F1_macro [95% normal CI] |")
        log_lines.append("|---|---|---|---|")
        for cfg_name in (
            "sft_only_tau_macro", "sft_only_tau_micro",
            "sft_band_verify", "full_pipeline",
        ):
            for scope in ("in_dist", "ood", "full"):
                m = all_scope_metrics[cfg_name][scope]
                lo_mi, hi_mi = m["f1_micro_normal_ci"]
                lo_ma, hi_ma = m["f1_macro_normal_ci"]
                log_lines.append(
                    f"| {cfg_name} | {scope} "
                    f"| {m['f1_micro']:.4f} [{lo_mi:.4f}, {hi_mi:.4f}] "
                    f"| {m['f1_macro']:.4f} [{lo_ma:.4f}, {hi_ma:.4f}] |"
                )
        log_lines.append("")
        log_lines.append("Paired tests on Full scope:")
        log_lines.append("")
        log_lines.append("| comparison | metric | Δ | 95% normal CI | p (two-sided) |")
        log_lines.append("|---|---|---|---|---|")
        for t in paired_payload:
            for metric, delta_key, ci_key, p_key in [
                ("micro", "delta_micro", "delta_micro_normal_ci", "p_two_sided_micro"),
                ("macro", "delta_macro", "delta_macro_normal_ci", "p_two_sided_macro"),
            ]:
                lo, hi = t[ci_key]
                log_lines.append(
                    f"| {t['a_config']} vs {t['b_config']} | {metric} "
                    f"| {t[delta_key]:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {t[p_key]:.4f} |"
                )

    # Append the consolidated section to the revision log.
    marker = "## Phase 3 + 4 + 5 — Re-evaluation at principled thresholds (auto-generated)"
    header = [
        "",
        marker,
        "",
        f"- n_resamples = {n}, seed = {seed}, resampling unit = admission_id.",
        "- Headline interval is the normal-approximation CI `point ± 1.96 · SE_boot`.",
        "- SFT-only on `ood` is structurally zero (in-dist vocab disjoint from OOD vocab).",
        "- Outputs under `outputs/revision/results/` (per-config JSONs + paired-tests JSONs).",
    ]
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(header + log_lines) + "\n"
    legacy_marker = (
        "## Phase 3 + 4 — Re-evaluation at principled thresholds (auto-generated)"
    )
    if LOG_PATH.exists():
        existing = LOG_PATH.read_text()
        for m in (marker, legacy_marker):
            if m in existing:
                existing = existing.split(m)[0].rstrip() + "\n"
        LOG_PATH.write_text(existing + body)
    else:
        LOG_PATH.write_text("# Revision log\n" + body)
    print(f"\nappended {marker!r} to {LOG_PATH.relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        print("\nSubcommands: validate | validate-filtered | bootstrap "
              "[--no-filter-mdace-leak] | pr-decomp [--no-filter-mdace-leak] "
              "[--macro-denominator=full|scorable] "
              "| re-eval [--filter-mdace-leak | --no-filter-mdace-leak]")
        return 0
    cmd = argv[0]
    rest = argv[1:]
    if cmd == "validate":
        return validate(filter_mdace_leak=False)
    if cmd == "validate-filtered":
        return validate(filter_mdace_leak=True)
    if cmd == "bootstrap":
        filter_leak = "--no-filter-mdace-leak" not in rest
        return bootstrap(filter_mdace_leak=filter_leak)
    if cmd == "pr-decomp":
        filter_leak = "--no-filter-mdace-leak" not in rest
        n_arg = next(
            (int(a.split("=", 1)[1]) for a in rest if a.startswith("--n=")), 1000,
        )
        denom_arg = next(
            (a.split("=", 1)[1] for a in rest if a.startswith("--macro-denominator=")),
            "full",
        )
        if denom_arg not in {"full", "scorable"}:
            print(
                f"unknown --macro-denominator value: {denom_arg!r}", file=sys.stderr,
            )
            return 2
        return pr_decomp(
            filter_mdace_leak=filter_leak, n=n_arg,
            macro_denominator=denom_arg,
        )
    if cmd == "re-eval":
        if "--no-filter-mdace-leak" in rest:
            filter_leak: bool | None = False
        elif "--filter-mdace-leak" in rest:
            filter_leak = True
        else:
            filter_leak = None  # spec-driven default per dataset
        n_arg = next(
            (int(a.split("=", 1)[1]) for a in rest if a.startswith("--n=")), 1000,
        )
        return re_eval(filter_mdace_leak=filter_leak, n=n_arg)
    print(f"unknown subcommand: {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
