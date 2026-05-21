"""Per-class threshold pilot for the top-50 MIMIC-III head codes.

Usage:
    python -m coding.eval.tables.per_class_thresholds

Tunes per-class thresholds on the head codes using ``${dataset_path}val_predictions.json``,
then evaluates three configurations on ``${dataset_path}predictions.json``:
    A) global tau*_macro applied to every code (baseline),
    B) per-class tuned thresholds on top-50, tau*_macro on the rest (pilot),
    C) per-class tuned thresholds on top-50, restricted to those columns (sanity).

Results are written under
``outputs/eval/results/per_class_threshold_pilot/<dataset_name>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import polars as pl

from coding.eval.paths import (
    DATASET_DIR,
    EVAL_DIR,
    FREQ_BANDS_PATH,
    REPO_ROOT,
)
from coding.eval.harness import SPEC

DEFAULT_HEAD_JSON = FREQ_BANDS_PATH
DEFAULT_THRESHOLDS = DATASET_DIR / "optimal_thresholds.json"


def load_probs_and_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a predictions JSON and return (probs, labels) as (N, C) arrays."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or "predictions" not in raw or "labels" not in raw:
        raise ValueError(
            f"{path} is not in the expected column-dict shape "
            "({'predictions': [[...]], 'labels': [[...]], ...})"
        )
    probs = np.asarray(raw["predictions"], dtype=np.float32)
    labels = np.asarray(raw["labels"], dtype=np.int8)
    if probs.shape != labels.shape:
        raise ValueError(
            f"predictions {probs.shape} and labels {labels.shape} disagree in {path}"
        )
    return probs, labels


def load_class_map(path: Path) -> dict[str, int]:
    """Return {code: column index} from a stage-1 class_map.parquet."""
    df = pl.read_parquet(path)
    return {str(k): int(v) for k, v in zip(df["key"].to_list(), df["value"].to_list(), strict=True)}


def load_global_thresholds(path: Path) -> tuple[float, float]:
    """Read {"micro": float, "macro": float} produced by evaluate_sft.py."""
    with open(path) as f:
        d = json.load(f)
    return float(d["macro"]), float(d["micro"])


def per_class_f1(
    probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For a (T,) threshold grid and (N, C) probs/labels, return ((T, C) F1, (C,) n_pos)."""
    labels_i8 = labels.astype(np.int8, copy=False)
    pos_per_class = labels_i8.sum(axis=0).astype(np.int64)  # (C,)
    f1_matrix = np.zeros((len(thresholds), labels.shape[1]), dtype=np.float64)
    for t_idx, t in enumerate(thresholds):
        preds = (probs >= t).astype(np.int8)
        tp = (preds & labels_i8).sum(axis=0).astype(np.int64)
        pred_pos = preds.sum(axis=0).astype(np.int64)
        fp = pred_pos - tp
        fn = pos_per_class - tp
        denom = 2 * tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_matrix[t_idx] = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1_matrix, pos_per_class


def pick_per_class_optima(
    probs: np.ndarray,
    labels: np.ndarray,
    coarse_step: float = 0.01,
    fine_step: float = 0.001,
    fine_window: float = 0.02,
    tie_break_anchor: float = 0.125,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class F1-optimal thresholds via coarse-then-fine search.

    Returns (tau_pc (C,), f1_at_tau_pc (C,), n_val_pos (C,)).
    Ties broken in favour of the threshold closest to `tie_break_anchor`.
    """
    coarse = np.arange(coarse_step, 1.0, coarse_step, dtype=np.float64)
    coarse_f1, n_pos = per_class_f1(probs, labels, coarse)  # (T_coarse, C), (C,)

    n_classes = probs.shape[1]
    tau_pc = np.full(n_classes, np.nan, dtype=np.float64)
    f1_pc = np.zeros(n_classes, dtype=np.float64)

    # coarse argmax per class (with tie-break)
    coarse_argmax = np.empty(n_classes, dtype=np.int64)
    for c in range(n_classes):
        col = coarse_f1[:, c]
        best = col.max()
        candidates = np.flatnonzero(col >= best - 1e-12)
        # pick the candidate whose threshold is closest to the anchor
        cand_thr = coarse[candidates]
        coarse_argmax[c] = candidates[int(np.argmin(np.abs(cand_thr - tie_break_anchor)))]

    # fine sweep per class around its coarse argmax
    for c in range(n_classes):
        centre = float(coarse[coarse_argmax[c]])
        lo = max(coarse_step, centre - fine_window)
        hi = min(1.0 - fine_step / 2, centre + fine_window)
        fine = np.arange(lo, hi + fine_step / 2, fine_step)
        fine_f1, _ = per_class_f1(probs[:, c : c + 1], labels[:, c : c + 1], fine)
        col = fine_f1[:, 0]
        # combine coarse-best with the fine grid for the global argmax
        full_thr = np.concatenate([fine, [centre]])
        full_f1 = np.concatenate([col, [coarse_f1[coarse_argmax[c], c]]])
        best = full_f1.max()
        cand = np.flatnonzero(full_f1 >= best - 1e-12)
        cand_thr = full_thr[cand]
        pick = cand[int(np.argmin(np.abs(cand_thr - tie_break_anchor)))]
        tau_pc[c] = full_thr[pick]
        f1_pc[c] = full_f1[pick]

    return tau_pc, f1_pc, n_pos


def metrics_with_vector_threshold(
    probs: np.ndarray, labels: np.ndarray, theta: np.ndarray, col_mask: np.ndarray | None = None
) -> dict[str, float]:
    """Full metric suite at a per-class threshold vector. If `col_mask` is given,
    metrics are restricted to those columns (for the top-50-only sanity config)."""
    labels_i8 = labels.astype(np.int8, copy=False)
    preds = (probs >= theta[None, :]).astype(np.int8)
    if col_mask is not None:
        preds = preds[:, col_mask]
        labels_i8 = labels_i8[:, col_mask]

    pos_per_class = labels_i8.sum(axis=0).astype(np.int64)
    tp_c = (preds & labels_i8).sum(axis=0).astype(np.int64)
    pred_per_class = preds.sum(axis=0).astype(np.int64)
    fp_c = pred_per_class - tp_c
    fn_c = pos_per_class - tp_c

    tp_total = int(tp_c.sum())
    fp_total = int(fp_c.sum())
    fn_total = int(fn_c.sum())
    n_cells = int(labels_i8.size)
    tn_total = n_cells - tp_total - fp_total - fn_total

    mi_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    mi_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    mi_f = 2 * mi_p * mi_r / (mi_p + mi_r) if (mi_p + mi_r) > 0 else 0.0
    accuracy = (tp_total + tn_total) / n_cells if n_cells > 0 else 0.0

    denom_p = tp_c + fp_c
    denom_r = tp_c + fn_c
    denom_f = 2 * tp_c + fp_c + fn_c
    with np.errstate(divide="ignore", invalid="ignore"):
        per_p = np.where(denom_p > 0, tp_c / np.maximum(denom_p, 1), 0.0)
        per_r = np.where(denom_r > 0, tp_c / np.maximum(denom_r, 1), 0.0)
        per_f = np.where(denom_f > 0, 2 * tp_c / np.maximum(denom_f, 1), 0.0)

    return {
        "f1_micro": float(mi_f),
        "f1_macro": float(per_f.mean()),
        "precision_micro": float(mi_p),
        "precision_macro": float(per_p.mean()),
        "recall_micro": float(mi_r),
        "recall_macro": float(per_r.mean()),
        "accuracy": float(accuracy),
        "n_classes_in_scope": int(labels_i8.shape[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sft-dir", type=Path, default=DATASET_DIR)
    ap.add_argument("--head-codes-json", type=Path, default=DEFAULT_HEAD_JSON)
    ap.add_argument("--global-thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    sft_dir = args.sft_dir.resolve()
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else EVAL_DIR / "results" / "per_class_threshold_pilot" / SPEC.name
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds_path = args.global_thresholds
    tau_macro, tau_micro = load_global_thresholds(thresholds_path)
    print(f"[thresholds] global tau_macro={tau_macro}  tau_micro={tau_micro}  (from {thresholds_path})")

    with open(args.head_codes_json) as f:
        freq_bands = json.load(f)
    head_codes = list(freq_bands["head_codes"])
    print(f"[head] loaded {len(head_codes)} head codes from {args.head_codes_json}")

    class_map = load_class_map(sft_dir / "class_map.parquet")
    print(f"[class_map] {len(class_map)} codes")

    resolved: list[tuple[str, int]] = []
    missing: list[str] = []
    for code in head_codes:
        if code in class_map:
            resolved.append((code, class_map[code]))
        else:
            missing.append(code)
    if missing:
        print(f"[warn] {len(missing)} head codes missing from class_map: {missing[:5]}...")
    print(f"[head] resolved {len(resolved)}/{len(head_codes)} head codes")
    head_idx = np.array([i for _, i in resolved], dtype=np.int64)

    print(f"[load] val: {sft_dir / 'val_predictions.json'}")
    val_probs, val_labels = load_probs_and_labels(sft_dir / "val_predictions.json")
    print(f"       probs={val_probs.shape}, labels={val_labels.shape}")

    val_probs_head = val_probs[:, head_idx]
    val_labels_head = val_labels[:, head_idx]
    print(f"[sweep] per-class coarse+fine search on {len(head_idx)} head codes...")
    tau_pc_head, val_f1_at_tau_pc, n_val_pos = pick_per_class_optima(
        val_probs_head, val_labels_head, tie_break_anchor=tau_macro
    )

    # validation F1 at global tau_macro for the same head codes (baseline)
    val_f1_at_tau_macro_grid, _ = per_class_f1(val_probs_head, val_labels_head, np.array([tau_macro]))
    val_f1_at_tau_macro = val_f1_at_tau_macro_grid[0]

    # fall back to tau_macro for head codes with zero validation positives
    zero_pos_mask = n_val_pos == 0
    if zero_pos_mask.any():
        print(f"[warn] {int(zero_pos_mask.sum())} head codes have 0 validation positives; using tau_macro for them")
        tau_pc_head[zero_pos_mask] = tau_macro
        val_f1_at_tau_pc[zero_pos_mask] = val_f1_at_tau_macro[zero_pos_mask]

    # sanity: tuned per-class F1 must be >= baseline per-class F1 on val (by construction)
    eps = 1e-9
    deficit = (val_f1_at_tau_macro - val_f1_at_tau_pc) > eps
    if deficit.any():
        bad = np.flatnonzero(deficit)
        raise AssertionError(
            f"per-class tuned F1 < baseline F1 on val for codes "
            f"{[resolved[int(i)][0] for i in bad]} — bug in sweep"
        )

    per_class_optima = {}
    for j, (code, _) in enumerate(resolved):
        per_class_optima[code] = {
            "tau_pc": float(tau_pc_head[j]),
            "n_val_pos": int(n_val_pos[j]),
            "val_f1_at_tau_macro": float(val_f1_at_tau_macro[j]),
            "val_f1_at_tau_pc": float(val_f1_at_tau_pc[j]),
        }
    optima_path = out_dir / "per_class_optima.json"
    with open(optima_path, "w") as f:
        json.dump(
            {"tau_macro_baseline": tau_macro, "tau_micro_baseline": tau_micro, "per_class": per_class_optima},
            f,
            indent=2,
        )
    print(f"[write] {optima_path}")

    del val_probs, val_labels, val_probs_head, val_labels_head

    print(f"[load] test: {sft_dir / 'predictions.json'}")
    test_probs, test_labels = load_probs_and_labels(sft_dir / "predictions.json")
    print(f"       probs={test_probs.shape}, labels={test_labels.shape}")

    n_classes = test_probs.shape[1]

    # build threshold vectors for configs A and B
    theta_A = np.full(n_classes, tau_macro, dtype=np.float64)
    theta_B = theta_A.copy()
    theta_B[head_idx] = tau_pc_head

    head_mask = np.zeros(n_classes, dtype=bool)
    head_mask[head_idx] = True
    nonhead_mask = ~head_mask

    test_A = metrics_with_vector_threshold(test_probs, test_labels, theta_A)
    test_B = metrics_with_vector_threshold(test_probs, test_labels, theta_B)
    test_A_top50 = metrics_with_vector_threshold(test_probs, test_labels, theta_A, col_mask=head_mask)
    test_B_top50 = metrics_with_vector_threshold(test_probs, test_labels, theta_B, col_mask=head_mask)
    test_A_rest = metrics_with_vector_threshold(test_probs, test_labels, theta_A, col_mask=nonhead_mask)
    test_B_rest = metrics_with_vector_threshold(test_probs, test_labels, theta_B, col_mask=nonhead_mask)

    # non-head columns must be identical between A and B
    for k in ("f1_macro", "f1_micro", "precision_macro", "recall_macro"):
        if abs(test_A_rest[k] - test_B_rest[k]) > 1e-9:
            raise AssertionError(
                f"non-head {k} differs between A and B (A={test_A_rest[k]}, B={test_B_rest[k]})"
            )
    print("[sanity] non-head metrics identical between A and B ✓")

    summary = {
        "delta_f1_macro_top50": test_B_top50["f1_macro"] - test_A_top50["f1_macro"],
        "delta_f1_micro_top50": test_B_top50["f1_micro"] - test_A_top50["f1_micro"],
        "delta_f1_macro_overall": test_B["f1_macro"] - test_A["f1_macro"],
        "delta_f1_micro_overall": test_B["f1_micro"] - test_A["f1_micro"],
    }

    test_metrics = {
        "sft_dir": str(sft_dir),
        "tuned_on": "val_predictions.json",
        "evaluated_on": "predictions.json",
        "tau_macro_baseline": tau_macro,
        "tau_micro_baseline": tau_micro,
        "n_head_codes_resolved": len(resolved),
        "n_head_codes_zero_val_pos": int(zero_pos_mask.sum()),
        "config_A_global_tau_macro": {"all": test_A, "top50": test_A_top50, "rest": test_A_rest},
        "config_B_hybrid_per_class": {"all": test_B, "top50": test_B_top50, "rest": test_B_rest},
        "summary_deltas_B_minus_A": summary,
    }
    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[write] {metrics_path}")

    # per-class table for paper appendix
    test_f1_at_tau_macro_grid, _ = per_class_f1(test_probs[:, head_idx], test_labels[:, head_idx], np.array([tau_macro]))
    test_f1_at_tau_macro = test_f1_at_tau_macro_grid[0]
    # evaluate test F1 at each code's tuned threshold (slice once, then per-class)
    test_f1_at_tau_pc = np.zeros(len(resolved), dtype=np.float64)
    for j in range(len(resolved)):
        grid, _ = per_class_f1(
            test_probs[:, head_idx[j] : head_idx[j] + 1],
            test_labels[:, head_idx[j] : head_idx[j] + 1],
            np.array([tau_pc_head[j]]),
        )
        test_f1_at_tau_pc[j] = grid[0, 0]

    table_path = out_dir / "per_class_table.csv"
    with open(table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "code", "n_val_pos", "tau_pc",
            "val_f1_at_tau_macro", "val_f1_at_tau_pc",
            "test_f1_at_tau_macro", "test_f1_at_tau_pc",
            "test_f1_delta",
        ])
        for j, (code, _) in enumerate(resolved):
            w.writerow([
                code,
                int(n_val_pos[j]),
                f"{tau_pc_head[j]:.4f}",
                f"{val_f1_at_tau_macro[j]:.6f}",
                f"{val_f1_at_tau_pc[j]:.6f}",
                f"{test_f1_at_tau_macro[j]:.6f}",
                f"{test_f1_at_tau_pc[j]:.6f}",
                f"{test_f1_at_tau_pc[j] - test_f1_at_tau_macro[j]:+.6f}",
            ])
    print(f"[write] {table_path}")

    print("\n=== Summary (test set, MIMIC-III) ===")
    print(f"  top-50 macro-F1:  A={test_A_top50['f1_macro']:.4f}  B={test_B_top50['f1_macro']:.4f}  "
          f"Δ={summary['delta_f1_macro_top50']:+.4f}")
    print(f"  overall macro-F1: A={test_A['f1_macro']:.4f}  B={test_B['f1_macro']:.4f}  "
          f"Δ={summary['delta_f1_macro_overall']:+.4f}")
    print(f"  overall micro-F1: A={test_A['f1_micro']:.4f}  B={test_B['f1_micro']:.4f}  "
          f"Δ={summary['delta_f1_micro_overall']:+.4f}")

    n_better = int((test_f1_at_tau_pc > test_f1_at_tau_macro).sum())
    n_worse = int((test_f1_at_tau_pc < test_f1_at_tau_macro).sum())
    print(f"  per-code test F1: {n_better}/{len(resolved)} better, {n_worse}/{len(resolved)} worse at tau_pc vs tau_macro")


if __name__ == "__main__":
    main()
