"""Admission-level bootstrap CIs and paired bootstrap tests for ICD-coding F1.

Resampling unit is admission_id (matches paper §4.2). A single
*per-admission TP/FP/FN index* table is built once per (predictions, gold,
vocab) triple; bootstrap iterations only resample admissions and accumulate
counts via `np.bincount`. Paired tests use the *same* resampled-admission
index sequence for the two prediction sets being compared, so the resulting
delta distribution captures only the difference between the two systems
(matched on the same noise draws).

The point estimate from `bootstrap_f1(n=1)` does NOT match the deterministic
point estimate (it's a single bootstrap resample). The deterministic point
estimate is always returned alongside the CI bounds.

Four CI flavours are returned per cell. Macro-F1 over a fixed long-tail
vocabulary is a heavily biased bootstrap statistic (rare classes drop out
of resamples, contributing 0 to a denominator that the deterministic point
includes), so the percentile interval can sit entirely below the point,
the basic/pivot reflection sits entirely above, and **BCa** (bias-corrected
and accelerated) collapses to the upper percentile when the bias exceeds
what the jackknife acceleration can absorb. The **normal** CI
``point ± z_{α/2} · SE_boot`` is therefore the recommended headline: it
brackets the point by construction and uses the bootstrap-estimated noise
level. Percentile, basic, and BCa are retained as diagnostics.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

MacroDenominator = Literal["full", "scorable"]


# --- Per-admission index tables -------------------------------------------


@dataclass
class IndexTable:
    """Per-admission TP/FP/FN class-index lists for one (preds, gold, vocab) triple.

    `tp_idx[i]` is a 1-D int64 array of class indices that the predictions for
    admission *i* got right (true-positive at class level). Likewise FP/FN.
    """

    n_admissions: int
    n_classes: int
    tp_idx: list[np.ndarray]
    fp_idx: list[np.ndarray]
    fn_idx: list[np.ndarray]


def build_index_table(
    pred_lookup: dict[str, set[str]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    vocab: set[str],
) -> IndexTable:
    """Build a per-admission TP/FP/FN class-index table.

    *event_ids* fixes the admission ordering. Codes outside *vocab* are
    discarded from both sides (matches `_score_one_scope` in eval.py).
    """
    class_to_idx = {c: i for i, c in enumerate(sorted(vocab))}
    n_classes = len(class_to_idx)
    tp_idx: list[np.ndarray] = []
    fp_idx: list[np.ndarray] = []
    fn_idx: list[np.ndarray] = []
    for eid in event_ids:
        p = pred_lookup.get(str(eid), set()) & vocab
        g = gold_lookup.get(str(eid), set()) & vocab
        tp = sorted(class_to_idx[c] for c in (p & g))
        fp = sorted(class_to_idx[c] for c in (p - g))
        fn = sorted(class_to_idx[c] for c in (g - p))
        tp_idx.append(np.asarray(tp, dtype=np.int64))
        fp_idx.append(np.asarray(fp, dtype=np.int64))
        fn_idx.append(np.asarray(fn, dtype=np.int64))
    return IndexTable(
        n_admissions=len(event_ids), n_classes=n_classes,
        tp_idx=tp_idx, fp_idx=fp_idx, fn_idx=fn_idx,
    )


# --- Metric computation ---------------------------------------------------


def _f1_from_counts(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
    *,
    macro_denominator: MacroDenominator = "full",
) -> tuple[float, float]:
    """Return (micro_f1, macro_f1) from per-class TP/FP/FN integer arrays.

    *macro_denominator*: ``"full"`` averages per-class F1 over every class in
    the input arrays (legacy behaviour). ``"scorable"`` restricts the mean to
    classes with at least one gold positive — i.e. ``(tp + fn) > 0`` — which
    follows Edin et al. (2023) and is the headline definition for the revised
    paper. Returns NaN when no class has support under ``"scorable"`` (e.g.
    SFT-only on the OOD vocab).
    """
    tp_total = int(tp.sum())
    fp_total = int(fp.sum())
    fn_total = int(fn.sum())
    denom_mi = 2 * tp_total + fp_total + fn_total
    micro = (2 * tp_total / denom_mi) if denom_mi > 0 else 0.0
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    if per_class.size == 0:
        return micro, 0.0
    if macro_denominator == "scorable":
        support = (tp + fn) > 0
        if not support.any():
            return micro, float("nan")
        macro = float(per_class[support].mean())
    else:
        macro = float(per_class.mean())
    return micro, macro


def _accumulate_counts(table: IndexTable, sample_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum per-class TP/FP/FN counts over the admissions in *sample_idx* (with multiplicity)."""
    if sample_idx.size == 0 or table.n_classes == 0:
        z = np.zeros(table.n_classes, dtype=np.int64)
        return z.copy(), z.copy(), z.copy()
    tp_concat = np.concatenate([table.tp_idx[i] for i in sample_idx]) if sample_idx.size else np.array([], dtype=np.int64)
    fp_concat = np.concatenate([table.fp_idx[i] for i in sample_idx]) if sample_idx.size else np.array([], dtype=np.int64)
    fn_concat = np.concatenate([table.fn_idx[i] for i in sample_idx]) if sample_idx.size else np.array([], dtype=np.int64)
    tp = np.bincount(tp_concat, minlength=table.n_classes)
    fp = np.bincount(fp_concat, minlength=table.n_classes)
    fn = np.bincount(fn_concat, minlength=table.n_classes)
    return tp, fp, fn


def deterministic_f1(
    table: IndexTable,
    *,
    macro_denominator: MacroDenominator = "full",
) -> tuple[float, float]:
    """Return (micro, macro) F1 over all admissions (no resampling)."""
    full_idx = np.arange(table.n_admissions, dtype=np.int64)
    tp, fp, fn = _accumulate_counts(table, full_idx)
    return _f1_from_counts(tp, fp, fn, macro_denominator=macro_denominator)


# --- Jackknife and BCa adjustment -----------------------------------------


def _jackknife_f1(
    table: IndexTable,
    *,
    macro_denominator: MacroDenominator = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-admission-out jackknife F1 estimates.

    Returns ``(jack_micro, jack_macro)``, each shape ``(n_admissions,)``. Used
    by `_bca_bounds` to estimate the acceleration constant. Reuses the full
    per-class TP/FP/FN sums and subtracts admission *i*'s contribution rather
    than re-summing the corpus for each LOO sample.
    """
    n = table.n_admissions
    K = table.n_classes
    if n == 0 or K == 0:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    full_idx = np.arange(n, dtype=np.int64)
    tp_full, fp_full, fn_full = _accumulate_counts(table, full_idx)
    jack_mi = np.empty(n, dtype=np.float64)
    jack_ma = np.empty(n, dtype=np.float64)
    for i in range(n):
        tp_i = np.bincount(table.tp_idx[i], minlength=K)
        fp_i = np.bincount(table.fp_idx[i], minlength=K)
        fn_i = np.bincount(table.fn_idx[i], minlength=K)
        jack_mi[i], jack_ma[i] = _f1_from_counts(
            tp_full - tp_i, fp_full - fp_i, fn_full - fn_i,
            macro_denominator=macro_denominator,
        )
    return jack_mi, jack_ma


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via `math.erf`."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (probit) via Acklam's rational approximation.

    Maximum absolute error ~1.15e-9 across (0, 1); avoids the scipy dependency.
    Returns ±inf at the bounds; callers clip *p* away from {0, 1} first.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
           ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def _bca_bounds(
    stats: np.ndarray,
    point: float,
    jack_vals: np.ndarray,
    *,
    alpha: float = 0.025,
    label: str = "",
) -> tuple[float, float]:
    """BCa percentile bounds at level (alpha, 1 - alpha).

    *stats* is the bootstrap distribution; *point* is the deterministic
    statistic; *jack_vals* is the LOO jackknife sequence. Falls back to:
    plain percentile bounds (a = 0, no bias correction) if the jackknife is
    degenerate; bias-corrected percentile (a = 0) if acceleration cannot be
    estimated.
    """
    B = stats.size
    if B == 0:
        return float(point), float(point)
    p_below = float(np.mean(stats < point))
    p_below = min(max(p_below, 1.0 / (B + 1)), B / (B + 1))
    z0 = _norm_ppf(p_below)
    if jack_vals.size == 0:
        a = 0.0
    else:
        diffs = jack_vals.mean() - jack_vals
        sq = float(np.sum(diffs ** 2))
        cu = float(np.sum(diffs ** 3))
        denom = 6.0 * (sq ** 1.5)
        if denom <= 0.0 or not math.isfinite(denom):
            if label:
                warnings.warn(
                    f"BCa acceleration undefined for {label!r}; falling back "
                    f"to bias-corrected percentile (a=0).", stacklevel=2,
                )
            a = 0.0
        else:
            a = cu / denom
    z_lo = _norm_ppf(alpha)
    z_hi = _norm_ppf(1.0 - alpha)

    def _adjusted(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if denom == 0.0 or not math.isfinite(denom):
            return _norm_cdf(z0 + z)
        return _norm_cdf(z0 + (z0 + z) / denom)

    a1 = min(max(_adjusted(z_lo), 0.0), 1.0)
    a2 = min(max(_adjusted(z_hi), 0.0), 1.0)
    lo = float(np.percentile(stats, 100.0 * a1))
    hi = float(np.percentile(stats, 100.0 * a2))
    return lo, hi


# --- Bootstrap APIs -------------------------------------------------------


@dataclass
class BootstrapResult:
    point_micro: float
    point_macro: float
    # Percentile CIs (2.5/97.5 of the bootstrap distribution).
    micro_ci_low: float
    micro_ci_high: float
    macro_ci_low: float
    macro_ci_high: float
    # Basic/pivot CIs (2*point - reversed percentile). For biased statistics
    # like macro-F1 with rare-class dropout, the basic CI does NOT bracket
    # the point either — it is the percentile CI reflected through the point.
    # Retained as a diagnostic; prefer BCa for paper reporting.
    micro_basic_ci_low: float
    micro_basic_ci_high: float
    macro_basic_ci_low: float
    macro_basic_ci_high: float
    # BCa percentile bounds (bias-corrected, jackknife-accelerated). These
    # bracket the point under moderate bias but can collapse to the upper
    # percentile when the bootstrap distribution is severely separated from
    # the point — retained as a diagnostic.
    micro_bca_ci_low: float
    micro_bca_ci_high: float
    macro_bca_ci_low: float
    macro_bca_ci_high: float
    # Normal-approximation CIs `point ± z_{α/2} · SE_boot` using the
    # bootstrap-estimated standard error. Brackets the point by construction
    # and is the recommended headline for biased statistics.
    micro_normal_ci_low: float
    micro_normal_ci_high: float
    macro_normal_ci_low: float
    macro_normal_ci_high: float
    # Bootstrap means — useful diagnostic for spotting bias direction.
    micro_boot_mean: float
    macro_boot_mean: float
    n_resamples: int
    seed: int


def bootstrap_f1_from_table(
    table: IndexTable, *, n: int = 1000, seed: int = 314, label: str = "",
    macro_denominator: MacroDenominator = "full",
) -> BootstrapResult:
    """Bootstrap CIs from a prebuilt IndexTable. Use this when the same
    (preds, gold, vocab) triple feeds into multiple downstream calls — the
    index-table build is the slow step on big corpora.
    """
    point_micro, point_macro = deterministic_f1(table, macro_denominator=macro_denominator)

    rng = np.random.default_rng(seed)
    n_adm = table.n_admissions
    micros = np.empty(n, dtype=np.float64)
    macros = np.empty(n, dtype=np.float64)
    for b in range(n):
        sample_idx = rng.integers(0, n_adm, n_adm)
        tp, fp, fn = _accumulate_counts(table, sample_idx)
        micros[b], macros[b] = _f1_from_counts(
            tp, fp, fn, macro_denominator=macro_denominator,
        )

    mi_lo, mi_hi = float(np.percentile(micros, 2.5)), float(np.percentile(micros, 97.5))
    ma_lo, ma_hi = float(np.percentile(macros, 2.5)), float(np.percentile(macros, 97.5))
    jack_mi, jack_ma = _jackknife_f1(table, macro_denominator=macro_denominator)
    bca_mi_lo, bca_mi_hi = _bca_bounds(
        micros, point_micro, jack_mi, label=f"{label}:micro" if label else "",
    )
    bca_ma_lo, bca_ma_hi = _bca_bounds(
        macros, point_macro, jack_ma, label=f"{label}:macro" if label else "",
    )
    z = 1.959963984540054  # Φ⁻¹(0.975)
    se_mi = float(np.std(micros, ddof=1)) if micros.size > 1 else 0.0
    se_ma = float(np.std(macros, ddof=1)) if macros.size > 1 else 0.0
    return BootstrapResult(
        point_micro=point_micro, point_macro=point_macro,
        micro_ci_low=mi_lo, micro_ci_high=mi_hi,
        macro_ci_low=ma_lo, macro_ci_high=ma_hi,
        micro_basic_ci_low=2 * point_micro - mi_hi,
        micro_basic_ci_high=2 * point_micro - mi_lo,
        macro_basic_ci_low=2 * point_macro - ma_hi,
        macro_basic_ci_high=2 * point_macro - ma_lo,
        micro_bca_ci_low=bca_mi_lo, micro_bca_ci_high=bca_mi_hi,
        macro_bca_ci_low=bca_ma_lo, macro_bca_ci_high=bca_ma_hi,
        micro_normal_ci_low=point_micro - z * se_mi,
        micro_normal_ci_high=point_micro + z * se_mi,
        macro_normal_ci_low=point_macro - z * se_ma,
        macro_normal_ci_high=point_macro + z * se_ma,
        micro_boot_mean=float(micros.mean()),
        macro_boot_mean=float(macros.mean()),
        n_resamples=n, seed=seed,
    )


def bootstrap_f1(
    pred_lookup: dict[str, set[str]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    vocab: set[str],
    *,
    n: int = 1000,
    seed: int = 314,
    macro_denominator: MacroDenominator = "full",
) -> BootstrapResult:
    """Bootstrap CIs for micro and macro F1 of a single prediction set.

    Resamples *event_ids* with replacement *n* times. Returns the deterministic
    point estimate plus 2.5/97.5 percentile bounds on the bootstrap distribution.
    """
    table = build_index_table(pred_lookup, gold_lookup, event_ids, vocab)
    return bootstrap_f1_from_table(
        table, n=n, seed=seed, macro_denominator=macro_denominator,
    )


@dataclass
class PairedBootstrapResult:
    delta_micro: float
    delta_macro: float
    # Percentile CIs on the delta distribution.
    delta_micro_ci_low: float
    delta_micro_ci_high: float
    delta_macro_ci_low: float
    delta_macro_ci_high: float
    # Basic/pivot CIs (diagnostic; reflects percentile through the point delta).
    delta_micro_basic_ci_low: float
    delta_micro_basic_ci_high: float
    delta_macro_basic_ci_low: float
    delta_macro_basic_ci_high: float
    # BCa-adjusted CIs on the delta distribution. May collapse under severe
    # bias; retained as a diagnostic.
    delta_micro_bca_ci_low: float
    delta_micro_bca_ci_high: float
    delta_macro_bca_ci_low: float
    delta_macro_bca_ci_high: float
    # Normal-approximation CIs on the delta. Brackets the point delta by
    # construction; recommended headline.
    delta_micro_normal_ci_low: float
    delta_micro_normal_ci_high: float
    delta_macro_normal_ci_low: float
    delta_macro_normal_ci_high: float
    # Bootstrap means of the delta distribution — diagnostic for bias direction.
    delta_micro_boot_mean: float
    delta_macro_boot_mean: float
    p_two_sided_micro: float
    p_two_sided_macro: float
    n_resamples: int
    seed: int


# --- Per-admission decomposition table (P/R/F1 + EMR + codes-per-note + top-k) -----


@dataclass
class DecompTable:
    """Per-admission decomposition for one (system, scope) cell.

    Carries the same per-admission TP/FP/FN class-index structure as
    `IndexTable` (so `_accumulate_counts` works directly) plus per-admission
    scalars needed for EMR, codes-per-note, and ranking-based metrics.

    `correct_at_*` arrays are populated only when raw classifier scores are
    available (SFT-only systems); pipeline systems (band_verify, hybrid)
    leave them at None and the bootstrap routine skips top-k columns.
    """

    n_admissions: int
    n_classes: int
    tp_idx: list[np.ndarray]
    fp_idx: list[np.ndarray]
    fn_idx: list[np.ndarray]
    n_pred: np.ndarray
    n_gold: np.ndarray
    exact_match: np.ndarray
    correct_at_8: np.ndarray | None
    correct_at_15: np.ndarray | None
    correct_at_R: np.ndarray | None


def _pr_f1_from_counts(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
    *,
    macro_denominator: MacroDenominator = "full",
) -> tuple[float, float, float, float, float, float]:
    """Return ``(mi_p, mi_r, mi_f, ma_p, ma_r, ma_f)`` from per-class arrays.

    Macro variants are unweighted means over `tp.size` classes (matching
    `_score_one_scope` in eval.py — class-by-class P/R/F1 averaged with
    zeros for unobserved codes) when ``macro_denominator="full"``. Under
    ``"scorable"`` the mean is restricted to classes with at least one gold
    positive — ``(tp + fn) > 0`` — and returns NaN macros when no class has
    support.
    """
    tp_total = int(tp.sum())
    fp_total = int(fp.sum())
    fn_total = int(fn.sum())
    mi_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    mi_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    mi_f = (2 * mi_p * mi_r / (mi_p + mi_r)) if (mi_p + mi_r) > 0 else 0.0
    if tp.size == 0:
        return mi_p, mi_r, mi_f, 0.0, 0.0, 0.0
    denom_p = tp + fp
    denom_r = tp + fn
    denom_f = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        per_p = np.where(denom_p > 0, tp / np.maximum(denom_p, 1), 0.0)
        per_r = np.where(denom_r > 0, tp / np.maximum(denom_r, 1), 0.0)
        per_f = np.where(denom_f > 0, 2 * tp / np.maximum(denom_f, 1), 0.0)
    if macro_denominator == "scorable":
        support = (tp + fn) > 0
        if not support.any():
            nan = float("nan")
            return mi_p, mi_r, mi_f, nan, nan, nan
        per_p = per_p[support]
        per_r = per_r[support]
        per_f = per_f[support]
    return mi_p, mi_r, mi_f, float(per_p.mean()), float(per_r.mean()), float(per_f.mean())


_DECOMP_BASE_METRICS = (
    "precision_micro", "recall_micro", "f1_micro",
    "precision_macro", "recall_macro", "f1_macro",
    "emr", "codes_per_note",
)
_DECOMP_RANKING_METRICS = (
    "p_at_8", "p_at_15", "r_at_8", "r_at_15", "r_precision",
)


def _decomp_metric_names(table: DecompTable) -> tuple[str, ...]:
    if table.correct_at_8 is not None:
        return _DECOMP_BASE_METRICS + _DECOMP_RANKING_METRICS
    return _DECOMP_BASE_METRICS


def deterministic_decomp(
    table: DecompTable,
    *,
    macro_denominator: MacroDenominator = "full",
) -> dict[str, float]:
    """Compute every decomposition metric on the full (un-resampled) corpus."""
    full_idx = np.arange(table.n_admissions, dtype=np.int64)
    tp, fp, fn = _accumulate_counts(table, full_idx)
    mi_p, mi_r, mi_f, ma_p, ma_r, ma_f = _pr_f1_from_counts(
        tp, fp, fn, macro_denominator=macro_denominator,
    )
    out: dict[str, float] = {
        "precision_micro": mi_p, "recall_micro": mi_r, "f1_micro": mi_f,
        "precision_macro": ma_p, "recall_macro": ma_r, "f1_macro": ma_f,
        "emr": float(table.exact_match.mean()) if table.n_admissions > 0 else 0.0,
        "codes_per_note": float(table.n_pred.mean()) if table.n_admissions > 0 else 0.0,
    }
    if table.correct_at_8 is not None:
        n = table.n_admissions
        denom = np.maximum(table.n_gold.astype(np.float64), 1.0)
        out["p_at_8"] = float((table.correct_at_8 / 8.0).mean()) if n > 0 else 0.0
        out["p_at_15"] = float((table.correct_at_15 / 15.0).mean()) if n > 0 else 0.0
        out["r_at_8"] = float((table.correct_at_8.astype(np.float64) / denom).mean()) if n > 0 else 0.0
        out["r_at_15"] = float((table.correct_at_15.astype(np.float64) / denom).mean()) if n > 0 else 0.0
        out["r_precision"] = float((table.correct_at_R.astype(np.float64) / denom).mean()) if n > 0 else 0.0
    return out


@dataclass
class BootstrapDecompResult:
    """Bootstrap decomposition for one (system, scope) cell.

    Each metric carries its deterministic point estimate, the bootstrap
    mean (diagnostic for bias), and the normal-approximation 95% CI
    ``point ± 1.96 · SE_boot``. Matches the headline interval in
    `tab:main` (see paper §4.2 / [revision/bootstrap.py:290](revision/bootstrap.py#L290)).
    """

    point: dict[str, float]
    boot_mean: dict[str, float]
    normal_ci_lo: dict[str, float]
    normal_ci_hi: dict[str, float]
    n_resamples: int
    seed: int


def bootstrap_decomp_metrics(
    table: DecompTable, *, n: int = 1000, seed: int = 314,
    macro_denominator: MacroDenominator = "full",
) -> BootstrapDecompResult:
    """Admission-level bootstrap CIs for every metric in `_decomp_metric_names`.

    Mirrors `bootstrap_f1_from_table` but accumulates more statistics per
    iteration. Reuses `_accumulate_counts` for the per-class TP/FP/FN sums.
    """
    metric_names = _decomp_metric_names(table)
    point = deterministic_decomp(table, macro_denominator=macro_denominator)
    n_adm = table.n_admissions
    M = len(metric_names)
    samples = np.empty((n, M), dtype=np.float64)
    has_topk = table.correct_at_8 is not None
    rng = np.random.default_rng(seed)
    for b in range(n):
        sample_idx = rng.integers(0, n_adm, n_adm)
        tp, fp, fn = _accumulate_counts(table, sample_idx)
        mi_p, mi_r, mi_f, ma_p, ma_r, ma_f = _pr_f1_from_counts(
            tp, fp, fn, macro_denominator=macro_denominator,
        )
        em = float(table.exact_match[sample_idx].mean()) if n_adm > 0 else 0.0
        cpn = float(table.n_pred[sample_idx].mean()) if n_adm > 0 else 0.0
        samples[b, 0] = mi_p
        samples[b, 1] = mi_r
        samples[b, 2] = mi_f
        samples[b, 3] = ma_p
        samples[b, 4] = ma_r
        samples[b, 5] = ma_f
        samples[b, 6] = em
        samples[b, 7] = cpn
        if has_topk:
            denom = np.maximum(
                table.n_gold[sample_idx].astype(np.float64), 1.0
            )
            c8 = table.correct_at_8[sample_idx].astype(np.float64)
            c15 = table.correct_at_15[sample_idx].astype(np.float64)
            cR = table.correct_at_R[sample_idx].astype(np.float64)
            samples[b, 8] = float((c8 / 8.0).mean())
            samples[b, 9] = float((c15 / 15.0).mean())
            samples[b, 10] = float((c8 / denom).mean())
            samples[b, 11] = float((c15 / denom).mean())
            samples[b, 12] = float((cR / denom).mean())

    z = 1.959963984540054  # Φ⁻¹(0.975)
    boot_mean: dict[str, float] = {}
    normal_lo: dict[str, float] = {}
    normal_hi: dict[str, float] = {}
    for j, name in enumerate(metric_names):
        col = samples[:, j]
        boot_mean[name] = float(col.mean())
        se = float(col.std(ddof=1)) if col.size > 1 else 0.0
        normal_lo[name] = point[name] - z * se
        normal_hi[name] = point[name] + z * se
    return BootstrapDecompResult(
        point=point, boot_mean=boot_mean,
        normal_ci_lo=normal_lo, normal_ci_hi=normal_hi,
        n_resamples=n, seed=seed,
    )


def paired_bootstrap_test_from_tables(
    table_A: IndexTable, table_B: IndexTable, *, n: int = 1000, seed: int = 314,
    label: str = "",
    macro_denominator: MacroDenominator = "full",
) -> PairedBootstrapResult:
    """Paired bootstrap test from prebuilt IndexTables (same admissions / vocab).

    Both tables must have been built with the same *event_ids* ordering and
    *vocab* — this is the caller's responsibility.
    """
    assert table_A.n_admissions == table_B.n_admissions
    assert table_A.n_classes == table_B.n_classes
    return _paired_bootstrap_core(
        table_A, table_B, n=n, seed=seed, label=label,
        macro_denominator=macro_denominator,
    )


def paired_bootstrap_test(
    pred_A_lookup: dict[str, set[str]],
    pred_B_lookup: dict[str, set[str]],
    gold_lookup: dict[str, set[str]],
    event_ids: list[str],
    vocab: set[str],
    *,
    n: int = 1000,
    seed: int = 314,
    macro_denominator: MacroDenominator = "full",
) -> PairedBootstrapResult:
    """Paired bootstrap test for delta = F1(A) - F1(B).

    Each bootstrap iteration draws *one* admission resample and applies it to
    both A and B, so the delta distribution isolates the system difference
    from sampling noise. The two-sided p-value is the bootstrap-percentile
    p-value: 2 × min(P(delta ≤ 0), P(delta ≥ 0)).
    """
    table_A = build_index_table(pred_A_lookup, gold_lookup, event_ids, vocab)
    table_B = build_index_table(pred_B_lookup, gold_lookup, event_ids, vocab)
    return _paired_bootstrap_core(
        table_A, table_B, n=n, seed=seed,
        macro_denominator=macro_denominator,
    )


def _paired_bootstrap_core(
    table_A: IndexTable, table_B: IndexTable, *, n: int, seed: int, label: str = "",
    macro_denominator: MacroDenominator = "full",
) -> PairedBootstrapResult:
    assert table_A.n_admissions == table_B.n_admissions
    assert table_A.n_classes == table_B.n_classes

    point_A_mi, point_A_ma = deterministic_f1(table_A, macro_denominator=macro_denominator)
    point_B_mi, point_B_ma = deterministic_f1(table_B, macro_denominator=macro_denominator)
    delta_mi = point_A_mi - point_B_mi
    delta_ma = point_A_ma - point_B_ma

    rng = np.random.default_rng(seed)
    n_adm = table_A.n_admissions
    deltas_mi = np.empty(n, dtype=np.float64)
    deltas_ma = np.empty(n, dtype=np.float64)
    for b in range(n):
        sample_idx = rng.integers(0, n_adm, n_adm)
        tp_a, fp_a, fn_a = _accumulate_counts(table_A, sample_idx)
        tp_b, fp_b, fn_b = _accumulate_counts(table_B, sample_idx)
        a_mi, a_ma = _f1_from_counts(tp_a, fp_a, fn_a, macro_denominator=macro_denominator)
        b_mi, b_ma = _f1_from_counts(tp_b, fp_b, fn_b, macro_denominator=macro_denominator)
        deltas_mi[b] = a_mi - b_mi
        deltas_ma[b] = a_ma - b_ma

    def _two_sided_p(deltas: np.ndarray) -> float:
        # Bootstrap-percentile p-value against the null delta = 0.
        p_le = float(np.mean(deltas <= 0.0))
        p_ge = float(np.mean(deltas >= 0.0))
        return min(1.0, 2.0 * min(p_le, p_ge))

    mi_lo, mi_hi = float(np.percentile(deltas_mi, 2.5)), float(np.percentile(deltas_mi, 97.5))
    ma_lo, ma_hi = float(np.percentile(deltas_ma, 2.5)), float(np.percentile(deltas_ma, 97.5))
    # Jackknife the per-admission delta for BCa acceleration.
    jack_a_mi, jack_a_ma = _jackknife_f1(table_A, macro_denominator=macro_denominator)
    jack_b_mi, jack_b_ma = _jackknife_f1(table_B, macro_denominator=macro_denominator)
    jack_delta_mi = jack_a_mi - jack_b_mi
    jack_delta_ma = jack_a_ma - jack_b_ma
    bca_mi_lo, bca_mi_hi = _bca_bounds(
        deltas_mi, delta_mi, jack_delta_mi,
        label=f"{label}:delta_micro" if label else "",
    )
    bca_ma_lo, bca_ma_hi = _bca_bounds(
        deltas_ma, delta_ma, jack_delta_ma,
        label=f"{label}:delta_macro" if label else "",
    )
    z = 1.959963984540054
    se_dmi = float(np.std(deltas_mi, ddof=1)) if deltas_mi.size > 1 else 0.0
    se_dma = float(np.std(deltas_ma, ddof=1)) if deltas_ma.size > 1 else 0.0
    return PairedBootstrapResult(
        delta_micro=delta_mi, delta_macro=delta_ma,
        delta_micro_ci_low=mi_lo, delta_micro_ci_high=mi_hi,
        delta_macro_ci_low=ma_lo, delta_macro_ci_high=ma_hi,
        delta_micro_basic_ci_low=2 * delta_mi - mi_hi,
        delta_micro_basic_ci_high=2 * delta_mi - mi_lo,
        delta_macro_basic_ci_low=2 * delta_ma - ma_hi,
        delta_macro_basic_ci_high=2 * delta_ma - ma_lo,
        delta_micro_bca_ci_low=bca_mi_lo, delta_micro_bca_ci_high=bca_mi_hi,
        delta_macro_bca_ci_low=bca_ma_lo, delta_macro_bca_ci_high=bca_ma_hi,
        delta_micro_normal_ci_low=delta_mi - z * se_dmi,
        delta_micro_normal_ci_high=delta_mi + z * se_dmi,
        delta_macro_normal_ci_low=delta_ma - z * se_dma,
        delta_macro_normal_ci_high=delta_ma + z * se_dma,
        delta_micro_boot_mean=float(deltas_mi.mean()),
        delta_macro_boot_mean=float(deltas_ma.mean()),
        p_two_sided_micro=_two_sided_p(deltas_mi),
        p_two_sided_macro=_two_sided_p(deltas_ma),
        n_resamples=n, seed=seed,
    )
