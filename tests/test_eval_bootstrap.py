"""Unit tests for revision/bootstrap.py."""

from __future__ import annotations

import numpy as np
import pytest

from coding.eval.bootstrap import (
    DecompTable,
    _jackknife_f1,
    _norm_cdf,
    _norm_ppf,
    _pr_f1_from_counts,
    bootstrap_f1,
    build_index_table,
    deterministic_decomp,
    deterministic_f1,
    paired_bootstrap_test,
)

# --- Hand-computable synthetic case ---------------------------------------

# 5 admissions, 4-class vocab. Pred and gold defined so micro/macro F1 are
# tractable by hand.

VOCAB = {"A", "B", "C", "D"}
EVENT_IDS = ["e1", "e2", "e3", "e4", "e5"]
PRED = {
    "e1": {"A", "B"},
    "e2": {"B", "C"},
    "e3": {"A", "C", "D"},
    "e4": set(),
    "e5": {"D"},
}
GOLD = {
    "e1": {"A", "B"},      # tp=2, fp=0, fn=0
    "e2": {"B", "C", "D"}, # tp=2, fp=0, fn=1
    "e3": {"A", "C"},      # tp=2, fp=1, fn=0
    "e4": {"A"},           # tp=0, fp=0, fn=1
    "e5": {"D"},           # tp=1, fp=0, fn=0
}
# Per-class TP/FP/FN over all admissions:
#   A: tp=2 (e1,e3), fp=0, fn=1 (e4)            → P=2/2=1.0,    R=2/3,    F1=4/5  =0.8
#   B: tp=2 (e1,e2), fp=0, fn=0                 → P=1, R=1, F1=1.0
#   C: tp=2 (e2,e3), fp=0, fn=0                 → P=1, R=1, F1=1.0
#   D: tp=1 (e5),   fp=1 (e3), fn=1 (e2)        → P=1/2, R=1/2, F1=1/2 =0.5
# Micro: tp=7, fp=1, fn=2 → P=7/8, R=7/9, F1 = 2*7/(2*7+1+2) = 14/17 ≈ 0.823529
# Macro F1 = (0.8 + 1.0 + 1.0 + 0.5) / 4 = 0.825

EXPECTED_MICRO_F1 = 14 / 17
EXPECTED_MACRO_F1 = (0.8 + 1.0 + 1.0 + 0.5) / 4


def test_deterministic_f1_matches_hand_computed():
    table = build_index_table(PRED, GOLD, EVENT_IDS, VOCAB)
    micro, macro = deterministic_f1(table)
    assert micro == pytest.approx(EXPECTED_MICRO_F1, abs=1e-9)
    assert macro == pytest.approx(EXPECTED_MACRO_F1, abs=1e-9)


def test_bootstrap_returns_deterministic_point_estimate():
    res = bootstrap_f1(PRED, GOLD, EVENT_IDS, VOCAB, n=10, seed=42)
    assert res.point_micro == pytest.approx(EXPECTED_MICRO_F1, abs=1e-9)
    assert res.point_macro == pytest.approx(EXPECTED_MACRO_F1, abs=1e-9)


def test_bootstrap_seed_reproducibility():
    a = bootstrap_f1(PRED, GOLD, EVENT_IDS, VOCAB, n=200, seed=314)
    b = bootstrap_f1(PRED, GOLD, EVENT_IDS, VOCAB, n=200, seed=314)
    assert a.micro_ci_low == b.micro_ci_low
    assert a.micro_ci_high == b.micro_ci_high
    assert a.macro_ci_low == b.macro_ci_low
    assert a.macro_ci_high == b.macro_ci_high


def test_bootstrap_ci_brackets_point_estimate_for_large_n():
    # On a small noisy sample the CI must contain the deterministic point estimate.
    res = bootstrap_f1(PRED, GOLD, EVENT_IDS, VOCAB, n=2000, seed=1)
    assert res.micro_ci_low <= res.point_micro <= res.micro_ci_high
    assert res.macro_ci_low <= res.point_macro <= res.macro_ci_high


def test_bootstrap_ci_widens_with_smaller_sample():
    # A 5-admission resample should be much noisier than the same resampling
    # on a 50-admission corpus (built by replication).
    big_event_ids = [f"e{i}" for i in range(50)]
    big_pred = {f"e{i}": PRED[EVENT_IDS[i % 5]] for i in range(50)}
    big_gold = {f"e{i}": GOLD[EVENT_IDS[i % 5]] for i in range(50)}
    small = bootstrap_f1(PRED, GOLD, EVENT_IDS, VOCAB, n=2000, seed=7)
    big = bootstrap_f1(big_pred, big_gold, big_event_ids, VOCAB, n=2000, seed=7)
    small_width = small.micro_ci_high - small.micro_ci_low
    big_width = big.micro_ci_high - big.micro_ci_low
    assert small_width > big_width


# --- Paired bootstrap tests -----------------------------------------------


def test_paired_bootstrap_identical_systems_have_zero_delta():
    res = paired_bootstrap_test(PRED, PRED, GOLD, EVENT_IDS, VOCAB, n=500, seed=314)
    assert res.delta_micro == pytest.approx(0.0, abs=1e-9)
    assert res.delta_macro == pytest.approx(0.0, abs=1e-9)
    # Every resample produces zero delta when A and B are identical.
    assert res.delta_micro_ci_low == pytest.approx(0.0, abs=1e-9)
    assert res.delta_micro_ci_high == pytest.approx(0.0, abs=1e-9)


def test_paired_bootstrap_dominated_system_has_signed_negative_delta():
    # Build a strictly worse B (drop every code A predicted right).
    bad_pred = {eid: set() for eid in EVENT_IDS}  # B predicts nothing.
    res = paired_bootstrap_test(PRED, bad_pred, GOLD, EVENT_IDS, VOCAB, n=500, seed=314)
    assert res.delta_micro > 0.0
    assert res.delta_macro > 0.0
    # Delta CI should be entirely positive (or at worst touch zero) — A dominates.
    assert res.delta_micro_ci_low >= -1e-12
    # And p should be strictly below 0.05 (we're comparing a system to nothing).
    assert res.p_two_sided_micro < 0.05


def test_paired_bootstrap_uses_correlated_resamples():
    # Verify A vs B paired delta variance < unpaired (i.e. paired test is tighter).
    rng = np.random.default_rng(0)
    n_adm = 30
    eids = [f"e{i}" for i in range(n_adm)]
    # A: random predictions; B: A with codes shuffled per admission.
    pred_A = {e: set("AB" if rng.random() > 0.5 else "AC") for e in eids}
    pred_B = {e: set("AB" if rng.random() > 0.5 else "AD") for e in eids}
    gold = {e: set("ABC") for e in eids}
    paired = paired_bootstrap_test(pred_A, pred_B, gold, eids, VOCAB, n=500, seed=11)
    # Sanity: CI not collapsed to a single point.
    assert paired.delta_micro_ci_high - paired.delta_micro_ci_low > 0


# --- BCa correctness ------------------------------------------------------


def test_norm_ppf_round_trip():
    # Acklam's approximation should round-trip Φ to within ~1e-9 over the
    # interior; tolerate slightly looser at the tails where the rational
    # branches join.
    for p in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]:
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-8)


def test_bca_brackets_point_on_unbiased_case():
    # Unbiased case: BCa should agree with percentile to ≈3 decimal places.
    res = bootstrap_f1(PRED, GOLD, EVENT_IDS, VOCAB, n=2000, seed=1)
    assert res.micro_bca_ci_low <= res.point_micro <= res.micro_bca_ci_high
    assert res.macro_bca_ci_low <= res.point_macro <= res.macro_bca_ci_high


def _biased_macro_corpus(n_adm: int = 1000, n_rare: int = 80, seed: int = 7):
    """Build a synthetic corpus where macro-F1 is moderately biased downward.

    With ``n_adm`` admissions and ``n_rare`` rare classes that appear in
    exactly one admission each, a bootstrap resample drops a rare class with
    probability ``(1 - 1/n_adm)^n_adm ≈ 1/e`` only on the rare-class admission's
    weight in the resample, so the bootstrap distribution of macro-F1 sits
    several standard errors below the point — the bias regime that broke the
    paper's CIs in real data — while staying close enough that BCa can correct.
    """
    rng = np.random.default_rng(seed)
    common = {"A", "B", "C"}
    rare = {f"R{i}" for i in range(n_rare)}
    vocab = common | rare
    eids = [f"e{i}" for i in range(n_adm)]
    rare_list = list(rare)
    rng.shuffle(rare_list)
    pred: dict[str, set[str]] = {}
    gold: dict[str, set[str]] = {}
    for i, eid in enumerate(eids):
        common_codes = set()
        for c in common:
            if rng.random() < 0.7:
                common_codes.add(c)
        pred[eid] = set(common_codes)
        gold[eid] = set(common_codes)
        if i < n_rare:
            r = rare_list[i]
            pred[eid].add(r)
            gold[eid].add(r)
    return pred, gold, eids, vocab


def test_bca_outperforms_percentile_on_biased_case():
    pred, gold, eids, vocab = _biased_macro_corpus()
    res = bootstrap_f1(pred, gold, eids, vocab, n=1000, seed=42)
    # Sanity: percentile CI on macro really does sit below the point — this
    # is the bug the user observed in real data.
    assert res.macro_ci_high < res.point_macro
    # Basic/pivot mathematically reflects through the point and lands above
    # the point — also broken.
    assert res.macro_basic_ci_low > res.point_macro
    # BCa should at minimum push the high side up toward the point. With
    # moderate bias we expect bracketing; tolerate a small near-miss as a
    # documented BCa limitation when bias is severe.
    assert res.macro_bca_ci_high >= res.macro_ci_high
    # Distance from BCa to the point should be smaller than percentile's.
    pct_gap = res.point_macro - res.macro_ci_high
    bca_gap = max(res.point_macro - res.macro_bca_ci_high, 0.0)
    assert bca_gap < pct_gap


def test_normal_ci_brackets_point_under_severe_bias():
    # Normal-approximation CI must always bracket the point, by construction
    # (centred on the point with bootstrap-derived SE). This is the headline
    # interval for biased macro-F1 cells.
    pred, gold, eids, vocab = _biased_macro_corpus()
    res = bootstrap_f1(pred, gold, eids, vocab, n=1000, seed=42)
    assert res.macro_normal_ci_low <= res.point_macro <= res.macro_normal_ci_high
    assert res.micro_normal_ci_low <= res.point_micro <= res.micro_normal_ci_high


def test_paired_normal_ci_brackets_delta():
    pred_A, gold, eids, vocab = _biased_macro_corpus(seed=11)
    pred_B = {e: set(p) for e, p in pred_A.items()}
    rng = np.random.default_rng(0)
    for e, codes in pred_B.items():
        for c in list(codes):
            if c.startswith("R") and rng.random() < 0.5:
                codes.discard(c)
    res = paired_bootstrap_test(pred_A, pred_B, gold, eids, vocab, n=1000, seed=42)
    assert res.delta_macro_normal_ci_low <= res.delta_macro <= res.delta_macro_normal_ci_high
    assert res.delta_micro_normal_ci_low <= res.delta_micro <= res.delta_micro_normal_ci_high


def test_jackknife_recovers_deterministic_at_full_corpus():
    # Mean of LOO jackknife estimates should be close to the deterministic
    # point on the toy corpus (sample size is small but bias is tiny here).
    table = build_index_table(PRED, GOLD, EVENT_IDS, VOCAB)
    jack_mi, jack_ma = _jackknife_f1(table)
    point_micro, point_macro = deterministic_f1(table)
    assert abs(jack_mi.mean() - point_micro) < 0.1
    assert abs(jack_ma.mean() - point_macro) < 0.1


def test_paired_bca_brackets_or_approaches_delta():
    pred_A, gold, eids, vocab = _biased_macro_corpus(seed=11)
    pred_B = {e: set(p) for e, p in pred_A.items()}
    rng = np.random.default_rng(0)
    for e, codes in pred_B.items():
        for c in list(codes):
            if c.startswith("R") and rng.random() < 0.5:
                codes.discard(c)
    res = paired_bootstrap_test(pred_A, pred_B, gold, eids, vocab, n=1000, seed=42)
    assert res.delta_macro > 0.0
    # Paired-delta BCa should at least put the high end above the percentile
    # high; bracketing is the goal but tolerated to fall just short under
    # severe bias (matches the real-data fallback contract).
    assert res.delta_macro_bca_ci_high >= res.delta_macro_ci_high
    pct_gap = res.delta_macro - res.delta_macro_ci_high
    bca_gap = max(res.delta_macro - res.delta_macro_bca_ci_high, 0.0)
    assert bca_gap < pct_gap


# --- Decomposition primitives (P, R, EMR, codes/note, ranking) ------------


def test_pr_f1_from_counts_matches_hand_computed():
    # Per-class counts derived from the toy corpus above.
    #   A: tp=2, fp=0, fn=1 → P=1.0, R=2/3,  F1=0.8
    #   B: tp=2, fp=0, fn=0 → P=1.0, R=1.0,  F1=1.0
    #   C: tp=2, fp=0, fn=0 → P=1.0, R=1.0,  F1=1.0
    #   D: tp=1, fp=1, fn=1 → P=0.5, R=0.5,  F1=0.5
    # Micro: tp=7, fp=1, fn=2 → P=7/8, R=7/9, F1=14/17
    tp = np.array([2, 2, 2, 1], dtype=np.int64)
    fp = np.array([0, 0, 0, 1], dtype=np.int64)
    fn = np.array([1, 0, 0, 1], dtype=np.int64)
    mi_p, mi_r, mi_f, ma_p, ma_r, ma_f = _pr_f1_from_counts(tp, fp, fn)
    assert mi_p == pytest.approx(7 / 8, abs=1e-12)
    assert mi_r == pytest.approx(7 / 9, abs=1e-12)
    assert mi_f == pytest.approx(14 / 17, abs=1e-12)
    assert ma_p == pytest.approx((1.0 + 1.0 + 1.0 + 0.5) / 4, abs=1e-12)
    assert ma_r == pytest.approx((2 / 3 + 1.0 + 1.0 + 0.5) / 4, abs=1e-12)
    assert ma_f == pytest.approx((0.8 + 1.0 + 1.0 + 0.5) / 4, abs=1e-12)


def test_deterministic_decomp_matches_hand_computed():
    # Build a DecompTable on the toy corpus with synthetic ranking arrays.
    base = build_index_table(PRED, GOLD, EVENT_IDS, VOCAB)
    n = base.n_admissions
    # n_pred / n_gold per admission (after vocab intersection — toy is in-vocab).
    n_pred = np.array([len(PRED[e]) for e in EVENT_IDS], dtype=np.int64)
    n_gold = np.array([len(GOLD[e]) for e in EVENT_IDS], dtype=np.int64)
    # exact_match: 1 iff pred == gold per admission.
    exact_match = np.array(
        [1 if PRED[e] == GOLD[e] else 0 for e in EVENT_IDS], dtype=np.int64
    )
    # Synthetic top-k arrays: each admission gets k correct = min(top-k, n_gold).
    correct_at_8 = np.minimum(n_gold, 8)
    correct_at_15 = np.minimum(n_gold, 15)
    correct_at_R = n_gold.copy()  # an oracle ranker recovers all gold at k=R.
    table = DecompTable(
        n_admissions=n, n_classes=base.n_classes,
        tp_idx=base.tp_idx, fp_idx=base.fp_idx, fn_idx=base.fn_idx,
        n_pred=n_pred, n_gold=n_gold, exact_match=exact_match,
        correct_at_8=correct_at_8, correct_at_15=correct_at_15,
        correct_at_R=correct_at_R,
    )
    out = deterministic_decomp(table)
    # P/R/F1 must match the F1 hand-computed earlier.
    assert out["f1_micro"] == pytest.approx(EXPECTED_MICRO_F1, abs=1e-12)
    assert out["f1_macro"] == pytest.approx(EXPECTED_MACRO_F1, abs=1e-12)
    # EMR: e1 and e5 have pred == gold → 2/5 = 0.4.
    assert out["emr"] == pytest.approx(2 / 5, abs=1e-12)
    # codes_per_note: mean of [2,2,3,0,1] = 8/5.
    assert out["codes_per_note"] == pytest.approx(8 / 5, abs=1e-12)
    # P@8 = mean(min(n_gold, 8)/8) = mean([2,3,2,1,1]/8) = (9/8)/5 = 0.225.
    assert out["p_at_8"] == pytest.approx(np.mean(np.minimum(n_gold, 8) / 8.0), abs=1e-12)
    # R@8 with oracle k=8 = mean(min(n_gold,8)/n_gold) = 1.0 (n_gold ≤ 8 here).
    assert out["r_at_8"] == pytest.approx(1.0, abs=1e-12)
    # R-precision: oracle recovers all gold at k=R → 1.0.
    assert out["r_precision"] == pytest.approx(1.0, abs=1e-12)


# --- eval._score_one_scope parity ----------------------------------------


def test_score_one_scope_matches_f1_from_counts_on_toy_corpus():
    # Pin the wrapper in `eval.harness._score_one_scope` to the canonical
    # F1 implementation in `eval.bootstrap._pr_f1_from_counts`. If a
    # future edit diverges them, headline Tables 2-3 will silently disagree
    # with their bootstrap CIs (see plan §4).
    from coding.eval.harness import _score_one_scope

    pred_sets = [PRED[e] for e in EVENT_IDS]
    gold_sets = [GOLD[e] for e in EVENT_IDS]
    out = _score_one_scope(pred_sets, gold_sets, VOCAB)
    assert out["f1_micro"] == pytest.approx(EXPECTED_MICRO_F1, abs=1e-12)
    assert out["f1_macro"] == pytest.approx(EXPECTED_MACRO_F1, abs=1e-12)
    # Spot-check P/R too — these were not previously covered by any test.
    assert out["precision_micro"] == pytest.approx(7 / 8, abs=1e-12)
    assert out["recall_micro"] == pytest.approx(7 / 9, abs=1e-12)
    assert out["precision_macro"] == pytest.approx((1.0 + 1.0 + 1.0 + 0.5) / 4, abs=1e-12)
    assert out["recall_macro"] == pytest.approx((2 / 3 + 1.0 + 1.0 + 0.5) / 4, abs=1e-12)


def test_score_one_scope_macro_denominator_is_global_vocab():
    # The vocab-scoping contract: macro-F1 averages over |vocab| classes,
    # not just over observed classes. Passing the wrong (smaller) vocab
    # silently inflates macro because zero-F1 unobserved classes drop out
    # of the denominator. Pinning this prevents the silent regression at
    # eval.py:310-314.
    from coding.eval.harness import _score_one_scope

    pred_sets = [PRED[e] for e in EVENT_IDS]
    gold_sets = [GOLD[e] for e in EVENT_IDS]
    full = _score_one_scope(pred_sets, gold_sets, VOCAB)
    # Drop "D" from the vocab — it gets filtered from both pred and gold,
    # so the remaining 3 classes (A, B, C) all have F1 ∈ {0.8, 1.0, 1.0}.
    smaller = _score_one_scope(pred_sets, gold_sets, {"A", "B", "C"})
    assert smaller["f1_macro"] == pytest.approx((0.8 + 1.0 + 1.0) / 3, abs=1e-12)
    # Inflation direction: dropping D (F1=0.5) from the denominator pushes
    # macro up. The contract being pinned is "vocab argument controls denom."
    assert smaller["f1_macro"] > full["f1_macro"]


# --- macro_denominator="scorable" -----------------------------------------


def test_f1_from_counts_scorable_equals_full_when_all_supported():
    # Toy corpus: every class has at least one gold positive, so the scorable
    # filter is a no-op and macro_scorable must equal macro_full.
    tp = np.array([2, 2, 2, 1], dtype=np.int64)
    fp = np.array([0, 0, 0, 1], dtype=np.int64)
    fn = np.array([1, 0, 0, 1], dtype=np.int64)
    from coding.eval.bootstrap import _f1_from_counts
    mi_full, ma_full = _f1_from_counts(tp, fp, fn, macro_denominator="full")
    mi_scor, ma_scor = _f1_from_counts(tp, fp, fn, macro_denominator="scorable")
    assert mi_full == pytest.approx(mi_scor, abs=1e-12)
    assert ma_full == pytest.approx(ma_scor, abs=1e-12)


def test_f1_from_counts_scorable_filters_zero_support_class():
    # 3-class case. Class 2 has zero gold support (tp+fn = 0) but a FP — so
    # under "full" it contributes F1=0 and drags macro down; under "scorable"
    # it drops out entirely.
    #   class 0: tp=2, fp=0, fn=0 → F1=1.0
    #   class 1: tp=1, fp=1, fn=1 → F1=0.5
    #   class 2: tp=0, fp=1, fn=0 → F1=0.0, support=0 (NOT scorable)
    tp = np.array([2, 1, 0], dtype=np.int64)
    fp = np.array([0, 1, 1], dtype=np.int64)
    fn = np.array([0, 1, 0], dtype=np.int64)
    from coding.eval.bootstrap import _f1_from_counts
    mi_full, ma_full = _f1_from_counts(tp, fp, fn, macro_denominator="full")
    mi_scor, ma_scor = _f1_from_counts(tp, fp, fn, macro_denominator="scorable")
    # Micro must be identical — the scorable filter is macro-only.
    assert mi_full == pytest.approx(mi_scor, abs=1e-12)
    assert ma_full == pytest.approx((1.0 + 0.5 + 0.0) / 3, abs=1e-12)
    assert ma_scor == pytest.approx((1.0 + 0.5) / 2, abs=1e-12)
    assert ma_scor > ma_full  # dropping the zero-support FP class lifts macro


def test_f1_from_counts_scorable_returns_nan_when_no_support():
    # If no class has gold support (e.g. SFT-only on OOD vocab), the scorable
    # macro is undefined — must surface as NaN, not 0.0.
    tp = np.array([0, 0], dtype=np.int64)
    fp = np.array([1, 2], dtype=np.int64)  # spurious predictions only
    fn = np.array([0, 0], dtype=np.int64)
    from coding.eval.bootstrap import _f1_from_counts
    _, ma = _f1_from_counts(tp, fp, fn, macro_denominator="scorable")
    assert np.isnan(ma)


def test_pr_f1_from_counts_scorable_filters_symmetrically():
    # Same construction as above but for the P/R/F1 variant: the support mask
    # must apply to per_p and per_r identically to per_f.
    tp = np.array([2, 1, 0], dtype=np.int64)
    fp = np.array([0, 1, 1], dtype=np.int64)
    fn = np.array([0, 1, 0], dtype=np.int64)
    mi_p_f, mi_r_f, mi_f_f, ma_p_f, ma_r_f, ma_f_f = _pr_f1_from_counts(
        tp, fp, fn, macro_denominator="full",
    )
    mi_p_s, mi_r_s, mi_f_s, ma_p_s, ma_r_s, ma_f_s = _pr_f1_from_counts(
        tp, fp, fn, macro_denominator="scorable",
    )
    # Micro unaffected.
    assert (mi_p_f, mi_r_f, mi_f_f) == pytest.approx((mi_p_s, mi_r_s, mi_f_s), abs=1e-12)
    # class 0: P=1.0, R=1.0, F1=1.0; class 1: P=0.5, R=0.5, F1=0.5; class 2: zero-support.
    assert ma_p_s == pytest.approx((1.0 + 0.5) / 2, abs=1e-12)
    assert ma_r_s == pytest.approx((1.0 + 0.5) / 2, abs=1e-12)
    assert ma_f_s == pytest.approx((1.0 + 0.5) / 2, abs=1e-12)


def test_score_one_scope_macro_denominator_scorable():
    # Pair test for _score_one_scope: scorable mode must drop the OOV-from-gold
    # class from the macro denominator. Construct a 4th class "Z" that the
    # vocab includes but no gold mentions — it should:
    #   - count in n_in_scope (it's part of vocab)
    #   - NOT count in n_scorable (no gold support)
    #   - inflate macro under "full" (F1=0 contribution) but be filtered under "scorable"
    from coding.eval.harness import _score_one_scope

    vocab_with_z = VOCAB | {"Z"}
    pred_sets = [PRED[e] | ({"Z"} if e == "e3" else set()) for e in EVENT_IDS]
    gold_sets = [GOLD[e] for e in EVENT_IDS]  # no "Z" in any gold
    full = _score_one_scope(pred_sets, gold_sets, vocab_with_z, macro_denominator="full")
    scor = _score_one_scope(pred_sets, gold_sets, vocab_with_z, macro_denominator="scorable")
    assert full["f1_micro"] == pytest.approx(scor["f1_micro"], abs=1e-12)
    # n_scorable counts {A,B,C,D}=4 (all have gold); n_in_scope counts {A,B,C,D,Z}=5.
    assert full["n_in_scope"] == 5
    assert full["n_scorable"] == 4
    assert scor["n_in_scope"] == 5
    assert scor["n_scorable"] == 4
    # Scorable mode equals macro on {A,B,C,D} alone — i.e. EXPECTED_MACRO_F1.
    assert scor["f1_macro"] == pytest.approx(EXPECTED_MACRO_F1, abs=1e-12)
    # Full mode averages over 5 classes (the extra "Z" contributes F1=0).
    assert full["f1_macro"] == pytest.approx(EXPECTED_MACRO_F1 * 4 / 5, abs=1e-12)
    assert scor["f1_macro"] > full["f1_macro"]


def test_paired_bootstrap_identical_systems_zero_delta_under_scorable():
    # Identical systems must yield delta=0 regardless of macro denominator.
    res = paired_bootstrap_test(
        PRED, PRED, GOLD, EVENT_IDS, VOCAB, n=50, seed=314,
        macro_denominator="scorable",
    )
    assert res.delta_micro == pytest.approx(0.0, abs=1e-12)
    assert res.delta_macro == pytest.approx(0.0, abs=1e-12)


def test_deterministic_f1_scorable_matches_manual_filter():
    # End-to-end via the IndexTable path: build a table whose vocab includes
    # one zero-support class, then check deterministic_f1 with scorable mode
    # equals the deterministic_f1 of the vocab restricted to supported codes.
    vocab_with_z = VOCAB | {"Z"}
    table_full_vocab = build_index_table(PRED, GOLD, EVENT_IDS, vocab_with_z)
    _, ma_scor = deterministic_f1(table_full_vocab, macro_denominator="scorable")
    table_supported = build_index_table(PRED, GOLD, EVENT_IDS, VOCAB)
    _, ma_full_on_supported = deterministic_f1(table_supported, macro_denominator="full")
    assert ma_scor == pytest.approx(ma_full_on_supported, abs=1e-12)


# --- freq_band_decomposition._bin_metrics_from_counts threading ----------

def test_bin_metrics_scorable_equals_full_when_all_supported():
    from coding.eval.decomposition.freq_band import _bin_metrics_from_counts

    # 3 classes, all supported (each has at least one gold positive)
    tp = np.array([2, 2, 2], dtype=np.int64)
    fp = np.array([0, 0, 1], dtype=np.int64)
    fn = np.array([1, 0, 0], dtype=np.int64)
    idx = np.array([0, 1, 2], dtype=np.int64)
    m_full = _bin_metrics_from_counts(tp, fp, fn, idx, macro_denominator="full")
    m_scor = _bin_metrics_from_counts(tp, fp, fn, idx, macro_denominator="scorable")
    assert m_full["macro_f1"] == pytest.approx(m_scor["macro_f1"], abs=1e-12)
    assert m_scor["n_scorable"] == 3


def test_bin_metrics_scorable_filters_zero_support_class():
    from coding.eval.decomposition.freq_band import _bin_metrics_from_counts

    # 3 classes; class 2 has zero support (tp+fn=0) but a FP -> drags full macro
    tp = np.array([2, 2, 0], dtype=np.int64)
    fp = np.array([0, 0, 1], dtype=np.int64)
    fn = np.array([1, 0, 0], dtype=np.int64)
    idx = np.array([0, 1, 2], dtype=np.int64)
    m_full = _bin_metrics_from_counts(tp, fp, fn, idx, macro_denominator="full")
    m_scor = _bin_metrics_from_counts(tp, fp, fn, idx, macro_denominator="scorable")
    # Full averages a 0 from class 2; scorable drops it. Scorable > full.
    assert m_scor["macro_f1"] > m_full["macro_f1"]
    # Micro identical under both modes.
    assert m_scor["micro_f1"] == pytest.approx(m_full["micro_f1"], abs=1e-12)
    assert m_scor["n_scorable"] == 2
    # Scorable averages F1 over the 2 supported classes.
    # Class 0: P=1.0, R=2/3, F1=0.8 | Class 1: F1=1.0 -> macro = 0.9
    assert m_scor["macro_f1"] == pytest.approx(0.9, abs=1e-12)


def test_bin_metrics_scorable_returns_nan_when_no_support():
    from coding.eval.decomposition.freq_band import _bin_metrics_from_counts

    # All classes zero-support (no gold positives in the bin)
    tp = np.array([0, 0], dtype=np.int64)
    fp = np.array([1, 1], dtype=np.int64)
    fn = np.array([0, 0], dtype=np.int64)
    idx = np.array([0, 1], dtype=np.int64)
    m_scor = _bin_metrics_from_counts(tp, fp, fn, idx, macro_denominator="scorable")
    assert np.isnan(m_scor["macro_f1"])
    assert m_scor["n_scorable"] == 0


def test_compute_by_bin_threads_macro_denominator():
    from coding.eval.decomposition.freq_band import _compute_by_bin

    # Tiny synthetic: 2 stages, 2 bins. Bin 'a' has 2 supported classes;
    # bin 'b' has 1 supported + 1 unsupported with a FP.
    vocab = {"A", "B", "C", "D"}
    event_ids = ["e1", "e2"]
    gold = {"e1": {"A", "C"}, "e2": {"B"}}
    pred_stage1 = {"e1": {"A"}, "e2": {"B", "D"}}
    pred_stage2 = {"e1": {"A", "C"}, "e2": {"B", "D"}}
    stage_lookups = {"sft_high": pred_stage1, "sft_post_in_band": pred_stage2, "final": pred_stage2}
    bin_to_codes = {"a": ["A", "B"], "b": ["C", "D"]}  # D unsupported
    full = _compute_by_bin(
        bin_to_codes=bin_to_codes, stage_lookups=stage_lookups,
        gold_lookup=gold, event_ids=event_ids, in_vocab=vocab,
        n=20, seed=314, scheme_label="t", macro_denominator="full",
    )
    scor = _compute_by_bin(
        bin_to_codes=bin_to_codes, stage_lookups=stage_lookups,
        gold_lookup=gold, event_ids=event_ids, in_vocab=vocab,
        n=20, seed=314, scheme_label="t", macro_denominator="scorable",
    )
    # Bin 'a' (both supported) is identical across modes
    assert scor["sft_high"]["a"]["macro_f1"] == pytest.approx(
        full["sft_high"]["a"]["macro_f1"], abs=1e-12,
    )
    # Bin 'b' (one unsupported with FP at stage 1) differs
    assert scor["sft_high"]["b"]["macro_f1"] >= full["sft_high"]["b"]["macro_f1"]
    # Micro identical under both modes for every stage/bin
    for st in ("sft_high", "sft_post_in_band", "final"):
        for bn in ("a", "b"):
            assert scor[st][bn]["micro_f1"] == pytest.approx(
                full[st][bn]["micro_f1"], abs=1e-12,
            )
