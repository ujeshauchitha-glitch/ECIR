"""Metrics checked against answers worked out BY HAND, not against the code."""
import math

import numpy as np
import pytest

from metrics import (
    average_precision, bootstrap_ci, bootstrap_metric_ci, dcg_at_k, directional_accuracy,
    holm_correction, ndcg_at_k, paired_bootstrap_pvalue, per_query_scores, r_squared, spearman_rho,
)


def test_dcg_hand_computed():
    # rels [3,2,0]: 3/log2(2) + 2/log2(3) + 0
    assert dcg_at_k([3, 2, 0], 3) == pytest.approx(3 + 2 / math.log2(3))
    assert dcg_at_k([3, 2, 0], 1) == pytest.approx(3.0)


def test_ndcg_perfect_and_reversed_and_zero():
    assert ndcg_at_k([1.0, 0.5, 0.0], 3) == pytest.approx(1.0)
    rev = ndcg_at_k([0.0, 0.5, 1.0], 3)
    assert 0 < rev < 1
    assert ndcg_at_k([0.0, 0.0, 0.0], 5) == 0.0            # no relevant items -> 0, not NaN
    assert ndcg_at_k([], 5) == 0.0                          # empty list must not crash


def test_ndcg_k_larger_than_list():
    assert ndcg_at_k([1.0, 0.0], 10) == pytest.approx(1.0)


def test_average_precision_hand_computed():
    # relevant at ranks 1 and 3: (1/1 + 2/3)/2
    assert average_precision([1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([0, 0, 0]) == 0.0
    assert average_precision([1, 1, 1]) == pytest.approx(1.0)
    assert average_precision([]) == 0.0


def test_average_precision_counts_graded_relevance_above_threshold_only():
    assert average_precision([0.4, 0.0, 0.9], relevance_threshold=0.5) == pytest.approx(1 / 3)


def test_directional_accuracy():
    assert directional_accuracy([1, -1, 1, -1], [2, -3, -1, 1]) == pytest.approx(0.5)
    assert directional_accuracy([1, 1], [1, 1]) == 1.0


def test_r_squared():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r_squared(y, y) == pytest.approx(1.0)
    assert r_squared(y, np.full(4, y.mean())) == pytest.approx(0.0)      # mean predictor
    assert r_squared(y, y[::-1]) < 0                                       # worse than mean: NEGATIVE, not clipped
    assert r_squared([2, 2, 2], [2, 2, 2]) == 0.0                          # zero variance must not divide by 0


def test_spearman():
    assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_rho([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)  # monotone, nonlinear


def test_bootstrap_ci_brackets_mean_and_constant_has_zero_width():
    vals = list(np.random.default_rng(1).normal(5, 1, 200))
    m, lo, hi = bootstrap_ci(vals)
    assert lo <= m <= hi
    m, lo, hi = bootstrap_ci([3.0] * 20)
    assert (m, lo, hi) == (3.0, 3.0, 3.0)


def test_bootstrap_ci_is_deterministic_given_seed():
    v = list(np.random.default_rng(2).normal(size=50))
    assert bootstrap_ci(v, seed=7) == bootstrap_ci(v, seed=7)


def test_bootstrap_metric_ci_point_matches_metric():
    y = list(np.arange(30.0))
    p = list(np.arange(30.0) + np.random.default_rng(0).normal(0, 1, 30))
    pt, lo, hi = bootstrap_metric_ci(y, p, r_squared)
    assert pt == pytest.approx(r_squared(y, p)) and lo <= pt <= hi


def test_paired_pvalue_identical_is_one_and_never_exactly_zero():
    a = list(np.random.default_rng(3).normal(size=40))
    assert paired_bootstrap_pvalue(a, a) == pytest.approx(1.0)
    hi = list(np.random.default_rng(4).normal(10, 0.1, 60))
    lo = list(np.random.default_rng(5).normal(0, 0.1, 60))
    p = paired_bootstrap_pvalue(hi, lo, n_resamples=1000)
    assert 0 < p < 0.01, "must be small but NEVER exactly 0"
    assert p == pytest.approx(1 / 1001)


def test_paired_pvalue_rejects_length_mismatch():
    with pytest.raises(AssertionError):
        paired_bootstrap_pvalue([1, 2, 3], [1, 2])


def test_holm_all_pass():
    r = holm_correction({"a": 0.001, "b": 0.02, "c": 0.04})
    assert [r[k][1] for k in "abc"] == [True, True, True]
    assert r["a"][0] == pytest.approx(0.05 / 3)
    assert r["b"][0] == pytest.approx(0.05 / 2)
    assert r["c"][0] == pytest.approx(0.05)


def test_holm_stops_at_first_failure_and_reports_each_rank_threshold():
    r = holm_correction({"a": 0.001, "b": 0.03, "c": 0.031, "d": 0.0001})
    # sorted: d .0001 (<.0125 ok), a .001 (<.0167 ok), b .03 (>= .025 FAIL), c .031 fails by step-down
    assert r["d"][1] and r["a"][1] and not r["b"][1] and not r["c"][1]
    assert r["c"][0] == pytest.approx(0.05 / 1)          # c's OWN rank threshold (regression: was mis-reported)
    assert r["b"][0] == pytest.approx(0.05 / 2)


def test_holm_thresholds_are_per_rank_after_failure():
    # m=5 thresholds are .01,.0125,.0167,.025,.05. 'b' fails; c,d,e must each report THEIR OWN rank's threshold
    # (the old code gave all three the same value).
    r = holm_correction({"a": 0.001, "b": 0.04, "c": 0.041, "d": 0.042, "e": 0.043})
    assert r["a"][1] and not any(r[k][1] for k in "bcde")
    assert r["c"][0] == pytest.approx(0.05 / 3)
    assert r["d"][0] == pytest.approx(0.05 / 2)
    assert r["e"][0] == pytest.approx(0.05 / 1)


def test_holm_step_down_blocks_later_pvalue_that_would_pass_alone():
    r = holm_correction({"x": 0.04, "y": 0.045}, alpha=0.05)
    assert not r["x"][1] and not r["y"][1]


def test_per_query_scores_shapes():
    out = per_query_scores([[1, 0, 0], [0, 1, 0]])
    assert set(out) == {"ndcg@5", "ndcg@10", "map"} and all(len(v) == 2 for v in out.values())
