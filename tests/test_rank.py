import numpy as np
import pytest

from exceptions import IncompatibleFeatureSpace, RankInputError
from metrics import rank


def test_identical_vectors_give_tau_one():
    v = np.array([0.4, 0.1, 0.3, 0.05, 0.15])
    assert rank.weighted_tau_between(v, v) == pytest.approx(1.0)
    assert rank.standard_tau_between(v, v) == pytest.approx(1.0)


def test_reversed_vector_gives_tau_minus_one():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rank.standard_tau_between(v, -v) == pytest.approx(-1.0)


def test_tau_on_argsort_output_is_rejected():
    v = np.array([0.4, 0.1, 0.3, 0.05, 0.15])
    with pytest.raises(RankInputError):
        rank.weighted_tau_between(np.argsort(v), np.argsort(-v))


def test_different_feature_counts_are_rejected():
    with pytest.raises(IncompatibleFeatureSpace):
        rank.weighted_tau_between(np.arange(1.0, 6.0) / 10, np.arange(1.0, 11.0) / 10)


def test_weighted_tau_punishes_top_swaps_more_than_tail_swaps():
    base = np.array([1.0, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    top_swap = base.copy()
    top_swap[[0, 1]] = top_swap[[1, 0]]
    tail_swap = base.copy()
    tail_swap[[6, 7]] = tail_swap[[7, 6]]
    assert rank.weighted_tau_between(base, top_swap) < rank.weighted_tau_between(base, tail_swap)


def test_mean_abs_shapes():
    arr = np.random.default_rng(0).normal(size=(20, 6, 5))
    assert rank.mean_abs(arr).shape == (6,)
    assert rank.mean_abs(arr, cls_idx=2).shape == (6,)
    assert rank.mean_abs(arr[:, :, 0]).shape == (6,)


def test_pairwise_tau_returns_one_value_per_pair():
    stack = np.random.default_rng(0).uniform(size=(5, 8))
    assert rank.pairwise_tau(stack).shape == (10,)


def test_cross_tau_is_matched_elementwise():
    stack = np.random.default_rng(0).uniform(size=(4, 8))
    assert rank.cross_tau(stack, stack) == pytest.approx(np.ones(4))


def test_margin_sign_reports_real_disagreement():
    out = rank.margin(cross=np.array([0.43]), ceiling_a=np.array([0.89]),
                      ceiling_b=np.array([0.72]))
    assert out["min_ceiling"] == pytest.approx(0.72)
    assert out["margin"] == pytest.approx(0.29)
    assert out["below_ceiling"]


def test_max_raw_spread_is_zero_for_a_deterministic_explainer():
    stack = np.tile(np.array([0.3, 0.2, 0.5]), (4, 1))
    assert rank.max_raw_spread(stack) == 0.0


def test_null_tau_scatter_grows_as_features_fall():
    null = rank.null_tau_vs_n(n_features=(5, 16), n_trials=300, random_state=1)
    sd = dict(zip(null["n_features"], null["tau_wtd_null_sd"]))
    assert sd[5] > sd[16]


def test_paired_permutation_detects_a_consistent_difference():
    a = np.array([0.70, 0.72, 0.68, 0.71, 0.69, 0.73, 0.70, 0.72])
    b = a - 0.25
    assert rank.paired_permutation_test(a, b, n_permutations=2000)["p_permutation"] < 0.05
    same = rank.paired_permutation_test(a, a.copy(), n_permutations=2000)
    assert same["p_permutation"] == pytest.approx(1.0)


def test_permutation_spearman_finds_a_planted_correlation():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=28)
    y = x + rng.normal(scale=0.05, size=28)
    out = rank.permutation_spearman(x, y, n_permutations=1000)
    assert out["spearman_rho"] > 0.9
    assert out["p_permutation"] < 0.01


def test_collinearity_detects_a_duplicated_feature():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    X = np.column_stack([X, X[:, 0] + rng.normal(scale=1e-3, size=500)])
    out = rank.collinearity(X, [f"f{i}" for i in range(5)])
    assert out["max_abs_corr"] > 0.99
    assert out["max_vif"] > 50
    # the duplicated pair must be the one reported first
    assert {out["top_pairs"][0][0], out["top_pairs"][0][1]} == {"f0", "f4"}


def test_collinearity_of_independent_features_is_low():
    rng = np.random.default_rng(1)
    out = rank.collinearity(rng.normal(size=(2000, 6)))
    assert out["max_abs_corr"] < 0.2
    assert out["frac_above_0.9"] == 0.0
    assert out["condition_number"] < 2.0


def test_pair_flip_rate_is_zero_when_every_seed_agrees():
    stack = np.array([[3.0, 2.0, 1.0], [3.1, 2.1, 0.9], [2.9, 2.2, 1.1]])
    assert rank.pair_flip_rate(stack).max() == 0.0


def test_pair_flip_rate_finds_the_swapping_pair():
    # features 0 and 1 trade places between seeds, feature 2 never moves
    stack = np.array([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1], [2.0, 1.0, 0.1]])
    flips = rank.pair_flip_rate(stack)
    assert flips[0, 1] > 0
    assert flips[0, 2] == 0.0 and flips[1, 2] == 0.0


def test_flips_against_collinearity_finds_a_planted_link():
    """Correlated pairs made to swap, uncorrelated ones held fixed."""
    corr = np.array([[1.0, 0.95, 0.05], [0.95, 1.0, 0.05], [0.05, 0.05, 1.0]])
    stack = np.array([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1], [2.0, 1.0, 0.1],
                      [1.0, 2.0, 0.1]])
    out = rank.flips_against_collinearity(stack, corr)
    assert out["defined"] and out["spearman_rho"] > 0.9


def test_flips_against_collinearity_refuses_a_mismatched_matrix():
    from exceptions import IncompatibleFeatureSpace

    stack = np.array([[2.0, 1.0, 0.5], [1.0, 2.0, 0.5]])
    with pytest.raises(IncompatibleFeatureSpace):
        rank.flips_against_collinearity(stack, np.eye(4))


def test_concentration_flags_a_dominant_feature():
    spread = rank.concentration(np.ones(10))
    peaked = rank.concentration(np.array([10.0] + [0.1] * 9))
    assert spread["eff_n_frac"] == pytest.approx(1.0)
    assert peaked["eff_n_frac"] < 0.3
    assert peaked["top1_over_top2"] == pytest.approx(100.0)
