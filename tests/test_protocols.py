import numpy as np
import pytest

from config import Config
from exceptions import IncompatibleFeatureSpace, InsufficientCapability
from protocols import ceiling, cross, floor, mechanism

N_FEATURES = 16


def fast_cfg():
    cfg = Config()
    cfg.protocol.n_bootstrap = 50
    cfg.protocol.n_permutations = 200
    return cfg


def attribution(base, jitter=0.0, seed=0, n_rows=20, n_classes=3):
    """An attribution array whose feature ranking is base, plus noise."""
    rng = np.random.default_rng(seed)
    imp = np.abs(base) + jitter * rng.normal(size=len(base))
    arr = np.abs(imp)[None, :, None] * np.ones((n_rows, len(base), n_classes))
    return arr * (1 + 0.01 * rng.normal(size=arr.shape))


def stack_of(base, n, jitter=0.0, seed0=0, **kwargs):
    return np.stack([attribution(base, jitter, seed0 + i, **kwargs) for i in range(n)])


BASE = np.linspace(1.0, 0.1, N_FEATURES)


def test_a_deterministic_explainer_has_a_floor_of_one():
    calls = []

    def explain_fn(seed):
        calls.append(seed)
        return attribution(BASE, jitter=0.0, seed=0)

    out = floor.measure_floor(explain_fn, n_repeats=4, verbose=False)
    assert calls == [0, 1, 2, 3]
    assert out["mean_tau"] == pytest.approx(1.0, abs=1e-9)
    assert out["deterministic"]


def test_a_noisy_explainer_has_a_floor_below_one():
    def explain_fn(seed):
        return attribution(BASE, jitter=0.30, seed=seed)

    out = floor.measure_floor(explain_fn, n_repeats=5, verbose=False)
    assert out["mean_tau"] < 1.0
    assert not out["deterministic"]
    assert out["max_raw_spread"] > 0


def test_a_floor_needs_more_than_one_repeat():
    with pytest.raises(ValueError):
        floor.measure_floor(lambda s: attribution(BASE), n_repeats=1, verbose=False)


def test_rankings_that_move_a_lot_give_a_lower_floor_than_ones_that_barely_move():
    tight = floor.floor_from_stack(stack_of(BASE, 5, jitter=0.02))
    loose = floor.floor_from_stack(stack_of(BASE, 5, jitter=0.40))
    assert tight["mean_tau"] > loose["mean_tau"]


def test_one_checkpoint_cannot_give_a_ceiling():
    """No amount of resampling recovers a retraining ceiling from one model."""
    with pytest.raises(InsufficientCapability, match="at least 2 seeds"):
        ceiling.ceiling_from_stack(stack_of(BASE, 1), cfg=fast_cfg())


def test_a_ceiling_reports_a_ci_resampled_over_seeds():
    out = ceiling.ceiling_from_stack(stack_of(BASE, 5, jitter=0.15), cfg=fast_cfg())
    assert out["n_seeds"] == 5
    assert len(out["per_seed_tau"]) == 5
    assert out["ci_lower"] <= out["mean_tau"] <= out["ci_upper"]


def test_ceiling_records_the_training_size_it_was_measured_at():
    """A ceiling without its training size is not interpretable."""
    rows = ceiling.ceiling_vs_size(
        {10000: stack_of(BASE, 4, jitter=0.40),
         200000: stack_of(BASE, 4, jitter=0.05)}, cfg=fast_cfg())
    assert [r["train_size"] for r in rows] == [10000, 200000]
    # more training data, tighter agreement between seeds
    assert rows[0]["mean_tau"] < rows[1]["mean_tau"]


def test_a_cross_comparison_must_be_paired_seed_by_seed():
    with pytest.raises(ValueError, match="paired"):
        cross.compare(stack_of(BASE, 4), stack_of(BASE, 3), "a", "b", cfg=fast_cfg())


def test_tau_is_refused_across_different_feature_counts():
    """16 jet observables and 10 summary statistics have no item correspondence."""
    other = np.linspace(1.0, 0.1, 10)
    with pytest.raises(IncompatibleFeatureSpace):
        cross.compare(stack_of(BASE, 4), stack_of(other, 4), "jets", "summary",
                      cfg=fast_cfg())


def test_two_explainers_that_rank_features_oppositely_disagree():
    same = cross.cross_explainer(stack_of(BASE, 5, jitter=0.02),
                                 stack_of(BASE, 5, jitter=0.02, seed0=50),
                                 "bdt", "shap", "lime", cfg=fast_cfg())
    opposed = cross.cross_explainer(stack_of(BASE, 5, jitter=0.02),
                                    stack_of(BASE[::-1], 5, jitter=0.02, seed0=50),
                                    "bdt", "shap", "lime", cfg=fast_cfg())
    assert same["mean_tau"] > opposed["mean_tau"]
    assert same["kind"] == "cross_explainer" and same["model"] == "bdt"


def test_a_disagreement_below_the_ceiling_reads_as_noise():
    ceil = ceiling.ceiling_from_stack(stack_of(BASE, 5, jitter=0.35), cfg=fast_cfg())
    tiny = cross.cross_explainer(stack_of(BASE, 5, jitter=0.02),
                                 stack_of(BASE, 5, jitter=0.02, seed0=50),
                                 "gnn", "shap", "lime", cfg=fast_cfg())
    out = cross.verdict(tiny, ceil)
    assert not out["below_ceiling"]
    assert out["reading"] == "indistinguishable from retraining noise"


def test_a_disagreement_above_the_ceiling_reads_as_real():
    ceil = ceiling.ceiling_from_stack(stack_of(BASE, 5, jitter=0.02), cfg=fast_cfg())
    big = cross.cross_explainer(stack_of(BASE, 5, jitter=0.02),
                                stack_of(BASE[::-1], 5, jitter=0.02, seed0=50),
                                "bdt", "shap", "lime", cfg=fast_cfg())
    out = cross.verdict(big, ceil)
    assert out["below_ceiling"]
    assert out["margin"] > 0
    assert out["reading"] == "disagreement exceeds retraining noise"


def test_the_verdict_uses_the_lower_of_two_ceilings():
    tight = ceiling.ceiling_from_stack(stack_of(BASE, 5, jitter=0.02), cfg=fast_cfg())
    loose = ceiling.ceiling_from_stack(stack_of(BASE, 5, jitter=0.35), cfg=fast_cfg())
    comparison = cross.cross_explainer(stack_of(BASE, 5, jitter=0.02),
                                       stack_of(BASE, 5, jitter=0.10, seed0=50),
                                       "bdt", "shap", "lime", cfg=fast_cfg())
    out = cross.verdict(comparison, tight, loose)
    assert out["min_ceiling"] == pytest.approx(loose["mean_tau"])


def test_the_interaction_detects_cross_model_numbers_straddling_cross_explainer():
    """The reversal is the result, so it has to be read off the four means."""
    cfg = fast_cfg()
    hi = cross.compare(stack_of(BASE, 6, jitter=0.02), stack_of(BASE, 6, jitter=0.02, seed0=60),
                       "bdt/shap", "dnn/shap", cfg=cfg)
    lo = cross.compare(stack_of(BASE, 6, jitter=0.02), stack_of(BASE[::-1], 6, jitter=0.02, seed0=60),
                       "bdt/lime", "dnn/lime", cfg=cfg)
    mid_a = cross.compare(stack_of(BASE, 6, jitter=0.02), stack_of(BASE, 6, jitter=0.45, seed0=70),
                          "bdt/shap", "bdt/lime", cfg=cfg)
    mid_b = cross.compare(stack_of(BASE, 6, jitter=0.02), stack_of(BASE, 6, jitter=0.50, seed0=80),
                          "dnn/shap", "dnn/lime", cfg=cfg)

    out = cross.interaction(hi, lo, mid_a, mid_b, cfg=cfg)
    assert out["reverses"]
    assert "reverses with the explainer" in out["reading"]
    assert 0.0 <= out["shap_test"]["p_permutation"] <= 1.0


def test_prediction_agreement_is_one_for_identical_seeds():
    preds = np.tile(np.array([0, 1, 2, 1, 0]), (4, 1))
    assert np.allclose(mechanism.pairwise_prediction_agreement(preds), 1.0)


def test_prediction_agreement_counts_matching_rows():
    preds = np.array([[0, 1, 2, 0], [0, 1, 2, 1]])
    assert mechanism.pairwise_prediction_agreement(preds) == pytest.approx([0.75])


def test_mechanism_links_prediction_agreement_to_attribution_agreement():
    """Seeds that predict alike are built to attribute alike, so rho must be positive."""
    rng = np.random.default_rng(0)
    n_seeds, n_rows = 8, 200
    base_pred = rng.integers(0, 3, size=n_rows)

    preds, stacks = [], []
    for i in range(n_seeds):
        # seed i departs from the shared function by a growing amount
        drift = 0.04 * i
        flip = rng.random(n_rows) < drift
        p = np.where(flip, rng.integers(0, 3, size=n_rows), base_pred)
        preds.append(p)
        stacks.append(attribution(BASE, jitter=drift, seed=100 + i))

    out = mechanism.measure_mechanism(np.stack(preds), np.stack(stacks), cfg=fast_cfg())
    assert out["n_pairs"] == n_seeds * (n_seeds - 1) // 2
    assert out["spearman_rho"] > 0
    assert 0.0 <= out["p_permutation"] <= 1.0
    assert out["resolvable"]


def varied_preds(n_seeds, n_rows=60, seed=0):
    """Seeds that disagree with each other, so prediction agreement varies."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 3, size=n_rows)
    out = []
    for i in range(n_seeds):
        flip = rng.random(n_rows) < 0.05 * (i + 1)
        out.append(np.where(flip, rng.integers(0, 3, size=n_rows), base))
    return np.stack(out)


def test_a_null_on_too_few_features_is_flagged_as_uninformative():
    """At 5 features tau's null SD is 0.44, so a null there means nothing."""
    small = np.linspace(1.0, 0.2, 5)
    stack = stack_of(small, 4, jitter=0.1)
    out = mechanism.measure_mechanism(varied_preds(4), stack, cfg=fast_cfg())
    assert not out["resolvable"]
    assert "null SD is too large" in out["note"]


def test_identical_seeds_give_no_correlation_rather_than_a_silent_nan():
    """A constant axis has no correlation, and that must not read as a null result."""
    preds = np.tile(np.array([0, 1, 2, 1, 0, 2]), (4, 1))
    out = mechanism.measure_mechanism(preds, stack_of(BASE, 4, jitter=0.1), cfg=fast_cfg())
    assert not out["defined"]
    assert not out["resolvable"]
    assert np.isnan(out["spearman_rho"])
    assert "predicts identically" in out["note"]


def test_compare_methods_returns_one_row_per_attribution_method():
    rows = mechanism.compare_methods(
        varied_preds(5), {"shap": stack_of(BASE, 5, jitter=0.05),
                          "lime": stack_of(BASE, 5, jitter=0.30)}, cfg=fast_cfg())
    assert [r["method"] for r in rows] == ["shap", "lime"]
    assert all(r["defined"] for r in rows)
