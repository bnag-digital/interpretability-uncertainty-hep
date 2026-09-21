from itertools import combinations

import numpy as np

from config import CONFIG
from metrics.rank import mean_abs, permutation_spearman, tau_between


def pairwise_prediction_agreement(predictions):
    """Fraction of rows on which each pair of seeds predicts the same class."""
    p = np.asarray(predictions)
    if p.ndim != 2:
        raise ValueError(f"expected (n_seeds, n_rows), got shape {p.shape}")
    if len(p) < 2:
        raise ValueError(f"need at least 2 seeds, got {len(p)}")
    return np.array([float((p[i] == p[j]).mean()) for i, j in combinations(range(len(p)), 2)])


def measure_mechanism(predictions, stack, cls_idx=None, weighted=True, cfg=None,
                      method=None):
    """Prediction agreement against attribution tau, over every pair of seeds."""
    cfg = cfg or CONFIG
    s = np.asarray(stack, dtype=float)
    agreement = pairwise_prediction_agreement(predictions)
    if len(s) != len(np.asarray(predictions)):
        raise ValueError(f"{len(s)} attribution stacks and "
                         f"{len(np.asarray(predictions))} prediction rows")

    imps = np.stack([mean_abs(s[i], cls_idx) for i in range(len(s))])
    taus = np.array([tau_between(imps[i], imps[j], weighted)
                     for i, j in combinations(range(len(s)), 2)])

    # a constant axis makes scipy return NaN, which reads too easily as a null result
    degenerate = ""
    if np.ptp(agreement) == 0:
        degenerate = "every pair of seeds predicts identically, so agreement is constant"
    elif np.ptp(taus) == 0:
        degenerate = "every pair of seeds attributes identically, so tau is constant"

    if degenerate:
        stats = {"spearman_rho": float("nan"), "p_parametric": float("nan"),
                 "p_permutation": float("nan")}
    else:
        stats = permutation_spearman(agreement, taus, cfg.protocol.n_permutations)

    n_features = int(imps.shape[1])
    notes = [n for n in (degenerate,
                         (f"{n_features} features: tau's null SD is too large here for "
                          "a correlation to be resolvable, so a null is uninformative")
                         if n_features < 10 else "") if n]
    return {"method": method, "n_seeds": len(s), "n_pairs": len(taus),
            "n_features": n_features,
            "agreement": agreement, "taus": taus,
            "mean_agreement": float(agreement.mean()),
            "sd_agreement": float(agreement.std()),
            **stats,
            "defined": not degenerate,
            "resolvable": bool(n_features >= 10 and not degenerate),
            "note": "; ".join(notes)}


def compare_methods(predictions, stacks_by_method, cls_idx=None, weighted=True, cfg=None):
    """One row per attribution method, so the nulls can be read alongside the hit."""
    return [measure_mechanism(predictions, stack, cls_idx, weighted, cfg, method)
            for method, stack in stacks_by_method.items()]
