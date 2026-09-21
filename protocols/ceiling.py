from itertools import combinations

import numpy as np

from config import CONFIG
from exceptions import InsufficientCapability
from metrics.rank import bootstrap_over_seeds, mean_abs, pairwise_tau
from results import stack_seeds


def measure_ceiling(model, explainer, level="ceiling", artifact_dir=None,
                    cls_idx=None, weighted=True, cfg=None, train_size=None):
    """Read every seed's attribution artifact and compare them pairwise."""
    stack, seeds = stack_seeds(model, explainer, level, artifact_dir)
    out = ceiling_from_stack(stack, cls_idx, weighted, cfg)
    out.update({"model": model, "explainer": explainer, "seeds": seeds,
                "train_size": train_size})
    return out


def ceiling_from_stack(stack, cls_idx=None, weighted=True, cfg=None, train_size=None):
    """Same measurement from an (n_seeds, rows, features[, classes]) stack."""
    cfg = cfg or CONFIG
    s = np.asarray(stack, dtype=float)
    if len(s) < 2:
        raise InsufficientCapability(
            f"a retraining ceiling needs at least 2 seeds, got {len(s)}. One "
            "checkpoint cannot give one, so the verdict has to be withheld."
        )

    imps = np.stack([mean_abs(s[i], cls_idx) for i in range(len(s))])
    taus = pairwise_tau(imps, weighted)

    # bootstrap the per-seed mean tau, so the unit resampled is the seed
    pairs = list(combinations(range(len(s)), 2))
    per_seed = np.array([np.mean([t for (a, b), t in zip(pairs, taus) if i in (a, b)])
                         for i in range(len(s))])
    ci = bootstrap_over_seeds(per_seed, cfg.protocol.n_bootstrap, cfg.protocol.ci)

    return {"level": "ceiling", "n_seeds": len(s), "n_features": int(imps.shape[1]),
            "taus": taus, "mean_tau": float(taus.mean()), "sd_tau": float(taus.std()),
            "per_seed_tau": per_seed, "ci_lower": ci["lower"], "ci_upper": ci["upper"],
            "importance": imps, "weighted": weighted, "train_size": train_size}


def ceiling_vs_size(stacks_by_size, cls_idx=None, weighted=True, cfg=None):
    """One ceiling per training size, from a map of size to stack."""
    rows = []
    for size in sorted(stacks_by_size):
        out = ceiling_from_stack(stacks_by_size[size], cls_idx, weighted, cfg, size)
        rows.append({"train_size": size, "n_seeds": out["n_seeds"],
                     "mean_tau": out["mean_tau"], "sd_tau": out["sd_tau"],
                     "ci_lower": out["ci_lower"], "ci_upper": out["ci_upper"]})
    return rows
