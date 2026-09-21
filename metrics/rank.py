from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import (binomtest, kendalltau, weightedtau, rankdata,
                         spearmanr)

from exceptions import IncompatibleFeatureSpace, RankInputError


def _as_importance(v, name="v"):
    """Tau goes on importance values, so reject anything that looks like argsort output."""
    a = np.asarray(v, dtype=float).ravel()
    if a.size < 2:
        raise ValueError(f"{name} needs at least 2 features, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    is_perm = np.array_equal(np.sort(a), np.arange(a.size, dtype=float))
    if is_perm and a.size > 2:
        raise RankInputError(
            f"{name} is a permutation of 0..{a.size - 1}, which is what argsort "
            "returns. Pass importance values instead."
        )
    return a


def _pair(a, b):
    ia, ib = _as_importance(a, "imp_a"), _as_importance(b, "imp_b")
    if ia.size != ib.size:
        raise IncompatibleFeatureSpace(
            f"importance vectors have {ia.size} and {ib.size} features, "
            "so there is no item correspondence and tau is undefined"
        )
    return ia, ib


def normalize_to_share(imp):
    a = np.abs(np.asarray(imp, dtype=float))
    return a / a.sum()


def ranks_from_importance(imp):
    return rankdata(-np.asarray(imp, dtype=float), method="average")


def weighted_tau_between(imp_a, imp_b):
    """Hyperbolic-weighted tau: swaps near the top of the ranking count for more."""
    a, b = _pair(imp_a, imp_b)
    return float(weightedtau(a, b)[0])


def standard_tau_between(imp_a, imp_b):
    a, b = _pair(imp_a, imp_b)
    return float(kendalltau(a, b)[0])


def tau_between(imp_a, imp_b, weighted=True):
    return weighted_tau_between(imp_a, imp_b) if weighted else standard_tau_between(imp_a, imp_b)


def mean_abs(arr, cls_idx=None):
    """Collapse (rows, features[, classes]) to one importance value per feature."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        return np.abs(a).mean(axis=0)
    if a.ndim == 3:
        if cls_idx is not None:
            return np.abs(a[:, :, cls_idx]).mean(axis=0)
        return np.abs(a).mean(axis=(0, 2))
    raise ValueError(f"expected a 2d or 3d attribution array, got shape {a.shape}")


def pairwise_tau(stack, weighted=True):
    """All C(n, 2) taus within an (n_repeats, n_features) stack."""
    s = np.asarray(stack, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"expected (n_repeats, n_features), got shape {s.shape}")
    return np.array([tau_between(s[i], s[j], weighted)
                     for i, j in combinations(range(len(s)), 2)])


def cross_tau(stack_a, stack_b, weighted=True):
    """Paired tau between two stacks, matched repeat by repeat."""
    a, b = np.asarray(stack_a, dtype=float), np.asarray(stack_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"stacks must be matched, got {len(a)} and {len(b)} repeats")
    return np.array([tau_between(a[i], b[i], weighted) for i in range(len(a))])


def pairwise_seed_tau(stack, cls_idx=None, weighted=True):
    """pairwise_tau over (n_seeds, rows, features[, classes]) attribution arrays."""
    s = np.asarray(stack, dtype=float)
    return pairwise_tau(np.stack([mean_abs(s[i], cls_idx) for i in range(len(s))]), weighted)


def margin(cross, ceiling_a, ceiling_b=None):
    """ceiling - cross, using the lower ceiling when two are given."""
    ceilings = [np.mean(ceiling_a)] + ([np.mean(ceiling_b)] if ceiling_b is not None else [])
    min_ceiling = float(min(ceilings))
    m = min_ceiling - float(np.mean(cross))
    return {"cross_tau": float(np.mean(cross)), "min_ceiling": min_ceiling,
            "margin": m, "below_ceiling": bool(m > 0)}


def max_raw_spread(stack):
    """Largest absolute difference between repeats, before any ranking."""
    s = np.asarray(stack, dtype=float)
    return float(np.abs(s - s[0]).max())


def bootstrap_over_seeds(values, n_bootstrap=500, ci=0.90, random_state=42):
    """Confidence interval on a mean tau, resampling seeds rather than pairs."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_state)
    draws = np.array([rng.choice(v, v.size, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return {"mean": float(v.mean()), "lower": float(np.quantile(draws, alpha)),
            "upper": float(np.quantile(draws, 1 - alpha)), "draws": draws}


def paired_permutation_test(tau_a, tau_b, n_permutations=5000, random_state=0):
    """Is one comparison's tau larger than another's, seed by seed?"""
    a, b = np.asarray(tau_a, dtype=float), np.asarray(tau_b, dtype=float)
    if a.size != b.size:
        raise ValueError(f"values must be paired, got {a.size} and {b.size}")
    d = a - b
    observed = float(d.mean())
    rng = np.random.default_rng(random_state)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, d.size))
    null = (signs * d).mean(axis=1)
    return {"observed_diff": observed, "n_pairs": int(d.size),
            "p_permutation": float((np.abs(null) >= abs(observed)).mean())}


def signed_rank_displacement(imps_a, imps_b):
    """Per-seed, per-feature rank difference between two explainers."""
    a = np.asarray(imps_a, dtype=float)
    b = np.asarray(imps_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise IncompatibleFeatureSpace(
            f"displacement needs two (n_seeds, n_features) arrays of one shape, "
            f"got {a.shape} and {b.shape}")
    return np.stack([ranks_from_importance(a[i]) - ranks_from_importance(b[i])
                     for i in range(a.shape[0])])


def sign_consistency(displacement):
    """How many features two explainers disagree about in a consistent direction."""
    d = np.asarray(displacement, dtype=float)
    if d.ndim != 2:
        raise RankInputError(f"displacement must be 2d, got shape {d.shape}")
    n_seeds, n_features = d.shape
    unanimous = ((d > 0).sum(axis=0) == n_seeds) | ((d < 0).sum(axis=0) == n_seeds)
    chance = 2.0 ** (1 - n_seeds)
    n_unanimous = int(unanimous.sum())
    p = float(binomtest(n_unanimous, n_features, chance,
                        alternative="greater").pvalue)
    return {"n_seeds": n_seeds, "n_features": n_features,
            "unanimous": unanimous, "n_unanimous": n_unanimous,
            "chance_rate": chance, "expected": chance * n_features,
            "p_binomial": p, "mean_displacement": d.mean(axis=0)}


def permutation_spearman(x, y, n_permutations=5000, random_state=0):
    """Spearman rho with a permutation p-value."""
    xv, yv = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho, p_param = spearmanr(xv, yv)
    rng = np.random.default_rng(random_state)
    null = np.array([spearmanr(rng.permutation(xv), yv)[0] for _ in range(n_permutations)])
    return {"spearman_rho": float(rho), "p_parametric": float(p_param),
            "p_permutation": float((np.abs(null) >= abs(rho)).mean())}


def null_tau_vs_n(n_features=(5, 10, 16, 20), n_trials=4000, random_state=0):
    """Scatter of tau between two random vectors, against feature count."""
    rng = np.random.default_rng(random_state)
    rows = []
    for n in n_features:
        ts, tw = np.empty(n_trials), np.empty(n_trials)
        for i in range(n_trials):
            a, b = rng.normal(size=n), rng.normal(size=n)
            ts[i] = kendalltau(a, b)[0]
            tw[i] = weightedtau(a, b)[0]
        rows.append({"n_features": n,
                     "tau_std_null_sd": float(ts.std()),
                     "tau_wtd_null_sd": float(tw.std()),
                     "tau_wtd_95th": float(np.percentile(tw, 95)),
                     "p_tau_wtd_gt_0.9": float((tw > 0.9).mean())})
    return pd.DataFrame(rows)


def concentration(imp):
    """How concentrated an importance vector is."""
    p = normalize_to_share(imp)
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    srt = np.sort(p)
    return {"n_features": int(p.size), "entropy": H, "eff_n": float(np.exp(H)),
            "eff_n_frac": float(np.exp(H) / p.size), "gini": float(1 - (p ** 2).sum()),
            "top1_share": float(srt[-1]), "top1_over_top2": float(srt[-1] / srt[-2])}


def collinearity(X, feature_names=None):
    """Correlation structure of an (rows, features) matrix."""
    a = np.asarray(X, dtype=float)
    if a.ndim != 2:
        raise RankInputError(f"expected (rows, features), got shape {a.shape}")
    n_features = a.shape[1]
    names = list(feature_names) if feature_names is not None else \
        [f"feature_{i}" for i in range(n_features)]

    corr = np.corrcoef(a, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    off = ~np.eye(n_features, dtype=bool)
    abs_off = np.abs(corr[off])

    eig = np.linalg.eigvalsh(corr)
    eig = np.clip(eig, 1e-12, None)
    vif = np.diag(np.linalg.pinv(corr))

    pairs = [(names[i], names[j], float(corr[i, j]))
             for i, j in combinations(range(n_features), 2)]
    pairs.sort(key=lambda p: -abs(p[2]))

    return {"n_features": n_features, "feature_names": names, "corr": corr,
            "max_abs_corr": float(abs_off.max()), "mean_abs_corr": float(abs_off.mean()),
            "frac_above_0.9": float((abs_off >= 0.9).mean()),
            "frac_above_0.7": float((abs_off >= 0.7).mean()),
            "condition_number": float(np.sqrt(eig.max() / eig.min())),
            "vif": vif, "max_vif": float(vif.max()), "top_pairs": pairs[:10]}


def pair_flip_rate(stack):
    """Per feature pair, how often two seeds order the pair differently."""
    s = np.asarray(stack, dtype=float)
    if s.ndim != 2 or len(s) < 2:
        raise RankInputError(f"expected (n_seeds, n_features) with 2+ seeds, got {s.shape}")

    n_features = s.shape[1]
    seed_pairs = list(combinations(range(len(s)), 2))
    out = np.zeros((n_features, n_features))
    for i, j in combinations(range(n_features), 2):
        flips = sum(np.sign(s[a, i] - s[a, j]) != np.sign(s[b, i] - s[b, j])
                    for a, b in seed_pairs)
        out[i, j] = out[j, i] = flips / len(seed_pairs)
    return out


def flips_against_collinearity(stack, corr):
    """Spearman correlation between |corr| and flip rate over all feature pairs."""
    flips = pair_flip_rate(stack)
    n_features = flips.shape[0]
    if np.asarray(corr).shape != flips.shape:
        raise IncompatibleFeatureSpace(
            f"correlation matrix is {np.asarray(corr).shape} but the importance "
            f"stack has {n_features} features")

    iu = np.triu_indices(n_features, k=1)
    x, y = np.abs(np.asarray(corr))[iu], flips[iu]
    if np.ptp(y) == 0 or np.ptp(x) == 0:
        return {"spearman_rho": float("nan"), "p_value": float("nan"),
                "n_pairs": int(len(x)), "mean_flip_rate": float(y.mean()),
                "defined": False}
    rho, p = spearmanr(x, y)
    return {"spearman_rho": float(rho), "p_value": float(p), "n_pairs": int(len(x)),
            "mean_flip_rate": float(y.mean()), "defined": True}


def rank_table(imps, feature_names):
    """{method: importance vector} -> shares and ranks, sorted by average rank."""
    ranks = {m: ranks_from_importance(v) for m, v in imps.items()}
    rows = []
    for j, fname in enumerate(feature_names):
        row = {"feature": fname}
        for m, v in imps.items():
            row[f"{m}_share"] = normalize_to_share(v)[j]
            row[f"{m}_rank"] = ranks[m][j]
        row["avg_rank"] = float(np.mean([ranks[m][j] for m in imps]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("avg_rank").reset_index(drop=True)
