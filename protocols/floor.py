import numpy as np

from config import CONFIG
from metrics.rank import mean_abs, max_raw_spread, pairwise_tau


def measure_floor(explain_fn, n_repeats=None, seeds=None, cls_idx=None,
                  weighted=True, cfg=None, verbose=True):
    """Rerun one explainer under different seeds with everything else fixed."""
    cfg = cfg or CONFIG
    if seeds is None:
        seeds = list(range(n_repeats if n_repeats is not None else cfg.protocol.n_floor_repeats))
    if len(seeds) < 2:
        raise ValueError(f"a floor needs at least 2 repeats, got {len(seeds)}")

    arrays = []
    for seed in seeds:
        arrays.append(np.asarray(explain_fn(seed), dtype=float))
        if verbose:
            print(f"  floor repeat {len(arrays)}/{len(seeds)} (seed {seed})", flush=True)

    raw = np.stack(arrays)
    stack = np.stack([mean_abs(a, cls_idx) for a in arrays])
    taus = pairwise_tau(stack, weighted)

    return {"level": "floor", "seeds": list(seeds), "n_repeats": len(seeds),
            "n_features": int(stack.shape[1]),
            "taus": taus, "mean_tau": float(taus.mean()), "sd_tau": float(taus.std()),
            "min_tau": float(taus.min()),
            "max_raw_spread": max_raw_spread(raw),
            "deterministic": bool(np.isclose(max_raw_spread(raw), 0.0)),
            "importance": stack, "weighted": weighted}


def floor_from_stack(stack, cls_idx=None, weighted=True):
    """Same measurement from attribution arrays that were already computed."""
    s = np.asarray(stack, dtype=float)
    if len(s) < 2:
        raise ValueError(f"a floor needs at least 2 repeats, got {len(s)}")
    imps = np.stack([mean_abs(s[i], cls_idx) for i in range(len(s))])
    taus = pairwise_tau(imps, weighted)
    return {"level": "floor", "n_repeats": len(s), "n_features": int(imps.shape[1]),
            "taus": taus, "mean_tau": float(taus.mean()), "sd_tau": float(taus.std()),
            "min_tau": float(taus.min()), "max_raw_spread": max_raw_spread(s),
            "importance": imps, "weighted": weighted}
