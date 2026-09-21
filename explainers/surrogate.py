import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from config import CONFIG, SUMMARY_NAMES
from exceptions import LowSurrogateFidelity


def compute_summary(X_raw, mask, node_features, summary_names=None):
    """Jets to a fixed-length vector of summary observables."""
    summary_names = list(summary_names or SUMMARY_NAMES)
    if summary_names != list(SUMMARY_NAMES):
        raise ValueError("compute_summary builds exactly the columns in "
                         f"config.SUMMARY_NAMES, got {summary_names}")

    names = list(node_features)
    i_eta, i_phi = names.index("j1_etarel"), names.index("j1_phirel")
    i_pt, i_dR = names.index("j1_ptrel"), names.index("j1_deltaR")
    i_cth = names.index("j1_costhetarel")

    mask = np.asarray(mask, dtype=np.float32)
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n = mask.sum(axis=1).clip(1)

    pt = X_raw[:, :, i_pt] * mask
    dR = X_raw[:, :, i_dR] * mask
    eta = X_raw[:, :, i_eta] * mask
    phi = X_raw[:, :, i_phi] * mask
    cth = X_raw[:, :, i_cth] * mask

    mean_pt = pt.sum(1) / n
    std_pt = np.sqrt(((pt - mean_pt[:, None]) ** 2 * mask).sum(1) / n)
    wtd_dR = (pt * dR).sum(1) / pt.sum(1).clip(1e-9)

    return np.stack([
        n, mean_pt, pt.max(1), std_pt,
        dR.sum(1) / n, dR.max(1), wtd_dR,
        eta.sum(1) / n, phi.sum(1) / n, cth.sum(1) / n,
    ], axis=1).astype(np.float32)


def fit_summary_surrogate(summaries, logits, class_names, seed=42, holdout_frac=0.2):
    """Fit summary features to model logits, one regressor per class."""
    S = np.asarray(summaries, dtype=np.float64)
    Z = np.asarray(logits, dtype=np.float64)
    if len(S) != len(Z):
        raise ValueError(f"got {len(S)} summary rows and {len(Z)} logit rows")
    n_classes = Z.shape[1]
    if n_classes != len(class_names):
        raise ValueError(f"logits have {n_classes} columns, {len(class_names)} class names given")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(S))
    n_hold = int(holdout_frac * len(perm))
    if n_hold < 1:
        raise ValueError(f"{len(S)} rows and holdout_frac={holdout_frac} leaves no holdout")
    ev_idx, tr_idx = perm[:n_hold], perm[n_hold:]

    regressors = []
    for c in range(n_classes):
        reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                            random_state=seed)
        reg.fit(S[tr_idx], Z[tr_idx, c])
        regressors.append(reg)

    def surrogate_logits(arr):
        arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        return np.stack([r.predict(arr) for r in regressors], axis=1)

    def predict_proba(arr):
        z = surrogate_logits(arr)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    pred_ev = surrogate_logits(S[ev_idx])
    fidelity = {
        "r2_per_class": {class_names[c]: float(r2_score(Z[ev_idx, c], pred_ev[:, c]))
                         for c in range(n_classes)},
        "argmax_agreement": float((pred_ev.argmax(1) == Z[ev_idx].argmax(1)).mean()),
        "seed": int(seed), "n_fit": int(len(tr_idx)), "n_holdout": int(len(ev_idx)),
    }
    fidelity["min_r2"] = float(min(fidelity["r2_per_class"].values()))
    return predict_proba, fidelity


def check_fidelity(fidelity, min_r2=0.7, min_argmax=0.7, raise_on_fail=True):
    """Refuse a surrogate explanation when the surrogate does not track the model."""
    worst_class = min(fidelity["r2_per_class"], key=fidelity["r2_per_class"].get)
    ok = fidelity["min_r2"] >= min_r2 and fidelity["argmax_agreement"] >= min_argmax
    reason = ("" if ok else
              f"surrogate fidelity below threshold: worst class {worst_class} "
              f"R2 {fidelity['min_r2']:.3f} (needs {min_r2}), argmax agreement "
              f"{fidelity['argmax_agreement']:.3f} (needs {min_argmax})")
    if not ok and raise_on_fail:
        raise LowSurrogateFidelity(reason)
    return ok, reason


def fidelity_row(fidelity, model="gnn", cfg=None):
    """Flatten a fidelity dict for the artifact sidecar, so it travels with the result."""
    cfg = cfg or CONFIG
    row = {"model": model, "surrogate_seed": fidelity["seed"],
           "min_r2": fidelity["min_r2"],
           "argmax_agreement": fidelity["argmax_agreement"],
           "n_fit": fidelity["n_fit"], "n_holdout": fidelity["n_holdout"],
           "n_features": len(SUMMARY_NAMES), "dataset": cfg.data.name}
    row.update({f"r2_{cls}": v for cls, v in fidelity["r2_per_class"].items()})
    return row
