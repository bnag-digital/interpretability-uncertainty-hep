import numpy as np

from config import CONFIG
from metrics.rank import (bootstrap_over_seeds, cross_tau, margin, mean_abs,
                          paired_permutation_test,
                          signed_rank_displacement, sign_consistency)


def _importance_stack(stack, cls_idx=None):
    s = np.asarray(stack, dtype=float)
    return np.stack([mean_abs(s[i], cls_idx) for i in range(len(s))])


def compare(stack_a, stack_b, label_a, label_b, cls_idx=None, weighted=True, cfg=None):
    """Paired tau between two stacks, matched seed by seed."""
    cfg = cfg or CONFIG
    a, b = _importance_stack(stack_a, cls_idx), _importance_stack(stack_b, cls_idx)
    if len(a) != len(b):
        raise ValueError(f"{label_a} has {len(a)} seeds and {label_b} has {len(b)}; "
                         "the comparison has to be paired")
    if a.shape[1] != b.shape[1]:
        from exceptions import IncompatibleFeatureSpace
        raise IncompatibleFeatureSpace(
            f"{label_a} has {a.shape[1]} features and {label_b} has {b.shape[1]}, "
            "so there is no item correspondence between them"
        )

    taus = cross_tau(a, b, weighted)
    ci = bootstrap_over_seeds(taus, cfg.protocol.n_bootstrap, cfg.protocol.ci)
    return {"comparison": f"{label_a} vs {label_b}", "a": label_a, "b": label_b,
            "n_seeds": len(a), "n_features": int(a.shape[1]),
            "taus": taus, "mean_tau": float(taus.mean()), "sd_tau": float(taus.std()),
            "ci_lower": ci["lower"], "ci_upper": ci["upper"], "weighted": weighted}


def cross_explainer(stack_a, stack_b, model, explainer_a, explainer_b, **kwargs):
    """Two explainers on one model. Both must be in the same output space."""
    out = compare(stack_a, stack_b, f"{model}/{explainer_a}", f"{model}/{explainer_b}", **kwargs)
    out.update({"kind": "cross_explainer", "model": model,
                "explainer_a": explainer_a, "explainer_b": explainer_b})
    return out


def cross_model(stack_a, stack_b, model_a, model_b, explainer, **kwargs):
    """One explainer on two models. Only valid if both share a feature space."""
    out = compare(stack_a, stack_b, f"{model_a}/{explainer}", f"{model_b}/{explainer}", **kwargs)
    out.update({"kind": "cross_model", "model_a": model_a, "model_b": model_b,
                "explainer": explainer})
    return out


def verdict(cross, ceiling_a, ceiling_b=None, floor=None):
    """Put a disagreement next to the noise it has to beat."""
    m = margin(cross["taus"], ceiling_a["taus"],
               ceiling_b["taus"] if ceiling_b is not None else None)
    out = {"comparison": cross["comparison"], **m,
           "ceiling_sd": float(np.std(ceiling_a["taus"])),
           "n_seeds": cross["n_seeds"], "n_features": cross["n_features"]}
    if floor is not None:
        out["floor_tau"] = floor["mean_tau"]
        out["above_floor"] = bool(cross["mean_tau"] < floor["mean_tau"])
    out["reading"] = ("disagreement exceeds retraining noise" if m["below_ceiling"]
                      else "indistinguishable from retraining noise")
    return out


def reproducibility(stack_a, stack_b, model, explainer_a, explainer_b, cls_idx=None):
    """Is the disagreement systematic across seeds, however small it is?"""
    d = signed_rank_displacement(_importance_stack(stack_a, cls_idx),
                                 _importance_stack(stack_b, cls_idx))
    out = sign_consistency(d)
    out.update({"model": model, "a": explainer_a, "b": explainer_b,
                "displacement": d})
    return out


def interaction(cross_shap_models, cross_lime_models, cross_explainers_a,
                cross_explainers_b, cfg=None):
    """The 2x2: does architecture or explainer dominate?"""
    cfg = cfg or CONFIG
    n_perm = cfg.protocol.n_permutations

    explained_with_shap = paired_permutation_test(
        cross_shap_models["taus"], cross_explainers_a["taus"], n_perm)
    explained_with_lime = paired_permutation_test(
        cross_lime_models["taus"], cross_explainers_b["taus"], n_perm)

    means = {"cross_model_shap": cross_shap_models["mean_tau"],
             "cross_model_lime": cross_lime_models["mean_tau"],
             "cross_explainer_a": cross_explainers_a["mean_tau"],
             "cross_explainer_b": cross_explainers_b["mean_tau"]}
    explainer_band = (min(means["cross_explainer_a"], means["cross_explainer_b"]),
                      max(means["cross_explainer_a"], means["cross_explainer_b"]))
    straddles = (means["cross_model_lime"] < explainer_band[0]
                 and means["cross_model_shap"] > explainer_band[1])

    return {"means": means, "explainer_band": explainer_band,
            "shap_test": explained_with_shap, "lime_test": explained_with_lime,
            "reverses": bool(straddles),
            "reading": ("which factor dominates reverses with the explainer used"
                        if straddles else
                        "one factor dominates under both explainers")}
