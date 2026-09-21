import matplotlib.pyplot as plt
import numpy as np

import results
from protocols.cross import cross_explainer, cross_model, interaction

from . import style

REQUIRES = [{"kind": "attribution", "level": "ceiling", "min_seeds": 2}]


def requires(dataset=None):
    return [{"kind": "attribution", "level": results.level_tag("ceiling", dataset),
             "min_seeds": 2}]

# (model a, model b, explainer a, explainer b)
JET_SQUARE = ("bdt", "dnn", "shap", "lime")
PARTICLE_SQUARE = ("gnn", "pfn", "ig", "occlusion")


def collect_square(square=JET_SQUARE, artifact_dir=None, dataset=None, cls_idx=None):
    """The four comparisons of one 2x2, plus the permutation tests over them."""
    model_a, model_b, expl_a, expl_b = square
    level = results.level_tag("ceiling", dataset)
    stacks = {}
    for model in (model_a, model_b):
        for explainer in (expl_a, expl_b):
            stacks[(model, explainer)] = results.stack_seeds(
                model, explainer, level, artifact_dir)[0]

    cross_models_a = cross_model(stacks[(model_a, expl_a)], stacks[(model_b, expl_a)],
                                 model_a, model_b, expl_a, cls_idx=cls_idx)
    cross_models_b = cross_model(stacks[(model_a, expl_b)], stacks[(model_b, expl_b)],
                                 model_a, model_b, expl_b, cls_idx=cls_idx)
    cross_expl_a = cross_explainer(stacks[(model_a, expl_a)], stacks[(model_a, expl_b)],
                                   model_a, expl_a, expl_b, cls_idx=cls_idx)
    cross_expl_b = cross_explainer(stacks[(model_b, expl_a)], stacks[(model_b, expl_b)],
                                   model_b, expl_a, expl_b, cls_idx=cls_idx)

    out = interaction(cross_models_a, cross_models_b, cross_expl_a, cross_expl_b)
    out["comparisons"] = {"cross_model_a": cross_models_a, "cross_model_b": cross_models_b,
                          "cross_explainer_a": cross_expl_a, "cross_explainer_b": cross_expl_b}
    out["square"] = square
    return out


def plot_square(square_result, ax=None):
    """The four means with their CIs, and the explainer band across them."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.2, 4.4))
    model_a, model_b, expl_a, expl_b = square_result["square"]
    comps = square_result["comparisons"]

    rows = [
        (f"{style.model_label(model_a)} vs {style.model_label(model_b)}, {expl_a}",
         comps["cross_model_a"], "#444444"),
        (f"{style.model_label(model_a)} vs {style.model_label(model_b)}, {expl_b}",
         comps["cross_model_b"], "#444444"),
        (f"{expl_a} vs {expl_b}, {style.model_label(model_a)}",
         comps["cross_explainer_a"], style.model_color(model_a)),
        (f"{expl_a} vs {expl_b}, {style.model_label(model_b)}",
         comps["cross_explainer_b"], style.model_color(model_b)),
    ]

    lo, hi = square_result["explainer_band"]
    ax.axvspan(lo, hi, color=style.model_color(model_a), alpha=0.12,
               label="range spanned by explainer choice")

    for i, (label, comp, color) in enumerate(rows):
        ax.plot([comp["ci_lower"], comp["ci_upper"]], [i, i], color=color, lw=2, alpha=0.6)
        ax.scatter(comp["mean_tau"], i, color=color, s=48, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("weighted Kendall tau")
    ax.set_title("Model choice against explainer choice")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16))

    p_a = square_result["shap_test"]["p_permutation"]
    p_b = square_result["lime_test"]["p_permutation"]
    # rows are in fixed order, not sorted, so the top right corner is the free one
    ax.text(0.99, 0.97, f"paired permutation p: {expl_a} {p_a:.3f}, {expl_b} {p_b:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#666666",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2))
    return fig


def plot_paired_deltas(square_result, ax=None):
    """Per-seed differences behind each permutation test."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4.0))
    model_a, model_b, expl_a, expl_b = square_result["square"]
    comps = square_result["comparisons"]

    series = [
        (expl_a, np.asarray(comps["cross_model_a"]["taus"])
         - np.asarray(comps["cross_explainer_a"]["taus"]), style.model_color(model_a)),
        (expl_b, np.asarray(comps["cross_model_b"]["taus"])
         - np.asarray(comps["cross_explainer_b"]["taus"]), style.model_color(model_b)),
    ]

    for i, (label, d, color) in enumerate(series):
        x = np.full(len(d), i) + np.linspace(-0.08, 0.08, len(d))
        ax.scatter(x, d, color=color, s=34, zorder=3, label=f"explained with {label}")
        ax.plot([i - 0.2, i + 0.2], [d.mean()] * 2, color=color, lw=2)

    ax.axhline(0.0, color="#888888", lw=1, ls="--")
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([s[0] for s in series])
    ax.set_ylabel("cross-model tau minus cross-explainer tau")
    ax.set_title("Per-seed differences the permutation test resamples")
    ax.legend()
    return fig
