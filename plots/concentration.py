import matplotlib.pyplot as plt
import numpy as np

import results
from metrics.rank import concentration, mean_abs

from . import style

REQUIRES = [{"kind": "attribution", "level": "ceiling"}]


def requires(dataset=None):
    return [{"kind": "attribution", "level": results.level_tag("ceiling", dataset)}]

# the EFN is left out here too, for the reason in DESIGN.md
DEFAULT_PAIRS = {"bdt": ["shap", "lime"],
                 "dnn": ["shap", "lime", "saliency", "ig", "smoothgrad"],
                 "gnn": ["saliency", "ig", "smoothgrad", "occlusion"],
                 "pfn": ["saliency", "ig", "smoothgrad", "occlusion"]}


def collect_concentration(pairs=None, seed=0, artifact_dir=None, dataset=None,
                          cls_idx=None):
    """Concentration of each model and explainer's importance vector."""
    pairs = pairs or DEFAULT_PAIRS
    level = results.level_tag("ceiling", dataset)
    out = []
    for model, explainers in pairs.items():
        for explainer in explainers:
            name = results.artifact_name("attribution", model, explainer, level, seed)
            try:
                arr, meta = results.load(name, artifact_dir)
            except FileNotFoundError:
                continue
            stats = concentration(mean_abs(arr, cls_idx))
            out.append({"model": model, "explainer": explainer, **stats})
    return out


def plot_concentration(rows, ax=None):
    """eff_n / n per model and explainer, lowest first."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.2, 5.0))
    ordered = sorted(rows, key=lambda r: r["eff_n_frac"])
    y = np.arange(len(ordered))

    ax.barh(y, [r["eff_n_frac"] for r in ordered], height=0.68,
            color=[style.model_color(r["model"]) for r in ordered])
    ax.axvline(1.0, color="#888888", lw=1, ls="--")
    ax.text(1.0, len(ordered) - 0.4, " every feature equal", fontsize=7, color="#888888",
            va="center")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{style.model_label(r['model'])} {r['explainer']}  "
                        f"({r['n_features']}f)" for r in ordered])
    ax.set_xlabel("effective number of features / number of features")
    ax.set_title("How concentrated each importance vector is")
    ax.set_xlim(0, 1.08)
    return fig


def plot_top1_share(rows, ax=None):
    """Share of the total importance held by the single largest feature."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.2, 5.0))
    ordered = sorted(rows, key=lambda r: r["top1_share"], reverse=True)
    y = np.arange(len(ordered))

    ax.barh(y, [r["top1_share"] for r in ordered], height=0.68,
            color=[style.model_color(r["model"]) for r in ordered])
    for i, r in enumerate(ordered):
        ax.text(r["top1_share"], i, f"  {r['top1_share']:.2f}", va="center", fontsize=7,
                color="#666666")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{style.model_label(r['model'])} {r['explainer']}  "
                        f"({r['n_features']}f)" for r in ordered])
    ax.set_xlabel("share of total importance in the top feature")
    ax.set_title("Is one feature carrying the explanation?")
    ax.margins(x=0.12)
    return fig
