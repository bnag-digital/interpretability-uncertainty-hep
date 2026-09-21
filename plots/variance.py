from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

import results
from protocols.ceiling import ceiling_from_stack
from protocols.cross import cross_explainer, verdict
from protocols.floor import floor_from_stack

from . import style

REQUIRES = [{"kind": "attribution", "level": "ceiling", "min_seeds": 2}]


def requires(dataset=None):
    return [{"kind": "attribution", "level": results.level_tag("ceiling", dataset),
             "min_seeds": 2}]

JET_EXPLAINERS = ["shap", "lime"]
# the DNN takes the gradient methods as well, so both families sit on one model
DNN_EXPLAINERS = ["shap", "lime", "saliency", "ig", "smoothgrad"]
PARTICLE_EXPLAINERS = ["saliency", "ig", "smoothgrad", "occlusion"]
# the EFN is left out of every figure; DESIGN.md says why
DEFAULT_PAIRS = {"bdt": JET_EXPLAINERS, "dnn": DNN_EXPLAINERS,
                 "gnn": PARTICLE_EXPLAINERS, "pfn": PARTICLE_EXPLAINERS}

# the verdict panel drops rows below this many features
MIN_READABLE_FEATURES = 10

DEFAULT_VERDICT_TITLE = ("Dots inside the grey band are disagreements "
                         "retraining alone produces")


def collect_levels(pairs=None, artifact_dir=None, dataset=None, cls_idx=None):
    """Floor and ceiling for every model and explainer that has artifacts."""
    pairs = pairs or DEFAULT_PAIRS
    ceiling_level = results.level_tag("ceiling", dataset)
    floor_level = results.level_tag("floor", dataset)
    out = []
    for model, explainers in pairs.items():
        for explainer in explainers:
            try:
                c_stack, seeds = results.stack_seeds(model, explainer, ceiling_level,
                                                     artifact_dir)
            except Exception:
                continue
            ceiling = ceiling_from_stack(c_stack, cls_idx)
            row = {"model": model, "explainer": explainer,
                   "ceiling_tau": ceiling["mean_tau"],
                   "ceiling_lo": ceiling["ci_lower"], "ceiling_hi": ceiling["ci_upper"],
                   "n_seeds": len(seeds), "n_features": ceiling["n_features"],
                   "floor_tau": None, "deterministic": True}
            try:
                f_stack, _ = results.stack_seeds(model, explainer, floor_level, artifact_dir)
                floor = floor_from_stack(f_stack, cls_idx)
                row["floor_tau"] = floor["mean_tau"]
                row["deterministic"] = bool(np.isclose(floor["max_raw_spread"], 0.0))
            except Exception:
                pass
            out.append(row)
    return out


def collect_verdicts(pairs=None, artifact_dir=None, dataset=None, cls_idx=None):
    """Cross-explainer disagreement against the ceiling it has to beat."""
    pairs = pairs or DEFAULT_PAIRS
    ceiling_level = results.level_tag("ceiling", dataset)
    floor_level = results.level_tag("floor", dataset)
    out = []
    for model, explainers in pairs.items():
        stacks, ceilings, floors = {}, {}, {}
        for explainer in explainers:
            try:
                stacks[explainer] = results.stack_seeds(model, explainer, ceiling_level,
                                                        artifact_dir)[0]
            except Exception:
                continue
            ceilings[explainer] = ceiling_from_stack(stacks[explainer], cls_idx)
            try:
                floors[explainer] = floor_from_stack(
                    results.stack_seeds(model, explainer, floor_level, artifact_dir)[0],
                    cls_idx)
            except Exception:
                floors[explainer] = None

        for a, b in combinations([e for e in explainers if e in stacks], 2):
            x = cross_explainer(stacks[a], stacks[b], model, a, b, cls_idx=cls_idx)
            v = verdict(x, ceilings[a], ceilings[b], floors.get(a))
            out.append({"model": model, "a": a, "b": b, "cross_tau": x["mean_tau"],
                        "cross_lo": x["ci_lower"], "cross_hi": x["ci_upper"],
                        "ceiling_tau": min(ceilings[a]["mean_tau"], ceilings[b]["mean_tau"]),
                        "margin": v["margin"], "below_ceiling": v["below_ceiling"],
                        "n_features": x["n_features"]})
    return out


def _labels(rows, fmt):
    """Tick labels carrying the feature count, so two axes are never conflated."""
    return [f"{fmt(r)}  ({r['n_features']}f)" for r in rows]


def _grouped(rows, key):
    """Order rows by feature space, widest first, and return the block edges."""
    sizes = sorted({r["n_features"] for r in rows}, reverse=True)
    ordered, edges = [], []
    for size in sizes:
        if ordered:
            edges.append(len(ordered) - 0.5)
        ordered.extend(sorted([r for r in rows if r["n_features"] == size], key=key))
    return ordered, edges


def _separate(ax, rows, edges):
    """Draw the feature-space dividers and say why they are there."""
    for edge in edges:
        ax.axhline(edge, color="#888888", lw=1, ls="--", zorder=2)
    if edges:
        n = rows[int(edges[-1] + 0.5)]["n_features"]
        ax.text(0.01, edges[-1] + 0.3,
                f"{n}-feature axis above: not comparable with the rows below",
                transform=ax.get_yaxis_transform(), fontsize=7, color="#666666",
                va="bottom", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))


def _legend_below(ax, ncol):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=ncol)


def plot_levels(levels, ax=None):
    """Floor and ceiling per model and explainer, worst ceiling at the bottom."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.5, 5.5))
    rows, edges = _grouped(levels, key=lambda r: r["ceiling_tau"])
    y = np.arange(len(rows))

    for i, r in enumerate(rows):
        color = style.model_color(r["model"])
        ax.plot([r["ceiling_lo"], r["ceiling_hi"]], [i, i], color=color, lw=2, alpha=0.5)
        ax.scatter(r["ceiling_tau"], i, color=color, s=45, zorder=3,
                   label="retraining ceiling" if i == 0 else None)
        if r["floor_tau"] is not None:
            marker = "|" if r["deterministic"] else "d"
            ax.scatter(r["floor_tau"], i, color="#333333", s=55, marker=marker, zorder=4,
                       label="explainer floor" if i == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(_labels(rows, lambda r: f"{style.model_label(r['model'])} {r['explainer']}"))
    ax.set_xlabel("weighted Kendall tau")
    ax.set_title("Explainer floor and retraining ceiling")
    _separate(ax, rows, edges)
    _legend_below(ax, 2)
    ax.grid(axis="y", alpha=0.15)
    return fig


def _drop_narrow(rows, min_features):
    """Split rows into the ones wide enough to read and the count dropped."""
    if min_features is None:
        return list(rows), 0
    kept = [r for r in rows if r["n_features"] >= min_features]
    return kept, len(rows) - len(kept)


def _verdict_legend(ax, rows, ncol=None):
    """Colour is the model, so the legend names the models rather than the levels."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    models = style.order_models({r["model"] for r in rows})
    handles = [Line2D([], [], marker="o", ls="", markersize=7,
                      color=style.model_color(m)) for m in models]
    labels = [style.model_label(m) for m in models]
    handles.append(Patch(facecolor="#bbbbbb", alpha=0.35))
    labels.append("within retraining noise")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=ncol or min(len(labels), 6))


def plot_verdicts(verdicts, ax=None, min_features=MIN_READABLE_FEATURES,
                  panel_title=None, show_legend=True):
    """Cross-explainer tau against the ceiling it has to clear."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.5, 5.0))
    kept, n_dropped = _drop_narrow(verdicts, min_features)
    rows, edges = _grouped(kept, key=lambda r: r["cross_tau"])

    for i, r in enumerate(rows):
        color = style.model_color(r["model"])
        ax.barh(i, 1.0 - r["ceiling_tau"], left=r["ceiling_tau"], height=0.7,
                color="#bbbbbb", alpha=0.35, zorder=1)
        ax.plot([r["cross_lo"], r["cross_hi"]], [i, i], color=color, lw=2, alpha=0.6)
        ax.scatter(r["cross_tau"], i, color=color, s=45, zorder=3)

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(_labels(rows, lambda r: f"{style.model_label(r['model'])} "
                                               f"{r['a']} vs {r['b']}"))
    ax.set_xlabel("weighted Kendall tau")
    ax.set_title(panel_title or DEFAULT_VERDICT_TITLE, pad=20)

    n_inside = sum(1 for r in rows if r["cross_tau"] >= r["ceiling_tau"])
    parts = [f"{n_inside} of {len(rows)} pairs inside the band"]
    if n_dropped:
        parts.append(f"{n_dropped} rows below {min_features} features not shown")
    # above the axes, since every corner holds either a dot or a band
    ax.text(0.5, 1.01, ("  " + chr(183) + "  ").join(parts), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8, color="#666666")

    _separate(ax, rows, edges)
    if show_legend:
        _verdict_legend(ax, rows)
    ax.grid(axis="y", alpha=0.15)
    return fig


def plot_verdicts_by_dataset(panels, min_features=MIN_READABLE_FEATURES,
                             figsize=(13.5, 5.4)):
    """One verdict panel per dataset, side by side on a shared tau axis."""
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (label, rows) in zip(axes, panels):
        plot_verdicts(rows, ax=ax, min_features=min_features, panel_title=label,
                      show_legend=False)
        ax.set_xlabel("weighted Kendall tau")

    shown = [r for _, rows in panels
             for r in _drop_narrow(rows, min_features)[0]]
    _verdict_legend(axes[0], shown)
    fig.suptitle(DEFAULT_VERDICT_TITLE, y=1.08)
    fig.tight_layout()
    return fig
