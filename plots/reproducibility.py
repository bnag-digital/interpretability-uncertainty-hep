import matplotlib.pyplot as plt
import numpy as np

import results
from protocols.cross import reproducibility

from . import style

REQUIRES = [{"kind": "attribution", "level": "ceiling", "min_seeds": 2}]


def requires(dataset=None):
    return [{"kind": "attribution", "level": results.level_tag("ceiling", dataset),
             "min_seeds": 2}]


def collect_consistency(pairs, artifact_dir=None, dataset=None, cls_idx=None):
    """One reproducibility result per explainer pair that has artifacts."""
    from itertools import combinations

    level = results.level_tag("ceiling", dataset)
    out = []
    for model, explainers in pairs.items():
        stacks = {}
        for explainer in explainers:
            try:
                stacks[explainer] = results.stack_seeds(model, explainer, level,
                                                        artifact_dir)[0]
            except Exception:
                continue
        for a, b in combinations([e for e in explainers if e in stacks], 2):
            out.append(reproducibility(stacks[a], stacks[b], model, a, b,
                                       cls_idx=cls_idx))
    return out


def plot_consistency(rows, ax=None, order=None, panel_title=None, show_legend=True):
    """Unanimous features per pair against the count random signs would give."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.5, 5.0))
    if order:
        index = {key: i for i, key in enumerate(order)}
        rows = sorted([r for r in rows if (r["model"], r["a"], r["b"]) in index],
                      key=lambda r: index[(r["model"], r["a"], r["b"])])

    for i, r in enumerate(rows):
        color = style.model_color(r["model"])
        ax.barh(i, r["n_unanimous"], height=0.6, color=color, alpha=0.75, zorder=2)

    expected = {r["expected"] for r in rows}
    for value in expected:
        ax.axvline(value, color="#333333", lw=1.2, ls="--", zorder=3)
    if len(expected) == 1:
        ax.text(list(expected)[0], len(rows) - 0.4, " expected from random signs",
                fontsize=8, color="#333333", ha="left", va="top")

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{style.model_label(r['model'])} {r['a']} vs {r['b']}"
                        f"  ({r['n_features']}f)" for r in rows])
    ax.set_xlabel("features the pair disagrees about in the same direction on every seed")
    ax.set_title(panel_title or "Small disagreements can still be systematic", pad=20)

    beat = sum(1 for r in rows if r["p_binomial"] < 0.05)
    ax.text(0.5, 1.01, f"{beat} of {len(rows)} pairs above chance at p < 0.05",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
            color="#666666")
    ax.grid(axis="y", alpha=0.15)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
    return fig


def plot_displacement_contrast(rows, feature_names=None):
    """Two displacement panels side by side, on one shared rank-difference axis."""
    fig, axes = plt.subplots(1, len(rows), figsize=(11.0, 5.5), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, rows):
        plot_displacement(row, feature_names=feature_names, ax=ax)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_displacement(row, feature_names=None, ax=None):
    """Per-feature signed rank difference, one point per seed."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.0, 5.5))
    d = np.asarray(row["displacement"], dtype=float)
    n_seeds, n_features = d.shape
    names = list(feature_names) if feature_names is not None else [
        f"f{j}" for j in range(n_features)]

    order = np.argsort(row["mean_displacement"])
    color = style.model_color(row["model"])
    for i, j in enumerate(order):
        marker = "o" if row["unanimous"][j] else "x"
        ax.scatter(d[:, j], np.full(n_seeds, i), color=color, s=28, marker=marker,
                   alpha=0.8, zorder=3)
        ax.plot([row["mean_displacement"][j]] * 2, [i - 0.25, i + 0.25],
                color=color, lw=2, zorder=4)

    ax.axvline(0.0, color="#888888", lw=1, ls="--", zorder=1)
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels([names[j] for j in order])
    ax.set_xlabel(f"rank difference, {row['a']} minus {row['b']}")
    ax.set_title(f"{style.model_label(row['model'])}: {row['a']} against {row['b']}, "
                 f"{row['n_unanimous']} of {n_features} features one-signed")
    ax.grid(axis="y", alpha=0.15)
    return fig
