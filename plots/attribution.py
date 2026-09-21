import matplotlib.pyplot as plt
import numpy as np

import results
from metrics.rank import mean_abs, normalize_to_share, tau_between

from . import style

REQUIRES = [{"kind": "attribution", "level": "ceiling"}]


def requires(dataset=None):
    """What this figure needs, for one dataset's artifacts."""
    return [{"kind": "attribution", "level": results.level_tag("ceiling", dataset)}]


def importance_of(model, explainer, seed=0, artifact_dir=None, cls_idx=None, dataset=None):
    """One importance vector per feature, from a stored attribution artifact."""
    level = results.level_tag("ceiling", dataset)
    name = results.artifact_name("attribution", model, explainer, level, seed)
    arr, meta = results.load(name, artifact_dir)
    return mean_abs(arr, cls_idx), meta


def plot_share_bars(imps, feature_names, ax=None, title=None):
    """Normalised importance share per feature, grouped by method."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9.0, 4.4))
    labels = list(imps)
    shares = {k: normalize_to_share(v) for k, v in imps.items()}
    order = np.argsort(-np.mean([shares[k] for k in labels], axis=0))

    width = 0.8 / len(labels)
    x = np.arange(len(feature_names))
    for i, label in enumerate(labels):
        ax.bar(x + i * width, shares[label][order], width=width,
               color=style.EXPLAINER_COLORS.get(label, f"C{i}"), label=label)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([feature_names[j] for j in order], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("share of total importance")
    ax.set_title(title or "Importance share per feature")
    ax.legend()
    return fig


def shares_across_seeds(model, explainer, artifact_dir=None, cls_idx=None,
                        dataset=None):
    """Importance shares per feature, one row per model seed."""
    level = results.level_tag("ceiling", dataset)
    stack, seeds = results.stack_seeds(model, explainer, level, artifact_dir)
    shares = np.stack([normalize_to_share(mean_abs(stack[i], cls_idx))
                       for i in range(len(stack))])
    return shares, seeds


def plot_share_bars_spread(shares, feature_names, ax=None, title=None):
    """Importance share per feature, with the spread retraining produces."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9.4, 4.6))
    labels = list(shares)
    means = {k: shares[k].mean(axis=0) for k in labels}
    order = np.argsort(-np.mean([means[k] for k in labels], axis=0))

    width = 0.8 / len(labels)
    x = np.arange(len(order))
    for i, label in enumerate(labels):
        m = means[label][order]
        lo = m - shares[label].min(axis=0)[order]
        hi = shares[label].max(axis=0)[order] - m
        ax.bar(x + i * width, m, width=width,
               color=style.EXPLAINER_COLORS.get(label, f"C{i}"), label=label)
        ax.errorbar(x + i * width, m, yerr=np.vstack([lo, hi]), fmt="none",
                    ecolor="#333333", elinewidth=1.0, capsize=2, zorder=4)

    n_seeds = len(next(iter(shares.values())))
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([feature_names[j] for j in order], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("share of total importance")
    ax.set_title(title or f"Importance share per feature. Bar is the mean over "
                          f"{n_seeds} model seeds, whisker the full range")
    ax.legend()
    return fig


def plot_share_scatter(imp_a, imp_b, feature_names, label_a, label_b, ax=None,
                       annotate=True):
    """One explainer's shares against another's, with y = x for reference."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.4, 5.0))
    a, b = normalize_to_share(imp_a), normalize_to_share(imp_b)

    ax.scatter(a, b, s=46, color=style.model_color("gnn"), zorder=3)
    if annotate:
        for j, fname in enumerate(feature_names):
            ax.annotate(fname, (a[j], b[j]), fontsize=6, xytext=(3, 3),
                        textcoords="offset points", color="#666666")

    top = max(a.max(), b.max()) * 1.12
    ax.plot([0, top], [0, top], color="#888888", ls="--", lw=1, zorder=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel(f"{label_a} share")
    ax.set_ylabel(f"{label_b} share")
    ax.set_title(f"{label_a} against {label_b}\nweighted tau = {tau_between(a, b):.3f}")
    ax.set_aspect("equal")
    return fig


def plot_seed_spread(stack, feature_names, ax=None, title=None):
    """Per-feature importance share across model seeds."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(9.0, 4.4))
    s = np.asarray(stack, dtype=float)
    shares = np.stack([normalize_to_share(mean_abs(s[i])) for i in range(len(s))])
    order = np.argsort(-shares.mean(axis=0))
    x = np.arange(shares.shape[1])

    ax.plot(x, shares.mean(axis=0)[order], color="#444444", lw=1, zorder=2)
    for i in range(len(shares)):
        ax.scatter(x, shares[i][order], s=18, alpha=0.75, zorder=3,
                   color=style.model_color("pfn"))

    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[j] for j in order], rotation=45, ha="right",
                       fontsize=7)
    ax.set_ylabel("share of total importance")
    ax.set_title(title or "Importance share per feature, one point per model seed")
    return fig
