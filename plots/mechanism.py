import matplotlib.pyplot as plt
import numpy as np

import results
from protocols.mechanism import measure_mechanism

from . import style

def requires(dataset=None):
    return [{"kind": "prediction", "min_seeds": 2},
            {"kind": "attribution", "level": results.level_tag("ceiling", dataset),
             "min_seeds": 2}]


REQUIRES = [{"kind": "prediction", "min_seeds": 2},
            {"kind": "attribution", "level": "ceiling", "min_seeds": 2}]


def prediction_stack(model, train_level, seeds, artifact_dir=None):
    """Predicted classes per seed, (n_seeds, n_rows), in seed order."""
    rows = []
    for seed in seeds:
        name = results.artifact_name("prediction", model, None, train_level, seed)
        rows.append(results.load(name, artifact_dir)[0])
    return np.stack(rows)


def collect_mechanism(model, explainers, train_level="n200000", artifact_dir=None,
                      dataset=None, cls_idx=None):
    """One mechanism measurement per explainer, sharing the model's predictions."""
    level = results.level_tag("ceiling", dataset)
    out = []
    for explainer in explainers:
        try:
            stack, seeds = results.stack_seeds(model, explainer, level, artifact_dir)
            predictions = prediction_stack(model, train_level, seeds, artifact_dir)
        except Exception:
            continue
        row = measure_mechanism(predictions, stack, cls_idx, method=explainer)
        row["model"] = model
        out.append(row)
    return out


def plot_agreement_vs_tau(rows, ax=None):
    """Prediction agreement against attribution tau, one point per seed pair."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.6, 4.6))

    for row in rows:
        if not row["defined"]:
            continue
        color = style.EXPLAINER_COLORS.get(row["method"], "#666666")
        label = f"{row['method']}"
        if np.isfinite(row["spearman_rho"]):
            label += f"  rho {row['spearman_rho']:+.2f}, p {row['p_permutation']:.3f}"
        ax.scatter(row["agreement"], row["taus"], s=38, color=color, label=label, zorder=3)

    ax.set_xlabel("fraction of jets two seeds predict identically")
    ax.set_ylabel("weighted Kendall tau between their attributions")
    title = "Do seeds that predict alike also explain alike?"
    if rows:
        title += f"  ({style.model_label(rows[0]['model'])})"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")

    unresolvable = [r for r in rows if not r["resolvable"]]
    if unresolvable:
        ax.text(0.01, 0.01, f"{len(unresolvable)} method(s) not resolvable: "
                            f"{unresolvable[0]['note'].split(';')[0]}",
                transform=ax.transAxes, fontsize=7, color="#9a5b08", va="bottom")
    return fig


def plot_agreement_spread(rows, ax=None):
    """How much the seeds disagree about jets at all, per model."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.6, 3.6))
    usable = [r for r in rows if r["defined"]]
    if not usable:
        ax.text(0.5, 0.5, "no defined mechanism measurements", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        return fig

    for i, row in enumerate(usable):
        color = style.model_color(row["model"])
        x = np.full(len(row["agreement"]), i) + np.linspace(-0.09, 0.09,
                                                            len(row["agreement"]))
        ax.scatter(x, row["agreement"], s=30, color=color, zorder=3)

    ax.set_xticks(range(len(usable)))
    ax.set_xticklabels([f"{style.model_label(r['model'])}\n{r['method']}" for r in usable],
                       fontsize=8)
    ax.set_ylabel("prediction agreement")
    ax.set_title("How much two seeds of the same model differ on jets")
    return fig
