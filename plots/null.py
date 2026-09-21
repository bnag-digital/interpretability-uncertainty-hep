import matplotlib.pyplot as plt
import numpy as np

from metrics.rank import null_tau_vs_n

from . import style

# nothing is required on disk: the null is simulated, not measured
REQUIRES = []

FEATURE_COUNTS = (3, 5, 8, 10, 16, 20)

# below this the null SD is too wide to read a tau; protocols/mechanism.py uses the same
RESOLVABLE_FEATURES = 10

# the counts the study measures at, marked on the null curve
MEASURED_AT = {10: "Covertype", 16: "jet view", 3: "EFN inner"}


def null_table(n_features=FEATURE_COUNTS, n_trials=2000):
    """Null scatter of tau against feature count."""
    return null_tau_vs_n(n_features=n_features, n_trials=n_trials)


def plot_null_vs_n(table, observed=None, ax=None):
    """Null SD of tau against feature count, with measured points on top."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()

    ax.plot(table["n_features"], table["tau_wtd_null_sd"], marker="o",
            color=style.model_color("gnn"), label="weighted tau, null SD")
    ax.plot(table["n_features"], table["tau_std_null_sd"], marker="s", ls="--",
            color=style.model_color("bdt"), label="standard tau, null SD")

    for n, label in sorted(MEASURED_AT.items()):
        ax.axvline(n, color="#888888", lw=1, ls=":", zorder=1)
        ax.text(n, ax.get_ylim()[1], f" {label}", fontsize=8, color="#666666",
                ha="left", va="top", rotation=90)

    ax.set_xlabel("number of features")
    ax.set_ylabel("SD of tau between two random rankings")
    ax.set_title("How much tau scatters when nothing is going on")
    ax.legend()

    if observed:
        for row in observed:
            ax.axvline(row["n_features"], color="#999999", lw=1, ls=":", zorder=1)
    return fig


def plot_false_agreement(table, ax=None):
    """Chance that two unrelated rankings clear a high tau, against feature count."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()

    p = table["p_tau_wtd_gt_0.9"] * 100
    colors = ["#b3462f" if v >= 5 else style.model_color("gnn") for v in p]
    ax.bar(table["n_features"].astype(str), p, color=colors, width=0.65)

    for x, v in enumerate(p):
        ax.text(x, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("number of features")
    ax.set_ylabel("chance of weighted tau > 0.9 (%)")
    ax.set_title("How often two unrelated rankings look like agreement")
    ax.margins(y=0.15)
    return fig


def plot_observed_against_null(table, observed, ax=None):
    """Measured taus against the null band for their own feature count."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7.0, 4.6))

    n = np.asarray(table["n_features"], dtype=float)
    sd = np.asarray(table["tau_wtd_null_sd"], dtype=float)
    order = np.argsort(n)
    ax.fill_between(n[order], -2 * sd[order], 2 * sd[order], color="#999999", alpha=0.25,
                    label="2 SD of the null")
    ax.plot(n[order], 2 * sd[order], color="#777777", lw=1)

    # spread points sharing a feature count, and label only the small spaces
    by_n = {}
    for row in observed:
        by_n.setdefault(row["n_features"], []).append(row)

    for count, group in by_n.items():
        offsets = np.linspace(-0.25, 0.25, len(group)) if len(group) > 1 else [0.0]
        for row, dx in zip(group, offsets):
            ax.scatter(count + dx, row["tau"], s=42, zorder=3, alpha=0.9,
                       color=style.model_color(row.get("model", "")))
            if count < RESOLVABLE_FEATURES:
                # ties are common on a small feature axis, so stagger the labels
                dy = 3 if group.index(row) % 2 == 0 else -10
                ax.annotate(row["label"], (count + dx, row["tau"]), fontsize=7,
                            xytext=(6, dy), textcoords="offset points", color="#444444")
        if count >= RESOLVABLE_FEATURES:
            top = max(r["tau"] for r in group)
            ax.annotate(f"{len(group)} results on {count} features", (count, top),
                        fontsize=7, xytext=(8, 6), textcoords="offset points",
                        color="#444444")

    ax.set_xlabel("number of features")
    ax.set_ylabel("weighted Kendall tau")
    ax.set_title("Measured agreement against the null for the same feature count")
    ax.legend(loc="lower right")
    return fig
