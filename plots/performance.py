import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CLASS_NAMES
from results import level_dataset, load, scan, select

from . import style

# what results.can_draw needs before this is worth calling
REQUIRES = [{"kind": "performance"}]

CLASS_LABELS = {"j_g": "g", "j_q": "q", "j_w": "W", "j_z": "Z", "j_t": "t"}

# `full` is the pre-shared-split OpenML run, on a different set of jets
SUPERSEDED_LEVELS = {"full"}

# the EFN is left out of every figure, for the reason in DESIGN.md
EXCLUDED_MODELS = {"efn"}


def accuracy_frame(df=None, level=None, artifact_dir=None, dataset=None):
    """One row per completed training run: model, level, seed, rows, accuracy."""
    df = scan(artifact_dir) if df is None else df
    rows = select(df, kind="performance")
    if level is not None:
        rows = select(rows, level=level)
    if dataset is not None and not rows.empty:
        rows = rows[rows["level"].map(level_dataset) == dataset]
    if not rows.empty:
        rows = rows[~rows["level"].isin(SUPERSEDED_LEVELS)]
        rows = rows[~rows["model"].isin(EXCLUDED_MODELS)]
    if rows.empty or "accuracy" not in rows.columns:
        return pd.DataFrame(columns=["model", "level", "seed", "n_rows", "accuracy", "epochs"])
    keep = [c for c in ("name", "model", "level", "seed", "n_rows", "accuracy",
                        "epochs", "epochs_run", "class_names") if c in rows.columns]
    out = rows[keep].copy()
    out["seed"] = out["seed"].astype(int)
    return out.sort_values(["model", "n_rows", "seed"]).reset_index(drop=True)


def spread_table(frame):
    """Mean accuracy and seed spread per model and level, in accuracy points."""
    grouped = frame.groupby(["model", "level"], dropna=False)["accuracy"]
    out = grouped.agg(n_seeds="count", mean="mean", lo="min", hi="max").reset_index()
    out["spread_pts"] = (out["hi"] - out["lo"]) * 100
    return out.sort_values(["model", "level"]).reset_index(drop=True)


def plot_accuracy(frame, ax=None):
    """Test accuracy per model, one marker per seed and a bar at the mean."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    models = style.order_models(frame["model"].unique())

    for i, model in enumerate(models):
        sub = frame[frame["model"] == model]
        color = style.model_color(model)
        ax.bar(i, sub["accuracy"].mean(), width=0.6, color=color, alpha=0.35)
        ax.scatter(np.full(len(sub), i), sub["accuracy"], color=color, s=22, zorder=3)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([style.model_label(m) for m in models])
    ax.set_ylabel("test accuracy")
    ax.set_title("Tagging performance, one marker per seed")
    return fig


def class_names_of(frame):
    """The classes a frame's runs were trained on, from the sidecars."""
    if "class_names" in frame.columns:
        for value in frame["class_names"]:
            if isinstance(value, (list, tuple)) and len(value):
                return list(value)
    return list(CLASS_NAMES)


def plot_per_class_auc(frame, ax=None, artifact_dir=None, class_names=None):
    """Per-class AUC, averaged over seeds, grouped by class."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    class_names = class_names or class_names_of(frame)
    models = style.order_models(frame["model"].unique())
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(class_names))

    for i, model in enumerate(models):
        names = frame[frame["model"] == model]["name"]
        aucs = np.stack([load(n, artifact_dir)[0] for n in names])
        ax.bar(x + i * width, aucs.mean(axis=0), width=width,
               color=style.model_color(model), label=style.model_label(model))

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in class_names])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    ax.set_xlabel("class")
    ax.set_title("Per-class AUC, mean over seeds")
    ax.legend(ncol=len(models))
    return fig


def plot_accuracy_vs_size(frame, models=None, ax=None):
    """Accuracy against training rows, with the seed range as the error bar."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
    models = style.order_models(models or frame["model"].unique())

    drawn = 0
    for model in models:
        sub = frame[frame["model"] == model]
        by_size = sub.groupby("n_rows")["accuracy"].agg(["mean", "min", "max"]).reset_index()
        if len(by_size) < 2:
            continue
        err = np.vstack([by_size["mean"] - by_size["min"], by_size["max"] - by_size["mean"]])
        ax.errorbar(by_size["n_rows"], by_size["mean"], yerr=err, marker="o",
                    capsize=3, color=style.model_color(model),
                    label=style.model_label(model))
        drawn += 1

    ax.set_xscale("log")
    ax.set_xlabel("training jets")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy and seed spread against training size")
    if drawn:
        ax.legend()
    return fig
