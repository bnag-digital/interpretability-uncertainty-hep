from pathlib import Path

import matplotlib.pyplot as plt

from config import CONFIG

# optional, so the figures still draw in matplotlib defaults without it
try:
    import mplhep
except ImportError:
    mplhep = None

# fixed order everywhere, jet view then particle view, so the legend reads once
MODEL_ORDER = ["bdt", "dnn", "gnn", "pfn", "efn"]

MODEL_LABELS = {"bdt": "BDT", "dnn": "DNN", "gnn": "GNN", "pfn": "PFN", "efn": "EFN"}

MODEL_COLORS = {
    "bdt": "#4c72b0",
    "dnn": "#dd8452",
    "gnn": "#55a868",
    "pfn": "#c44e52",
    "efn": "#8172b3",
}

EXPLAINER_COLORS = {
    "shap": "#4c72b0",
    "lime": "#dd8452",
    "saliency": "#55a868",
    "ig": "#c44e52",
    "occlusion": "#8172b3",
}

# levels are drawn in this order wherever more than one appears together
LEVEL_LABELS = {"floor": "explainer floor", "ceiling": "retraining ceiling",
                "cross": "cross-explainer"}

FIGSIZE = (7.0, 4.2)
DPI = 150


def setup(hep=True):
    """Matplotlib defaults. Call once before drawing anything."""
    if hep and mplhep is not None:
        plt.style.use(mplhep.style.CMS)
    plt.rcParams.update({
        "figure.figsize": FIGSIZE,
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        # the CMS minor ticks read as clutter under a categorical y axis
        "ytick.minor.visible": False,
    })


def using_hep():
    """Whether the CMS style is actually in force, for a figure to record."""
    return mplhep is not None


def model_color(model):
    return MODEL_COLORS.get(model, "#666666")


def model_label(model):
    return MODEL_LABELS.get(model, str(model).upper())


def order_models(models):
    """Sort a set of model names into MODEL_ORDER, unknown names last."""
    known = [m for m in MODEL_ORDER if m in set(models)]
    return known + sorted(set(models) - set(known))


def note_feature_space(ax, feature_space_id, n_features=None):
    """Stamp the feature axis on any panel that draws tau."""
    text = str(feature_space_id)
    if n_features is not None:
        text = f"{n_features} features, {text}"
    ax.text(0.99, 0.01, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#888888")


def save(fig, name, figure_dir=None, formats=("png", "pdf")):
    """Write a figure once per format. Returns the PNG path."""
    d = Path(figure_dir or CONFIG.figure_dir)
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = d / f"{name}.{ext}"
        fig.savefig(path)
        written.append(path)
    return written[0]
