import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, DATASET_NAMES, DEFAULT_DATASET
from exceptions import InsufficientCapability

# fields every sidecar carries, in the order they appear in a scan
META_FIELDS = ["kind", "model", "explainer", "level", "seed", "cls", "epochs",
               "n_rows", "n_features", "feature_space_id", "dataset", "output_space"]


def level_tag(name, dataset=None):
    """Prefix a level with its dataset, so two datasets cannot share a filename."""
    dataset = dataset or DEFAULT_DATASET
    if dataset not in DATASET_NAMES:
        raise ValueError(f"unknown dataset '{dataset}', expected one of {DATASET_NAMES}")
    return str(name) if dataset == DEFAULT_DATASET else f"{dataset}_{name}"


def level_dataset(level):
    """Which dataset a level tag belongs to."""
    head = str(level).split("_", 1)[0]
    return head if head in DATASET_NAMES and head != DEFAULT_DATASET else DEFAULT_DATASET


def bare_level(level):
    """A level tag with its dataset prefix removed."""
    dataset = level_dataset(level)
    return str(level) if dataset == DEFAULT_DATASET else str(level).split("_", 1)[1]


def artifact_name(kind, model=None, explainer=None, level=None, seed=None, cls=None):
    """Deterministic filename, so re-running overwrites instead of accumulating."""
    parts = [kind]
    for value in (model, explainer, level):
        if value is not None:
            parts.append(str(value))
    if seed is not None:
        parts.append(f"seed{seed}")
    if cls is not None:
        parts.append(f"cls{cls}")
    return "__".join(parts)


def save(array, kind, artifact_dir=None, **meta):
    """Write payload and sidecar. Returns the path to the .npy."""
    d = Path(artifact_dir or CONFIG.artifact_dir)
    d.mkdir(parents=True, exist_ok=True)

    a = np.asarray(array)
    name = artifact_name(kind, meta.get("model"), meta.get("explainer"),
                         meta.get("level"), meta.get("seed"), meta.get("cls"))
    npy_path = d / f"{name}.npy"
    np.save(npy_path, a)

    sidecar = {"kind": kind, "shape": list(a.shape),
               "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    sidecar.update({k: v for k, v in meta.items() if v is not None})
    sidecar.setdefault("dataset", CONFIG.data.name)
    sidecar.setdefault("n_features", int(a.shape[-1]) if a.ndim else None)
    with open(d / f"{name}.json", "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    return npy_path


def load(name, artifact_dir=None):
    """Load one artifact by name or path. Returns (array, sidecar)."""
    d = Path(artifact_dir or CONFIG.artifact_dir)
    stem = Path(name).stem if str(name).endswith((".npy", ".json")) else str(name)
    npy_path = Path(name) if Path(name).is_file() and str(name).endswith(".npy") else d / f"{stem}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"no artifact at {npy_path}")
    json_path = npy_path.with_suffix(".json")
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    return np.load(npy_path), meta


def scan(artifact_dir=None):
    """Table of every artifact present, read from sidecars only."""
    d = Path(artifact_dir or CONFIG.artifact_dir)
    rows = []
    for json_path in sorted(d.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue
        meta["name"] = json_path.stem
        rows.append(meta)
    if not rows:
        return pd.DataFrame(columns=["name"] + META_FIELDS)

    df = pd.DataFrame(rows)
    ordered = ["name"] + [c for c in META_FIELDS if c in df.columns]
    return df[ordered + [c for c in df.columns if c not in ordered]]


def select(df, **filters):
    """Rows of a scan matching every given field."""
    out = df
    for key, value in filters.items():
        if key not in out.columns:
            return out.iloc[0:0]
        out = out[out[key] == value]
    return out


def seeds_available(df, model, explainer=None, level=None):
    """Sorted list of seeds present for one model, optionally per explainer/level."""
    filters = {"model": model}
    if explainer is not None:
        filters["explainer"] = explainer
    if level is not None:
        filters["level"] = level
    sub = select(df, **filters)
    if "seed" not in sub.columns or sub.empty:
        return []
    return sorted(int(s) for s in sub["seed"].dropna().unique())


def stack_seeds(model, explainer, level="ceiling", artifact_dir=None):
    """Load every seed for one model and explainer as a single stacked array."""
    df = scan(artifact_dir)
    seeds = seeds_available(df, model, explainer, level)
    if len(seeds) < 2:
        raise InsufficientCapability(
            f"{model}/{explainer} has {len(seeds)} seed(s) at level '{level}'. "
            "The retraining ceiling needs at least 2, and no resampling recovers "
            "it from a single checkpoint."
        )
    arrays = []
    for s in seeds:
        arr, _ = load(artifact_name("attribution", model, explainer, level, s), artifact_dir)
        arrays.append(arr)
    return np.stack(arrays), seeds


def can_draw(df, requirements):
    """Check a figure's inputs against a scan. Returns (ok, reason)."""
    missing = []
    for req in requirements:
        req = dict(req)
        min_seeds = req.pop("min_seeds", 1)
        sub = select(df, **req)
        n = len(seeds_available(sub, req.get("model"), req.get("explainer"), req.get("level"))) \
            if req.get("model") else len(sub)
        if sub.empty:
            missing.append(f"nothing matching {req}")
        elif n < min_seeds:
            missing.append(f"{req} has {n} seed(s), needs {min_seeds}")
    if missing:
        return False, "; ".join(missing)
    return True, ""


def summarise(artifact_dir=None):
    """One row per model and explainer: how many seeds, epochs, rows explained."""
    df = scan(artifact_dir)
    if df.empty:
        return df
    # kind belongs in the key; a performance row and a prediction row otherwise merge
    keys = [c for c in ("kind", "model", "explainer", "level") if c in df.columns]
    agg = {}
    if "seed" in df.columns:
        agg["n_seeds"] = ("seed", "nunique")
    for col in ("epochs", "n_rows", "n_features"):
        if col in df.columns:
            agg[col] = (col, "max")
    return df.groupby(keys, dropna=False).agg(**agg).reset_index()
