import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data import hls4ml
from split import load_split

LOG = []


def say(line=""):
    print(line)
    LOG.append(str(line))


def check(label, condition, detail=""):
    say(f"  [{'ok  ' if condition else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return bool(condition)


def check_particle_level(n_jets=None):
    say("=" * 72)
    say("PARTICLE LEVEL (h5)")
    say("=" * 72)
    say(f"file: {CONFIG.data.particle_path}")

    start = time.time()
    data = hls4ml.load_particle_level(n_jets=n_jets, verbose=False)
    say(f"loaded in {time.time() - start:.1f} s")

    n = sum(len(data[f"y_{s}"]) for s in ("train", "val", "test"))
    n_feat = len(data["node_features"])
    ok = []

    say("\nshapes")
    for s in ("train", "val", "test"):
        say(f"  {s:5} X {data[f'X_{s}'].shape}  coords {data[f'coords_{s}'].shape}  "
            f"mask {data[f'mask_{s}'].shape}  y {data[f'y_{s}'].shape}")

    say("\nchecks")
    ok.append(check("node features match the config", n_feat == len(CONFIG.data.node_features),
                    f"{data['node_features']}"))
    ok.append(check("mask is binary",
                    set(np.unique(data["mask_train"])).issubset({0.0, 1.0})))
    ok.append(check("every jet has at least one real constituent",
                    data["mask_train"].sum(axis=1).min() > 0))
    ok.append(check("constituents per jet is within the documented range",
                    5 <= data["mask_train"].sum(axis=1).mean() <= 150,
                    f"mean {data['mask_train'].sum(axis=1).mean():.1f}, "
                    f"max {int(data['mask_train'].sum(axis=1).max())}"))
    ok.append(check("no NaN or inf in the features",
                    np.all(np.isfinite(data["X_train"]))))

    say("\nclass balance (should be roughly even across 5 classes)")
    for s in ("train", "val", "test"):
        counts = np.bincount(data[f"y_{s}"], minlength=len(CONFIG.data.class_names))
        frac = counts / counts.sum()
        say(f"  {s:5} " + "  ".join(f"{c}={f:.3f}" for c, f in
                                    zip(CONFIG.data.class_names, frac)))

    say("\nconstituents per jet, by class (documented as 38-67)")
    counts = data["mask_train"].sum(axis=1)
    for i, cls in enumerate(CONFIG.data.class_names):
        sel = data["y_train"] == i
        if sel.any():
            say(f"  {cls}: {counts[sel].mean():.1f}")

    say("\nnormalisation stats, from training rows only")
    for name, mu, sd in zip(data["node_features"], data["norm"]["mean"], data["norm"]["std"]):
        say(f"  {name:20} mean {mu:+.4f}  std {sd:.4f}")
    ok.append(check("no zero-variance feature", np.all(data["norm"]["std"] > 1e-6)))

    # the stats must not come from val or test rows
    all_mask = np.concatenate([data[f"mask_{s}"] for s in ("train", "val", "test")]).astype(bool)
    all_X = np.concatenate([data[f"X_{s}"] for s in ("train", "val", "test")])
    pooled_mean = all_X[all_mask].mean(axis=0)
    ok.append(check("stats come from train only, not from every row",
                    not np.allclose(data["norm"]["mean"], pooled_mean, atol=1e-9),
                    "train mean differs from the pooled mean, as it should"))

    split_path = CONFIG.data.split_index_path.replace(".npz", "_particle.npz")
    reloaded = load_split(split_path, source_id=hls4ml.PARTICLE_SOURCE_ID, n=n)
    ok.append(check("split index written and reloads",
                    np.array_equal(reloaded["train"], data["split"]["train"]), split_path))

    say(f"\n{n} jets total, {n_feat} node features")
    return all(ok), data


def check_jet_level():
    say()
    say("=" * 72)
    say("JET LEVEL (OpenML)")
    say("=" * 72)

    start = time.time()
    try:
        data = hls4ml.load_jet_level(verbose=False)
    except Exception as exc:
        say(f"  [SKIP] could not load: {type(exc).__name__}: {exc}")
        say("  If this is a network error, the AF may not reach OpenML. The file can be "
            "downloaded once and cached under CONFIG.data.openml_cache.")
        return None, None
    say(f"loaded in {time.time() - start:.1f} s")

    ok = []
    say("\nshapes")
    for s in ("train", "val", "test"):
        say(f"  {s:5} X {data[f'X_{s}'].shape}  y {data[f'y_{s}'].shape}")

    say("\nchecks")
    ok.append(check("16 features", data["X_train"].shape[1] == 16))
    ok.append(check("feature order matches the expected order",
                    data["feature_names"] == hls4ml.EXPECTED_JET_FEATURES))
    ok.append(check("no NaN or inf", np.all(np.isfinite(data["X_train"]))))
    ok.append(check("scaled copy has zero mean on train",
                    np.allclose(data["X_train_sc"].mean(axis=0), 0, atol=1e-9)))
    ok.append(check("5 classes", len(data["class_names"]) == 5, str(data["class_names"])))

    counts = np.bincount(data["y_train"], minlength=5)
    say("\nclass balance (train): " + "  ".join(
        f"{c}={f:.3f}" for c, f in zip(data["class_names"], counts / counts.sum())))

    n = sum(len(data[f"y_{s}"]) for s in ("train", "val", "test"))
    say(f"\n{n} jets total")
    return all(ok), data


def check_views_are_kept_apart(particle_data):
    """The two views come from different files, so their splits must not be shared."""
    say()
    say("=" * 72)
    say("CROSS-VIEW GUARD")
    say("=" * 72)
    from exceptions import IncompatibleFeatureSpace

    n = sum(len(particle_data[f"y_{s}"]) for s in ("train", "val", "test"))
    path = CONFIG.data.split_index_path.replace(".npz", "_particle.npz")
    try:
        load_split(path, source_id=hls4ml.JET_SOURCE_ID, n=n)
        return check("particle split is refused for jet-level data", False,
                     "it was accepted, which would let rows be mismatched")
    except IncompatibleFeatureSpace:
        return check("particle split is refused for jet-level data", True,
                     "row indices cannot be shared between the two files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particle", action="store_true", help="skip the OpenML download")
    parser.add_argument("--n-jets", type=int, default=None, help="limit jets read from the h5")
    parser.add_argument("--log", default=f"{CONFIG.log_dir}/check_data.log")
    args = parser.parse_args()

    CONFIG.make_dirs()
    say(f"artifact dir: {CONFIG.artifact_dir}")

    particle_ok, particle_data = check_particle_level(args.n_jets)
    guard_ok = check_views_are_kept_apart(particle_data)
    jet_ok = None if args.particle else check_jet_level()[0]

    say()
    say("=" * 72)
    say(f"particle level : {'PASS' if particle_ok else 'FAIL'}")
    say(f"cross-view     : {'PASS' if guard_ok else 'FAIL'}")
    say(f"jet level      : {'skipped' if jet_ok is None else ('PASS' if jet_ok else 'FAIL')}")
    say("=" * 72)

    Path(args.log).write_text("\n".join(LOG) + "\n")
    print(f"\nwritten to {args.log}")
    failed = not particle_ok or not guard_ok or jet_ok is False
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
