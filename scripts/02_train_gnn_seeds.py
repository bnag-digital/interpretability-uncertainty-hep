import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import results
from config import CONFIG
from data import hls4ml
from models import gnn


def subset(data, n_read, n_max):
    """Nested subset of an already-loaded dataset, keeping the same test set."""
    if n_read >= n_max:
        return data
    share = n_read / n_max
    out = dict(data)
    for split in ("train", "val"):
        keep = int(round(len(data[f"y_{split}"]) * share))
        for key in ("X", "coords", "mask", "y"):
            out[f"{key}_{split}"] = data[f"{key}_{split}"][:keep]
    # stats must come from this subset's training rows, not the full set's
    out["norm"] = hls4ml._norm_from(out["X_train"], out["mask_train"])
    return out


def already_done(size, seed, artifact_dir):
    name = results.artifact_name("performance", "gnn", None, f"n{size}", seed)
    return (Path(artifact_dir) / f"{name}.json").exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 50000, 200000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-test-jets", type=int, default=50000)
    parser.add_argument("--force", action="store_true", help="retrain runs that already finished")
    parser.add_argument("--patience", type=int, default=None,
                        help="override early stopping patience")
    args = parser.parse_args()

    cfg = CONFIG
    cfg.make_dirs()
    if args.patience is not None:
        cfg.gnn.early_stop_patience = args.patience
    cfg.data.n_test_jets = args.n_test_jets
    n_max = max(args.sizes)
    cfg.data.split_index_path = f"artifacts/split_gnn_{n_max}.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72, flush=True)
    print(f"GNN SEED STUDY  sizes {args.sizes}  seeds {args.seeds}", flush=True)
    print(f"device {device}, {len(cfg.data.node_features)} node features, "
          f"cap {cfg.gnn.epochs} epochs, patience {cfg.gnn.early_stop_patience}", flush=True)
    print("=" * 72, flush=True)

    start = time.time()
    full = hls4ml.load_particle_level(cfg=cfg, n_jets=n_max, verbose=True)
    print(f"load: {time.time() - start:.0f} s\n", flush=True)

    total = len(args.sizes) * len(args.seeds)
    done = 0
    job_start = time.time()

    for size in sorted(args.sizes):
        data = subset(full, size, n_max)
        n_train = len(data["y_train"])
        test_loader = gnn.make_loader(data, "test", data["norm"], k=cfg.gnn.k_neighbors,
                                      batch_size=cfg.gnn.batch_size)

        for seed in args.seeds:
            done += 1
            tag = f"[{done}/{total}] size {size} seed {seed}"
            if not args.force and already_done(size, seed, cfg.artifact_dir):
                print(f"{tag}: already done, skipping", flush=True)
                continue

            ckpt = f"{cfg.checkpoint_dir}/gnn_n{size}_seed{seed}.pt"
            run_start = time.time()
            model, dev, info = gnn.train_gnn(data, cfg=cfg, seed=seed,
                                             checkpoint_path=ckpt, verbose=False)
            metrics = gnn.evaluate_gnn(model, dev, test_loader, data["class_names"],
                                       verbose=False)
            minutes = (time.time() - run_start) / 60

            # the sidecar is what results.scan() reports, so the notebook sees what finished
            results.save(np.array([metrics["per_class_auc"][c] for c in data["class_names"]]),
                         "performance", model="gnn", level=f"n{size}", seed=seed,
                         epochs=info["epochs"], n_rows=n_train,
                         n_features=len(data["node_features"]),
                         accuracy=metrics["accuracy"], val_acc=info["best_val_acc"],
                         epochs_run=info["epochs_run"], checkpoint=ckpt,
                         minutes=round(minutes, 1))

            elapsed = (time.time() - job_start) / 60
            print(f"{tag}: test acc {metrics['accuracy']:.4f} "
                  f"(val {info['best_val_acc']:.4f}, best epoch {info['epochs']} "
                  f"of {info['epochs_run']} run) in {minutes:.1f} min "
                  f"| elapsed {elapsed:.0f} min", flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"FINISHED {total} runs in {(time.time() - job_start) / 60:.0f} min", flush=True)
    summary = results.summarise(cfg.artifact_dir)
    print(summary.to_string(index=False), flush=True)
    (Path(cfg.log_dir) / "train_gnn_summary.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, default=str))


if __name__ == "__main__":
    main()
