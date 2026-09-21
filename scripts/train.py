import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import results
from config import CONFIG, DATASET_NAMES, DEFAULT_DATASET
from data import has_set_valued_view, hls4ml, load_tabular_view
from models import bdt as bdt_mod
from models import dnn as dnn_mod
from models import efn as efn_mod
from models import gnn as gnn_mod
from models import pfn as pfn_mod

JET_MODELS = ("bdt", "dnn")
PARTICLE_MODELS = ("gnn", "pfn", "efn")
ALL_MODELS = JET_MODELS + PARTICLE_MODELS


def already_done(model, level, seed, artifact_dir):
    name = results.artifact_name("performance", model, None, level, seed)
    return (Path(artifact_dir) / f"{name}.json").exists()


def train_bdt_seed(data, cfg, seed, ckpt):
    model, info = bdt_mod.train_bdt(data["X_train"], data["y_train"],
                                    data["X_val"], data["y_val"], cfg=cfg,
                                    n_classes=len(data["class_names"]), seed=seed,
                                    checkpoint_path=ckpt, verbose=False)
    metrics = bdt_mod.evaluate_bdt(model, data["X_test"], data["y_test"],
                                   data["class_names"], verbose=False)
    return info, metrics, len(data["y_train"]), len(data["feature_names"])


def train_dnn_seed(data, cfg, seed, ckpt):
    model, device, info = dnn_mod.train_dnn(
        data["X_train_sc"], data["y_train"], data["X_val_sc"], data["y_val"],
        cfg=cfg, n_classes=len(data["class_names"]), seed=seed,
        checkpoint_path=ckpt, norm=data["norm"],
        feature_names=data["feature_names"], verbose=False)
    metrics = dnn_mod.evaluate_dnn(model, device, data["X_test_sc"], data["y_test"],
                                   data["class_names"], verbose=False)
    return info, metrics, len(data["y_train"]), len(data["feature_names"])


def train_gnn_seed(data, cfg, seed, ckpt, test_loader):
    model, device, info = gnn_mod.train_gnn(data, cfg=cfg, seed=seed,
                                            checkpoint_path=ckpt, verbose=False)
    metrics = gnn_mod.evaluate_gnn(model, device, test_loader, data["class_names"],
                                   verbose=False)
    return info, metrics, len(data["y_train"]), len(data["node_features"])


def train_pfn_seed(data, cfg, seed, ckpt, test_loader):
    model, device, info = pfn_mod.train_pfn(data, cfg=cfg, seed=seed,
                                            checkpoint_path=ckpt, verbose=False)
    metrics = pfn_mod.evaluate_pfn(model, device, test_loader, data["class_names"],
                                   verbose=False)
    return info, metrics, len(data["y_train"]), len(data["node_features"])


def train_efn_seed(data, cfg, seed, ckpt, test_loader):
    model, device, info = efn_mod.train_efn(data, cfg=cfg, seed=seed,
                                            checkpoint_path=ckpt, verbose=False)
    metrics = efn_mod.evaluate_efn(model, device, test_loader, data["class_names"],
                                   verbose=False)
    return info, metrics, len(data["y_train"]), len(data["node_features"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-jets", type=int, default=None,
                        help="cap on jets read. Pass the same value the GNN grid used "
                             "so PFN and EFN share its split.")
    parser.add_argument("--n-test-jets", type=int, default=None,
                        help="rows held out for testing. Default 50000 on hls4ml, "
                             "and the dataset's own test fraction otherwise.")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default=DEFAULT_DATASET,
                        help="which data the tabular models read. Only hls4ml has a "
                             "set-valued view, so the others take bdt and dnn.")
    parser.add_argument("--force", action="store_true", help="retrain runs that already finished")
    parser.add_argument("--patience", type=int, default=None,
                        help="override early stopping patience on every model that has one")
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="override the BDT's tree cap. Raise it when a run reports a "
                             "best iteration equal to the cap, which means the model was "
                             "still improving when it ran out of trees.")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="override the BDT's tree depth. The default suits the 16 jet "
                             "observables; a dataset with deeper structure needs more.")
    parser.add_argument("--hidden", type=int, nargs="+", default=None,
                        help="override the DNN's layer widths. The default was sized for "
                             "16 jet observables and is small for anything wider.")
    parser.add_argument("--lr", type=float, default=None,
                        help="override the DNN's learning rate")
    parser.add_argument("--jet-source", choices=("openml", "h5"), default="openml",
                        help="where BDT and DNN read their 16 observables. h5 shares "
                             "the particle view's jets, openml does not.")
    parser.add_argument("--time-only", action="store_true",
                        help="report per-seed wall clock and write no artifacts")
    args = parser.parse_args()

    cfg = CONFIG
    cfg.make_dirs()
    if args.patience is not None:
        for name in ALL_MODELS:
            sub = getattr(cfg, name, None)
            if hasattr(sub, "early_stop_patience"):
                sub.early_stop_patience = args.patience
    if args.n_estimators is not None:
        cfg.bdt.n_estimators = args.n_estimators
    if args.max_depth is not None:
        cfg.bdt.max_depth = args.max_depth
    if args.hidden is not None:
        cfg.dnn.hidden = list(args.hidden)
    if args.lr is not None:
        cfg.dnn.lr = args.lr

    wanted = [m for m in ALL_MODELS if m in args.models]
    if not has_set_valued_view(args.dataset):
        refused = [m for m in wanted if m in PARTICLE_MODELS]
        if refused:
            raise SystemExit(
                f"{args.dataset} has no set-valued view, so {refused} cannot be trained "
                f"on it. Run those on hls4ml, or pass --models bdt dnn.")

    n_test = args.n_test_jets
    if n_test is None and args.dataset == DEFAULT_DATASET:
        n_test = 50000
    cfg.data.name = args.dataset
    cfg.data.n_jets = args.n_jets
    cfg.data.n_test_jets = n_test
    level = results.level_tag(f"n{args.n_jets}" if args.n_jets else "full", args.dataset)
    if args.dataset == DEFAULT_DATASET:
        # matches 02_train_gnn_seeds.py, so the same --n-jets reuses its split
        cfg.data.split_index_path = f"artifacts/split_gnn_{args.n_jets or 'full'}.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72, flush=True)
    print(f"TRAIN {wanted}  seeds {args.seeds}  dataset {args.dataset}  level {level}"
          f"{f'  patience {args.patience}' if args.patience else ''}", flush=True)
    source_note = (f", jet source {args.jet_source}, split {cfg.data.split_index_path}"
                   if args.dataset == DEFAULT_DATASET else "")
    print(f"device {device}{source_note}"
          f"{'  [TIMING ONLY, no artifacts]' if args.time_only else ''}", flush=True)
    print("=" * 72, flush=True)

    jet_data = particle_data = None
    loaders = {}
    if any(m in JET_MODELS for m in wanted):
        start = time.time()
        jet_data = load_tabular_view(args.dataset, cfg=cfg, n_rows=args.n_jets,
                                     n_test_rows=n_test, jet_source=args.jet_source,
                                     verbose=True)
        print(f"tabular load: {time.time() - start:.0f} s", flush=True)
    if any(m in PARTICLE_MODELS for m in wanted):
        start = time.time()
        particle_data = hls4ml.load_particle_level(cfg=cfg, n_jets=args.n_jets, verbose=True)
        print(f"particle-level load: {time.time() - start:.0f} s", flush=True)
        norm = particle_data["norm"]
        if "gnn" in wanted:
            loaders["gnn"] = gnn_mod.make_loader(particle_data, "test", norm,
                                                 k=cfg.gnn.k_neighbors,
                                                 batch_size=cfg.gnn.batch_size)
        if "pfn" in wanted:
            loaders["pfn"] = pfn_mod.make_loader(particle_data, "test", norm,
                                                 batch_size=cfg.pfn.batch_size)
        if "efn" in wanted:
            loaders["efn"] = efn_mod.make_loader(particle_data, "test", norm, cfg.efn,
                                                 batch_size=cfg.efn.batch_size)

    total, done, timings = len(wanted) * len(args.seeds), 0, []
    job_start = time.time()

    for model_name in wanted:
        data = jet_data if model_name in JET_MODELS else particle_data
        for seed in args.seeds:
            done += 1
            tag = f"[{done}/{total}] {model_name} seed {seed}"
            if not args.force and not args.time_only \
                    and already_done(model_name, level, seed, cfg.artifact_dir):
                print(f"{tag}: already done, skipping", flush=True)
                continue

            ckpt_ext = "json" if model_name == "bdt" else "pt"
            ckpt = f"{cfg.checkpoint_dir}/{model_name}_{level}_seed{seed}.{ckpt_ext}"
            run_start = time.time()

            if model_name == "bdt":
                info, metrics, n_train, n_features = train_bdt_seed(data, cfg, seed, ckpt)
            elif model_name == "dnn":
                info, metrics, n_train, n_features = train_dnn_seed(data, cfg, seed, ckpt)
            elif model_name == "gnn":
                info, metrics, n_train, n_features = train_gnn_seed(data, cfg, seed, ckpt,
                                                                    loaders["gnn"])
            elif model_name == "pfn":
                info, metrics, n_train, n_features = train_pfn_seed(data, cfg, seed, ckpt,
                                                                    loaders["pfn"])
            else:
                info, metrics, n_train, n_features = train_efn_seed(data, cfg, seed, ckpt,
                                                                    loaders["efn"])

            minutes = (time.time() - run_start) / 60
            timings.append({"model": model_name, "seed": seed, "minutes": round(minutes, 2),
                            "epochs": info["epochs"], "epochs_run": info.get("epochs_run"),
                            "n_train": n_train, "accuracy": metrics["accuracy"]})

            if not args.time_only:
                results.save(
                    np.array([metrics["per_class_auc"][c] for c in data["class_names"]]),
                    "performance", model=model_name, level=level, seed=seed,
                    epochs=info["epochs"], n_rows=n_train, n_features=n_features,
                    class_names=list(data["class_names"]), n_test_rows=n_test,
                    n_estimators=(cfg.bdt.n_estimators if model_name == "bdt" else None),
                    max_depth=(cfg.bdt.max_depth if model_name == "bdt" else None),
                    hidden=(list(cfg.dnn.hidden) if model_name == "dnn" else None),
                    lr=(cfg.dnn.lr if model_name == "dnn" else None),
                    patience=(cfg.dnn.early_stop_patience if model_name == "dnn" else None),
                    accuracy=metrics["accuracy"],
                    val_score=info.get("best_val_acc", info.get("best_val_loss",
                                                                info.get("best_val_mlogloss"))),
                    epochs_run=info.get("epochs_run"), checkpoint=ckpt,
                    jet_source=(args.jet_source if model_name in JET_MODELS
                                and args.dataset == DEFAULT_DATASET else None),
                    minutes=round(minutes, 1))

            elapsed = (time.time() - job_start) / 60
            print(f"{tag}: test acc {metrics['accuracy']:.4f} "
                  f"(best epoch {info['epochs']} of {info.get('epochs_run', '?')} run) "
                  f"in {minutes:.1f} min | elapsed {elapsed:.0f} min", flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"FINISHED {len(timings)} runs in {(time.time() - job_start) / 60:.0f} min", flush=True)
    for row in timings:
        print(f"  {row['model']:4s} seed {row['seed']}: {row['minutes']:6.2f} min, "
              f"best epoch {row['epochs']} of {row['epochs_run']} run, "
              f"{row['n_train']} train jets, test acc {row['accuracy']:.4f}", flush=True)

    out = Path(cfg.log_dir) / f"train_timings_{level}.json"
    out.write_text(json.dumps(timings, indent=2, default=str))
    print(f"\ntimings written to {out}", flush=True)
    if not args.time_only:
        print(results.summarise(cfg.artifact_dir).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
