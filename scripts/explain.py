import argparse
import hashlib
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
from explainers import setvalued, tabular
from models import bdt as bdt_mod
from models import dnn as dnn_mod
from models import efn as efn_mod
from models import gnn as gnn_mod
from models import pfn as pfn_mod

JET_MODELS = ("bdt", "dnn")
PARTICLE_MODELS = ("gnn", "pfn", "efn")
ALL_MODELS = JET_MODELS + PARTICLE_MODELS

JET_EXPLAINERS = ("shap", "lime")
# the DNN is differentiable, so it takes the gradient methods as well
DNN_EXPLAINERS = ("shap", "lime", "saliency", "ig", "smoothgrad")
PARTICLE_EXPLAINERS = ("saliency", "ig", "smoothgrad", "occlusion")
ALL_EXPLAINERS = tuple(dict.fromkeys(JET_EXPLAINERS + DNN_EXPLAINERS + PARTICLE_EXPLAINERS))

EXPLAINERS_FOR = {"bdt": JET_EXPLAINERS, "dnn": DNN_EXPLAINERS,
                  "gnn": PARTICLE_EXPLAINERS, "pfn": PARTICLE_EXPLAINERS,
                  "efn": PARTICLE_EXPLAINERS}

PARTICLE_LOADERS = {"gnn": gnn_mod.load_gnn, "pfn": pfn_mod.load_pfn, "efn": efn_mod.load_efn}

GRADIENT_EXPLAINERS = {"saliency": tabular.explain_dnn_saliency,
                       "ig": tabular.explain_dnn_ig,
                       "smoothgrad": tabular.explain_dnn_smoothgrad}


def feature_space_id(names):
    """Short id for one feature axis. tau is not comparable across two of these."""
    digest = hashlib.sha1("|".join(names).encode()).hexdigest()[:8]
    return f"{len(names)}f-{digest}"


def already_done(model, explainer, level, seed, artifact_dir):
    name = results.artifact_name("attribution", model, explainer, level, seed)
    return (Path(artifact_dir) / f"{name}.json").exists()


def resolve_level(df, model, override=None, dataset=DEFAULT_DATASET):
    """Which training level a model's checkpoints were written at, for one dataset."""
    if override:
        return override
    sub = results.select(df, kind="performance", model=model)
    found = sorted(str(x) for x in sub["level"].dropna().unique()) if not sub.empty else []
    levels = [x for x in found if results.level_dataset(x) == dataset]
    if not levels:
        raise SystemExit(f"{model} has no performance artifacts on {dataset}, so it has "
                         f"not been trained there. "
                         f"Run: pixi run train --models {model} --dataset {dataset}")
    if len(levels) > 1:
        raise SystemExit(f"{model} was trained at levels {levels}. "
                         f"Pass --train-level to say which one to explain.")
    return levels[0]


def jet_source_of(df, model, level=None, default="openml"):
    """Which loader trained this model, taken from its performance sidecar."""
    sub = results.select(df, kind="performance", model=model)
    if level is not None:
        sub = results.select(sub, level=level)
    if sub.empty or "jet_source" not in sub.columns:
        return default
    found = sorted(str(x) for x in sub["jet_source"].dropna().unique())
    if len(found) > 1:
        raise SystemExit(f"{model} has artifacts from jet sources {found}. "
                         "Explaining them together would mix two sets of jets.")
    return found[0] if found else default


def n_test_rows_of(df, model, level):
    """The test-block size a model was trained against."""
    sub = results.select(df, kind="performance", model=model, level=level)
    if sub.empty or "n_test_rows" not in sub.columns:
        return None
    found = sorted({int(x) for x in sub["n_test_rows"].dropna().unique()})
    if len(found) > 1:
        raise SystemExit(f"{model} at level {level} was trained against test blocks "
                         f"of {found} rows. Those are different test sets.")
    return found[0] if found else None


def n_jets_of(level):
    """The --n-jets that produced a level name, so the same rows load back."""
    bare = results.bare_level(level)
    return None if bare == "full" else int(bare.lstrip("n"))


def checkpoint_path(cfg, model, level, seed):
    ext = "json" if model == "bdt" else "pt"
    return f"{cfg.checkpoint_dir}/{model}_{level}_seed{seed}.{ext}"


def pick_rows(n_available, n_rows, cfg):
    """The same jets every run, so seeds and models are compared on one sample."""
    rng = np.random.default_rng(cfg.data.split_seed)
    n = min(n_rows, n_available)
    return np.sort(rng.choice(n_available, size=n, replace=False))


def torch_proba(model, device, batch_size=1024):
    """predict_proba for a torch model, which is what LIME perturbs against."""
    def predict_proba(x):
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = torch.tensor(np.asarray(x[start:start + batch_size]),
                                     dtype=torch.float32).to(device)
                out.append(torch.softmax(model(batch), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)
    return predict_proba


def explain_jet(model_name, explainer_name, model, device, data, rows, cfg, seed, verbose):
    """One attribution array for a jet-level model, (rows, features, classes)."""
    feature_names, class_names = data["feature_names"], data["class_names"]
    n_classes = len(class_names)

    if model_name == "bdt":
        X_explain, X_train = data["X_test"][rows], data["X_train"]
        predict_proba = model.predict_proba
    else:
        X_explain, X_train = data["X_test_sc"][rows], data["X_train_sc"]
        predict_proba = torch_proba(model, device)

    if explainer_name == "shap":
        if model_name == "bdt":
            return tabular.explain_bdt_shap(model, X_explain, feature_names, n_classes,
                                            cfg=cfg, seed=seed, verbose=verbose)
        return tabular.explain_dnn_shap(model, device, X_explain, X_train, feature_names,
                                        n_classes, cfg=cfg, seed=seed, verbose=verbose)

    if explainer_name in GRADIENT_EXPLAINERS:
        if model_name != "dnn":
            raise ValueError(f"{explainer_name} needs a differentiable model and "
                             f"{model_name} is not one")
        return GRADIENT_EXPLAINERS[explainer_name](
            model, device, X_explain, feature_names, n_classes, cfg=cfg, seed=seed,
            verbose=verbose)

    lime = tabular.build_lime_explainer(X_train, feature_names, class_names, seed=seed)
    return tabular.explain_lime(lime, X_explain, predict_proba, len(feature_names),
                                n_classes, verbose=verbose)


def predicted_classes(model_name, model, device, jet_data, rows, inputs, cfg):
    """Predicted class per explained row, for the mechanism protocol."""
    if model_name == "bdt":
        return model.predict_proba(jet_data["X_test"][rows]).argmax(axis=1)
    if model_name == "dnn":
        return torch_proba(model, device)(jet_data["X_test_sc"][rows]).argmax(axis=1)
    k = cfg.gnn.k_neighbors if model_name == "gnn" else None
    return setvalued.predict_logits(model, model_name, inputs, device, k=k).argmax(axis=1)


def explain_particle(explainer_name, model, kind, inputs, device, cfg, seed, verbose):
    """One attribution array for a set-valued model, (rows, features)."""
    k = cfg.gnn.k_neighbors if kind == "gnn" else None
    if explainer_name == "saliency":
        return setvalued.saliency(model, kind, inputs, device, k=k, verbose=verbose)
    if explainer_name == "ig":
        return setvalued.integrated_gradients(model, kind, inputs, device, cfg=cfg,
                                              k=k, verbose=verbose)
    if explainer_name == "smoothgrad":
        return setvalued.smoothgrad(model, kind, inputs, device, cfg=cfg, seed=seed,
                                    k=k, verbose=verbose)
    return setvalued.feature_occlusion(model, kind, inputs, device, k=k, verbose=verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS)
    parser.add_argument("--explainers", nargs="+", default=None, choices=ALL_EXPLAINERS,
                        help="default: shap and lime for jet models, the three "
                             "gradient and occlusion methods for set-valued ones")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="model seeds, one ceiling attribution each")
    parser.add_argument("--train-level", default=None,
                        help="override the level resolved from the performance sidecars")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default=DEFAULT_DATASET,
                        help="which dataset's checkpoints to explain. Must match the "
                             "--dataset the models were trained with.")
    parser.add_argument("--n-rows", type=int, default=None,
                        help="test jets to explain, default cfg.explainer.n_rows")
    parser.add_argument("--explainer-seed", type=int, default=42,
                        help="held fixed for every ceiling run")
    parser.add_argument("--floor-repeats", type=int, default=0,
                        help="explainer seeds to run against --floor-model-seed. "
                             "0 writes no floor level.")
    parser.add_argument("--floor-model-seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="redo runs that already finished")
    parser.add_argument("--time-only", action="store_true",
                        help="report wall clock and write no artifacts")
    parser.add_argument("--quiet", action="store_true", help="no per-batch progress")
    args = parser.parse_args()

    cfg = CONFIG
    cfg.make_dirs()
    n_rows = args.n_rows or cfg.explainer.n_rows
    verbose = not args.quiet
    wanted = [m for m in ALL_MODELS if m in args.models]
    if not has_set_valued_view(args.dataset):
        refused = [m for m in wanted if m in PARTICLE_MODELS]
        if refused:
            raise SystemExit(
                f"{args.dataset} has no set-valued view, so {refused} have no "
                f"checkpoints there. Pass --models bdt dnn.")
    cfg.data.name = args.dataset

    scanned = results.scan(cfg.artifact_dir)
    levels = {m: resolve_level(scanned, m, args.train_level, args.dataset) for m in wanted}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72, flush=True)
    print(f"EXPLAIN {wanted}  model seeds {args.seeds}  rows {n_rows}  "
          f"dataset {args.dataset}", flush=True)
    print(f"levels {levels}  output_space {cfg.explainer.output_space}"
          f"{'  [TIMING ONLY, no artifacts]' if args.time_only else ''}", flush=True)
    print("=" * 72, flush=True)

    # each view is loaded once, and only if a model needs it
    jet_data = particle_data = None
    jet_rows = particle_rows = None
    if any(m in JET_MODELS for m in wanted):
        start = time.time()
        jet_model = next(m for m in wanted if m in JET_MODELS)
        level = levels[jet_model]
        n_jets = n_jets_of(level)
        source = jet_source_of(scanned, jet_model, level)
        cfg.data.n_jets = n_jets
        if args.dataset == DEFAULT_DATASET:
            # load_jet_level appends _jet, the split the BDT and DNN trained on
            cfg.data.split_index_path = f"artifacts/split_gnn_{n_jets or 'full'}.npz"
        n_test = (cfg.data.n_test_jets if args.dataset == DEFAULT_DATASET
                  else n_test_rows_of(scanned, jet_model, level))
        jet_data = load_tabular_view(args.dataset, cfg=cfg, n_rows=n_jets,
                                     n_test_rows=n_test, jet_source=source, verbose=True)
        jet_rows = pick_rows(len(jet_data["y_test"]), n_rows, cfg)
        print(f"tabular load: {time.time() - start:.0f} s, "
              f"{len(jet_rows)} of {len(jet_data['y_test'])} test rows", flush=True)
    if any(m in PARTICLE_MODELS for m in wanted):
        start = time.time()
        level = levels[next(m for m in wanted if m in PARTICLE_MODELS)]
        n_jets = n_jets_of(level)
        cfg.data.n_jets = n_jets
        cfg.data.split_index_path = f"artifacts/split_gnn_{n_jets or 'full'}.npz"
        particle_data = hls4ml.load_particle_level(cfg=cfg, n_jets=n_jets, verbose=True)
        particle_rows = pick_rows(len(particle_data["y_test"]), n_rows, cfg)
        print(f"particle-level load: {time.time() - start:.0f} s, "
              f"{len(particle_rows)} of {len(particle_data['y_test'])} test jets", flush=True)

    # the protocol level carries the dataset, so two datasets cannot share a filename
    ceiling_level = results.level_tag("ceiling", args.dataset)
    floor_level = results.level_tag("floor", args.dataset)

    # (model, explainer, protocol level, artifact seed, explainer seed)
    jobs = []
    for model_name in wanted:
        family = EXPLAINERS_FOR[model_name]
        chosen = [e for e in family if args.explainers is None or e in args.explainers]
        for explainer_name in chosen:
            for seed in args.seeds:
                jobs.append((model_name, explainer_name, ceiling_level, seed,
                             args.explainer_seed))
            for repeat in range(args.floor_repeats):
                jobs.append((model_name, explainer_name, floor_level, repeat, repeat))

    total, done, timings = len(jobs), 0, []
    job_start = time.time()

    for model_name, explainer_name, level, seed, explainer_seed in jobs:
        done += 1
        model_seed = args.floor_model_seed if level == floor_level else seed
        tag = f"[{done}/{total}] {model_name} {explainer_name} {level} seed {seed}"
        if not args.force and not args.time_only \
                and already_done(model_name, explainer_name, level, seed, cfg.artifact_dir):
            print(f"{tag}: already done, skipping", flush=True)
            continue

        train_level = levels[model_name]
        ckpt = checkpoint_path(cfg, model_name, train_level, model_seed)
        if not Path(ckpt).exists():
            print(f"{tag}: no checkpoint at {ckpt}, skipping", flush=True)
            continue

        run_start = time.time()
        if model_name in JET_MODELS:
            if model_name == "bdt":
                model, _ = bdt_mod.load_bdt(ckpt)
                model_device = None
            else:
                model, model_device = dnn_mod.load_dnn(ckpt, device)
            arr = explain_jet(model_name, explainer_name, model, model_device, jet_data,
                              jet_rows, cfg, explainer_seed, verbose)
            names = jet_data["feature_names"]
            n_explained = len(jet_rows)
            rows_here, inputs = jet_rows, None
        else:
            model, model_device = PARTICLE_LOADERS[model_name](ckpt, device)
            inputs = setvalued.prepare_inputs(model_name, particle_data, "test", cfg,
                                              rows=particle_rows)
            arr = explain_particle(explainer_name, model, model_name, inputs, model_device,
                                   cfg, explainer_seed, verbose)
            names = inputs["feature_names"]
            n_explained = len(particle_rows)
            rows_here = particle_rows

        # one per model seed is enough, and the mechanism protocol needs these rows
        pred_name = results.artifact_name("prediction", model_name, None, train_level,
                                          model_seed)
        if not args.time_only and not (Path(cfg.artifact_dir) / f"{pred_name}.json").exists():
            results.save(
                predicted_classes(model_name, model, model_device, jet_data, rows_here,
                                  inputs, cfg),
                "prediction", model=model_name, level=train_level, seed=model_seed,
                n_rows=n_explained, checkpoint=ckpt)

        minutes = (time.time() - run_start) / 60
        timings.append({"model": model_name, "explainer": explainer_name, "level": level,
                        "seed": seed, "minutes": round(minutes, 2)})

        if not args.time_only:
            results.save(arr, "attribution", model=model_name, explainer=explainer_name,
                         level=level, seed=seed, n_rows=n_explained, n_features=len(names),
                         feature_space_id=feature_space_id(names), feature_names=list(names),
                         output_space=cfg.explainer.output_space,
                         train_level=train_level, model_seed=model_seed,
                         explainer_seed=explainer_seed, checkpoint=ckpt,
                         minutes=round(minutes, 1))

        elapsed = (time.time() - job_start) / 60
        print(f"{tag}: {arr.shape} in {minutes:.1f} min | elapsed {elapsed:.0f} min", flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"FINISHED {len(timings)} runs in {(time.time() - job_start) / 60:.0f} min", flush=True)
    for row in timings:
        print(f"  {row['model']:4s} {row['explainer']:9s} {row['level']:7s} "
              f"seed {row['seed']}: {row['minutes']:6.2f} min", flush=True)

    out = Path(cfg.log_dir) / "explain_timings.json"
    out.write_text(json.dumps(timings, indent=2, default=str))
    print(f"\ntimings written to {out}", flush=True)
    if not args.time_only:
        print(results.summarise(cfg.artifact_dir).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
