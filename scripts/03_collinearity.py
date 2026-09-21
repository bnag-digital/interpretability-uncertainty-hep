import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import results
from config import CONFIG, DATASET_NAMES, DEFAULT_DATASET
from data import has_set_valued_view, hls4ml, load_tabular_view
from metrics.rank import (collinearity, flips_against_collinearity, mean_abs,
                          pair_flip_rate)

JET_PAIRS = {"bdt": ["shap", "lime"],
             "dnn": ["shap", "lime", "saliency", "ig", "smoothgrad"]}
PARTICLE_PAIRS = {m: ["saliency", "ig", "smoothgrad", "occlusion"] for m in ("gnn", "pfn")}


def report_space(name, X, feature_names):
    """Print the correlation structure of one feature space."""
    out = collinearity(X, feature_names)
    print(f"\n{name}: {out['n_features']} features, {len(X)} rows")
    print(f"  max |r|            {out['max_abs_corr']:.3f}")
    print(f"  mean |r|           {out['mean_abs_corr']:.3f}")
    print(f"  pairs |r| >= 0.9   {out['frac_above_0.9'] * 100:.1f}%")
    print(f"  pairs |r| >= 0.7   {out['frac_above_0.7'] * 100:.1f}%")
    print(f"  condition number   {out['condition_number']:.1f}")
    print(f"  max VIF            {out['max_vif']:.1f}")
    print("  most correlated pairs:")
    for a, b, r in out["top_pairs"][:6]:
        print(f"    {r:+.3f}  {a} / {b}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jets", type=int, default=200000)
    parser.add_argument("--level", default="n200000")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default=DEFAULT_DATASET,
                        help="which dataset to measure. Only hls4ml has a particle view.")
    parser.add_argument("--cls-idx", type=int, default=None,
                        help="measure the flip rate on one class instead of the "
                             "average over all of them")
    parser.add_argument("--n-test-jets", type=int, default=None,
                        help="test block size the models were trained against, on a "
                             "dataset other than hls4ml")
    args = parser.parse_args()

    cfg = CONFIG
    cfg.make_dirs()
    cfg.data.name = args.dataset
    cfg.data.n_jets = args.n_jets
    if args.dataset == DEFAULT_DATASET:
        cfg.data.split_index_path = f"artifacts/split_gnn_{args.n_jets}.npz"
    ceiling_level = results.level_tag("ceiling", args.dataset)

    print("=" * 72)
    print(f"COLLINEARITY, dataset {args.dataset}")
    print("=" * 72)

    # correlate on training rows, which is what the model saw
    jet = load_tabular_view(args.dataset, cfg=cfg, n_rows=args.n_jets,
                            n_test_rows=args.n_test_jets, jet_source="h5", verbose=False)
    jet_corr = report_space(f"{args.dataset} tabular view", jet["X_train"],
                            jet["feature_names"])

    particle_corr = None
    if has_set_valued_view(args.dataset):
        particle = hls4ml.load_particle_level(cfg=cfg, n_jets=args.n_jets, verbose=False)
        real = particle["mask_train"].astype(bool)
        constituents = particle["X_train"][real]
        particle_corr = report_space("particle view", constituents, cfg.data.node_features)

    print("\n" + "=" * 72)
    print("DOES COLLINEARITY PREDICT WHICH FEATURES SWAP BETWEEN SEEDS?")
    print("=" * 72)
    print(f"{'model':5s} {'explainer':11s} {'rho':>7s} {'p':>8s} {'mean flip rate':>15s}")
    print("-" * 52)

    spaces = [(JET_PAIRS, jet_corr)]
    if particle_corr is not None:
        spaces.append((PARTICLE_PAIRS, particle_corr))

    rows = []
    for pairs, corr in spaces:
        for model, explainers in pairs.items():
            for explainer in explainers:
                try:
                    stack, seeds = results.stack_seeds(model, explainer, ceiling_level)
                except Exception as exc:
                    print(f"{model:5s} {explainer:11s} skipped: {type(exc).__name__}")
                    continue
                imps = np.stack([mean_abs(stack[i], args.cls_idx)
                                 for i in range(len(stack))])
                out = flips_against_collinearity(imps, corr["corr"])
                rows.append({"model": model, "explainer": explainer, "n_seeds": len(seeds),
                             **{k: v for k, v in out.items() if k != "defined"}})
                rho = f"{out['spearman_rho']:+.3f}" if out["defined"] else "n/a"
                p = f"{out['p_value']:.4f}" if out["defined"] else "n/a"
                print(f"{model:5s} {explainer:11s} {rho:>7s} {p:>8s} "
                      f"{out['mean_flip_rate']:15.3f}")

    # two features carrying the same information have no right order
    print("\n" + "=" * 72)
    print("REDUNDANT PAIRS: is the ordering arbitrary?")
    print("=" * 72)
    redundant = []
    named_spaces = [(args.dataset, jet_corr, JET_PAIRS)]
    if particle_corr is not None:
        named_spaces.append(("particle", particle_corr, PARTICLE_PAIRS))
    for space_name, corr, pairs in named_spaces:
        names = corr["feature_names"]
        c = np.abs(corr["corr"])
        idx = [(a, b) for a, b in zip(*np.triu_indices(len(names), k=1))
               if c[a, b] >= 0.999]
        if not idx:
            continue
        for a, b in idx:
            print(f"\n{space_name} view: {names[a]} / {names[b]}  |r| = {c[a, b]:.4f}")
            for model, explainers in pairs.items():
                for explainer in explainers:
                    try:
                        stack, _ = results.stack_seeds(model, explainer, ceiling_level)
                    except Exception:
                        continue
                    imps = np.stack([mean_abs(stack[i], args.cls_idx)
                                     for i in range(len(stack))])
                    flips = pair_flip_rate(imps)
                    iu = np.triu_indices(flips.shape[0], k=1)
                    print(f"  {model:4s} {explainer:11s} flip rate {flips[a, b]:.2f}   "
                          f"(all pairs mean {flips[iu].mean():.3f})")
                    redundant.append({"space": space_name, "pair": [names[a], names[b]],
                                      "model": model, "explainer": explainer,
                                      "flip_rate": float(flips[a, b]),
                                      "mean_flip_rate": float(flips[iu].mean())})

    defined = [r for r in rows if not np.isnan(r["spearman_rho"])]
    if defined:
        rhos = np.array([r["spearman_rho"] for r in defined])
        print(f"\n{len(defined)} measurements, rho mean {rhos.mean():+.3f}, "
              f"{int((rhos > 0).sum())} positive, {int((rhos < 0).sum())} negative")

    summary = {
        "dataset": args.dataset,
        "cls_idx": args.cls_idx,
        "jet_view": {k: v for k, v in jet_corr.items()
                     if k not in ("corr", "vif", "feature_names")},
        "flips_vs_collinearity": rows,
        "redundant_pairs": redundant,
    }
    if particle_corr is not None:
        summary["particle_view"] = {k: v for k, v in particle_corr.items()
                                    if k not in ("corr", "vif", "feature_names")}
    stem = "collinearity" if args.dataset == DEFAULT_DATASET else f"collinearity_{args.dataset}"
    out_path = Path(cfg.log_dir) / f"{stem}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
