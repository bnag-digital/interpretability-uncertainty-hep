import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots import variance

PAIRS = {"bdt": ["shap", "lime"],
         "dnn": ["shap", "lime", "saliency", "ig", "smoothgrad"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="covertype")
    parser.add_argument("--cls-idx", type=int, default=None)
    args = parser.parse_args()

    print(f"dataset {args.dataset}, "
          f"{'class ' + str(args.cls_idx) if args.cls_idx is not None else 'all classes'}")

    print("\nFLOOR AND CEILING")
    print(f"{'model':5s} {'explainer':11s} {'floor':>8s} {'ceiling':>9s} {'feats':>6s} {'seeds':>6s}")
    print("-" * 50)
    for row in variance.collect_levels(PAIRS, dataset=args.dataset, cls_idx=args.cls_idx):
        floor = f"{row['floor_tau']:.4f}" if row["floor_tau"] is not None else "1.0000*"
        print(f"{row['model']:5s} {row['explainer']:11s} {floor:>8s} "
              f"{row['ceiling_tau']:9.4f} {row['n_features']:6d} {row['n_seeds']:6d}")
    print("* deterministic, no floor measured")

    print("\nCROSS-EXPLAINER AGAINST THE CEILING")
    print(f"{'model':5s} {'pair':22s} {'cross':>8s} {'90% CI':>18s} {'margin':>8s}  verdict")
    print("-" * 78)
    for row in variance.collect_verdicts(PAIRS, dataset=args.dataset, cls_idx=args.cls_idx):
        ci = f"[{row['cross_lo']:.3f}, {row['cross_hi']:.3f}]"
        pair = f"{row['a']} vs {row['b']}"
        margin = row["margin"]
        verdict = "clears" if row["below_ceiling"] else "INSIDE NOISE"
        print(f"{row['model']:5s} {pair:22s} {row['cross_tau']:8.4f} {ci:>18s} "
              f"{margin:+8.4f}  {verdict}")


if __name__ == "__main__":
    main()
