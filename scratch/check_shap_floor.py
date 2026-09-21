import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import results
from metrics.rank import mean_abs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dnn")
    parser.add_argument("--explainer", default="shap")
    parser.add_argument("--dataset", default="covertype")
    parser.add_argument("--cls-idx", type=int, default=None)
    args = parser.parse_args()

    level = results.level_tag("floor", args.dataset)
    stack, seeds = results.stack_seeds(args.model, args.explainer, level)
    print(f"{args.model}/{args.explainer} floor on {args.dataset}: "
          f"{len(seeds)} explainer seeds, stack {stack.shape}")

    pairs = [(i, j) for i in range(len(stack)) for j in range(i + 1, len(stack))]
    raw = max(np.abs(stack[i] - stack[j]).max() for i, j in pairs)
    print(f"largest attribution difference between any two seeds: {raw:.3e}")
    if np.isclose(raw, 0.0):
        print("  -> identical values. The seed did not reach the sampling.")
    else:
        print("  -> values move with the seed, as a sampled explainer should")

    imps = np.stack([mean_abs(stack[i], args.cls_idx) for i in range(len(stack))])
    orders = {tuple(np.argsort(-imp)) for imp in imps}
    print(f"distinct feature orderings across seeds: {len(orders)}")
    share = imps[0] / imps[0].sum()
    top = np.argsort(-share)[:3]
    print(f"top-3 importance shares, seed 0: {', '.join(f'{share[i]:.3f}' for i in top)}")
    print(f"per-seed spread of the largest importance: "
          f"{imps[:, top[0]].min():.4e} to {imps[:, top[0]].max():.4e}")


if __name__ == "__main__":
    main()
