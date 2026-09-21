import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_DATASET
from plots.interaction import JET_SQUARE, collect_square

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=DEFAULT_DATASET)
parser.add_argument("--cls-idx", type=int, default=None)
args = parser.parse_args()

out = collect_square(JET_SQUARE, dataset=args.dataset, cls_idx=args.cls_idx)
model_a, model_b, expl_a, expl_b = out["square"]
means = out["means"]

n_seeds = len(out["comparisons"]["cross_model_a"]["taus"])
print(f"dataset {args.dataset}, square {model_a}/{model_b} x {expl_a}/{expl_b}, "
      f"{n_seeds} seeds")
print(f"  {model_a} vs {model_b} under {expl_a}: {means['cross_model_shap']:.4f}")
print(f"  {model_a} vs {model_b} under {expl_b}: {means['cross_model_lime']:.4f}")
print(f"  {expl_a} vs {expl_b} on {model_a}:     {means['cross_explainer_a']:.4f}")
print(f"  {expl_a} vs {expl_b} on {model_b}:     {means['cross_explainer_b']:.4f}")
print(f"  explainer band {out['explainer_band'][0]:.4f} to "
      f"{out['explainer_band'][1]:.4f}, straddled: {out['reverses']}")
for key in ("shap_test", "lime_test"):
    test = out[key]
    print(f"  {key}: p = {test['p_permutation']:.3f}, observed diff "
          f"{test['observed_diff']:+.4f}")
