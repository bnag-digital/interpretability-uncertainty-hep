import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import results
from protocols.ceiling import ceiling_from_stack
from protocols.cross import cross_explainer, verdict
from protocols.floor import floor_from_stack

JET = {"bdt": ["shap", "lime"], "dnn": ["shap", "lime"]}
PARTICLE = {m: ["saliency", "ig", "smoothgrad", "occlusion"] for m in ("gnn", "pfn", "efn")}
PAIRS = {**JET, **PARTICLE}

floors, ceilings = {}, {}

print(f"{'model':5s} {'explainer':10s} {'floor tau':>10s} {'raw spread':>11s} "
      f"{'ceiling tau':>12s} {'ceiling CI':>18s} {'feats':>6s}")
print("-" * 78)

for model, explainers in PAIRS.items():
    for explainer in explainers:
        try:
            f_stack, _ = results.stack_seeds(model, explainer, "floor")
            c_stack, seeds = results.stack_seeds(model, explainer, "ceiling")
        except Exception as exc:
            print(f"{model:5s} {explainer:10s} skipped: {type(exc).__name__}: {exc}")
            continue
        f = floor_from_stack(f_stack)
        c = ceiling_from_stack(c_stack)
        floors[(model, explainer)] = f
        ceilings[(model, explainer)] = c
        print(f"{model:5s} {explainer:10s} {f['mean_tau']:10.4f} "
              f"{f['max_raw_spread']:11.3e} {c['mean_tau']:12.4f} "
              f"[{c['ci_lower']:.4f}, {c['ci_upper']:.4f}] {c['n_features']:6d}")

print("\ncross-explainer, against the noise each pair has to beat")
print("-" * 78)
for model, explainers in PAIRS.items():
    for a, b in combinations(explainers, 2):
        if (model, a) not in ceilings or (model, b) not in ceilings:
            continue
        x = cross_explainer(results.stack_seeds(model, a, "ceiling")[0],
                            results.stack_seeds(model, b, "ceiling")[0],
                            model, a, b)
        v = verdict(x, ceilings[(model, a)], ceilings[(model, b)], floors[(model, a)])
        print(f"{model:5s} {a:9s} vs {b:10s} cross {x['mean_tau']:.4f} "
              f"[{x['ci_lower']:.4f}, {x['ci_upper']:.4f}]  "
              f"margin {v['margin']:+.4f}  {v['reading']}")
