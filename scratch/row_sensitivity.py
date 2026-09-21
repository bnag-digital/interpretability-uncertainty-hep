import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import results
from metrics.rank import mean_abs, tau_between

PAIRS = [("bdt", "shap"), ("bdt", "lime"), ("dnn", "shap"), ("dnn", "lime"),
         ("gnn", "saliency"), ("gnn", "ig"), ("gnn", "occlusion"),
         ("pfn", "saliency"), ("pfn", "ig"), ("pfn", "occlusion")]
SIZES = [125, 250, 500, 1000]
B = 200


def row_tau(arr, n, rng, b=B):
    """tau between two independent n-row resamples of the same attributions."""
    taus = []
    for _ in range(b):
        ia = rng.integers(0, len(arr), size=n)
        ib = rng.integers(0, len(arr), size=n)
        taus.append(tau_between(mean_abs(arr[ia]), mean_abs(arr[ib])))
    return np.array(taus)


rng = np.random.default_rng(0)
print(f"{'model':5s} {'explainer':10s} " + " ".join(f"{'n=' + str(n):>16s}" for n in SIZES))
print("-" * 78)

for model, explainer in PAIRS:
    try:
        arr, meta = results.load(
            results.artifact_name("attribution", model, explainer, "ceiling", 0))
    except FileNotFoundError:
        print(f"{model:5s} {explainer:10s} missing")
        continue
    cells = []
    for n in SIZES:
        t = row_tau(arr, n, rng)
        cells.append(f"{t.mean():.3f} [{np.percentile(t, 2.5):.2f}]")
    print(f"{model:5s} {explainer:10s} " + " ".join(f"{c:>16s}" for c in cells))

print("\nmean tau between two independent draws of n rows, with the 2.5th percentile.")
print("Compare against the retraining ceilings: bdt/shap 0.959, dnn/lime 0.830,")
print("gnn/occlusion 0.950, pfn/occlusion 0.858.")

# does the cross-explainer result survive row resampling?
print("\ncross-explainer under row resampling, model seed 0")
print("-" * 78)
for model, a, b in [("bdt", "shap", "lime"), ("dnn", "shap", "lime"),
                    ("gnn", "ig", "occlusion"), ("pfn", "ig", "occlusion")]:
    arr_a, _ = results.load(results.artifact_name("attribution", model, a, "ceiling", 0))
    arr_b, _ = results.load(results.artifact_name("attribution", model, b, "ceiling", 0))
    taus = []
    for _ in range(B):
        idx = rng.integers(0, len(arr_a), size=len(arr_a))
        taus.append(tau_between(mean_abs(arr_a[idx]), mean_abs(arr_b[idx])))
    taus = np.array(taus)
    print(f"{model:5s} {a:9s} vs {b:10s} cross {taus.mean():.4f} "
          f"[{np.percentile(taus, 2.5):.4f}, {np.percentile(taus, 97.5):.4f}]")
