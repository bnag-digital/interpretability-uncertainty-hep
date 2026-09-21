import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import results
from config import DEFAULT_DATASET
from metrics.rank import mean_abs, tau_between
from plots import variance
from scripts.make_figures import JET_PAIRS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=DEFAULT_DATASET)
parser.add_argument("--n-rows", type=int, default=500)
parser.add_argument("--reps", type=int, default=200)
args = parser.parse_args()

pairs = None if args.dataset == DEFAULT_DATASET else JET_PAIRS
level = results.level_tag("ceiling", args.dataset)
rng = np.random.default_rng(0)


def row_tau(arr, n, reps):
    """tau between two independent n-row resamples of the same attributions."""
    taus = []
    for _ in range(reps):
        ia = rng.integers(0, len(arr), size=n)
        ib = rng.integers(0, len(arr), size=n)
        taus.append(tau_between(mean_abs(arr[ia]), mean_abs(arr[ib])))
    return np.array(taus)


print("dataset %s, row sampling at n=%d over %d resamples"
      % (args.dataset, args.n_rows, args.reps))
print("%-5s %-11s %8s %10s %8s %9s %9s"
      % ("model", "explainer", "floor", "rowsample", "ceiling", "row-ceil", "feats"))

for r in variance.collect_levels(pairs, dataset=args.dataset):
    try:
        arr, _ = results.load(results.artifact_name(
            "attribution", r["model"], r["explainer"], level, 0))
    except FileNotFoundError:
        print("%-5s %-11s  no seed-0 artifact" % (r["model"], r["explainer"]))
        continue
    t = row_tau(arr, args.n_rows, args.reps)
    floor = "%.4f" % r["floor_tau"] if r["floor_tau"] is not None else "     -"
    print("%-5s %-11s %8s %10.4f %8.4f %+9.4f %9d"
          % (r["model"], r["explainer"], floor, t.mean(), r["ceiling_tau"],
             t.mean() - r["ceiling_tau"], r["n_features"]))
