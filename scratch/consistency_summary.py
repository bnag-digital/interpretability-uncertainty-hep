import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots import reproducibility, variance
from scripts.make_figures import JET_PAIRS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="hls4ml")
args = parser.parse_args()

verdicts = {(r["model"], r["a"], r["b"]): r
            for r in variance.collect_verdicts(JET_PAIRS, dataset=args.dataset)}
rows = reproducibility.collect_consistency(JET_PAIRS, dataset=args.dataset)

print("dataset %s | chance %.2f unanimous of %d features | %d seeds"
      % (args.dataset, rows[0]["expected"], rows[0]["n_features"], rows[0]["n_seeds"]))
print("%-4s %-10s %-10s %9s %8s %9s %s"
      % ("model", "a", "b", "unanimous", "p", "margin", "magnitude"))
for r in sorted(rows, key=lambda x: -x["n_unanimous"]):
    v = verdicts[(r["model"], r["a"], r["b"])]
    # the verdict's own margin, ceiling minus cross, positive when the gap beats retraining
    inside = v["cross_tau"] >= v["ceiling_tau"]
    print("%-4s %-10s %-10s %6d/%-2d %8.4f %+9.4f %s"
          % (r["model"], r["a"], r["b"], r["n_unanimous"], r["n_features"],
             r["p_binomial"], v["margin"], "INSIDE NOISE" if inside else "clears"))
