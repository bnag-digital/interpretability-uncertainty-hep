import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_DATASET
from plots import concentration, reproducibility, variance
from scripts.make_figures import JET_PAIRS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=DEFAULT_DATASET)
parser.add_argument("--cls-idx", type=int, default=None)
args = parser.parse_args()

pairs = None if args.dataset == DEFAULT_DATASET else JET_PAIRS
kw = {"dataset": args.dataset, "cls_idx": args.cls_idx}

print("dataset %s, %s" % (args.dataset,
                          "all classes" if args.cls_idx is None else f"class {args.cls_idx}"))

print("\nFLOOR AND CEILING")
print("%-4s %-11s %8s %9s %6s %6s" % ("model", "explainer", "floor", "ceiling",
                                      "feats", "seeds"))
for r in variance.collect_levels(pairs, **kw):
    floor = "%.4f" % r["floor_tau"] if r["floor_tau"] is not None else "     -"
    print("%-4s %-11s %8s %9.4f %6d %6d"
          % (r["model"], r["explainer"], floor, r["ceiling_tau"], r["n_features"],
             r["n_seeds"]))

verdicts = variance.collect_verdicts(pairs, **kw)
rows = {(r["model"], r["a"], r["b"]): r
        for r in reproducibility.collect_consistency(pairs or variance.DEFAULT_PAIRS, **kw)}

print("\nBOTH TESTS")
print("%-4s %-24s %8s %18s %9s %-13s %10s %8s"
      % ("model", "pair", "cross", "90% CI", "margin", "magnitude", "unanimous", "p"))
for v in sorted(verdicts, key=lambda x: x["cross_tau"]):
    r = rows.get((v["model"], v["a"], v["b"]))
    unanimous = "%d/%d" % (r["n_unanimous"], r["n_features"]) if r else "-"
    p = "%.3f" % r["p_binomial"] if r else "-"
    print("%-4s %-24s %8.4f  [%.3f, %.3f] %+9.4f %-13s %10s %8s"
          % (v["model"], f"{v['a']} vs {v['b']}", v["cross_tau"], v["cross_lo"],
             v["cross_hi"], v["margin"],
             "clears" if v["below_ceiling"] else "INSIDE NOISE", unanimous, p))

if rows:
    first = next(iter(rows.values()))
    print("chance %.2f unanimous features of %d, %d seeds"
          % (first["expected"], first["n_features"], first["n_seeds"]))

print("\nCONCENTRATION")
for c in sorted(concentration.collect_concentration(pairs, **kw),
                key=lambda x: x["eff_n_frac"]):
    print("%-4s %-11s %6.2f  (%d features)"
          % (c["model"], c["explainer"], c["eff_n_frac"], c["n_features"]))
