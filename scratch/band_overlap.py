import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DEFAULT_DATASET
from plots import variance
from scripts.make_figures import JET_PAIRS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=DEFAULT_DATASET)
parser.add_argument("--cls-idx", type=int, default=None)
args = parser.parse_args()

pairs = None if args.dataset == DEFAULT_DATASET else JET_PAIRS
kw = {"dataset": args.dataset, "cls_idx": args.cls_idx}

levels = {(r["model"], r["explainer"]): r for r in variance.collect_levels(pairs, **kw)}
verdicts = variance.collect_verdicts(pairs, **kw)

print("dataset %s, %s" % (args.dataset,
                          "all classes" if args.cls_idx is None else f"class {args.cls_idx}"))
print("\n%-4s %-24s %8s %17s %9s %9s %-12s %s"
      % ("model", "pair", "cross", "90% CI", "ceiling", "margin", "verdict", "reaches"))

n_touch = n_overlap = n_inside = n_escape = 0
for v in sorted(verdicts, key=lambda x: x["margin"]):
    # the lower ceiling decides, so its interval is the one that matters
    dec = min((levels[(v["model"], v["a"])], levels[(v["model"], v["b"])]),
              key=lambda r: r["ceiling_tau"])
    touches = v["cross_hi"] >= dec["ceiling_tau"]
    overlaps = v["cross_hi"] >= dec["ceiling_lo"]
    escapes = v["cross_lo"] <= dec["ceiling_tau"]

    if v["below_ceiling"]:
        note = "into band" if touches else ("to ceiling CI" if overlaps else "-")
        n_touch += touches
        n_overlap += overlaps
    else:
        n_inside += 1
        note = "out of band" if escapes else "-"
        n_escape += escapes

    print("%-4s %-24s %8.4f [%.4f, %.4f] %9.4f %+9.4f %-12s %s"
          % (v["model"], f"{v['a']} vs {v['b']}", v["cross_tau"], v["cross_lo"],
             v["cross_hi"], dec["ceiling_tau"], v["margin"],
             "clears" if v["below_ceiling"] else "INSIDE NOISE", note))

n = len(verdicts)
print("\n%d of %d pairs inside the band on the point estimates" % (n_inside, n))
print("%d of the %d clearing pairs have a CI reaching into the band" % (n_touch, n - n_inside))
print("%d of the %d clearing pairs have a CI reaching the ceiling's own CI" % (n_overlap, n - n_inside))
print("%d of the %d inside-band pairs have a CI reaching back out" % (n_escape, n_inside))
print("unresolved either way: %d of %d" % (n_overlap + n_escape, n))
