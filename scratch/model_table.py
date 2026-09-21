import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import DEFAULT_DATASET
from plots import performance
from results import load

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=DEFAULT_DATASET)
args = parser.parse_args()

frame = performance.accuracy_frame(dataset=args.dataset)
if frame.empty:
    raise SystemExit(f"no performance artifacts for {args.dataset}")

spread = performance.spread_table(frame)
names = performance.class_names_of(frame)

print("dataset %s" % args.dataset)
print("\nACCURACY AND SEED SPREAD")
print("%-5s %-22s %6s %9s %9s %9s %10s"
      % ("model", "level", "seeds", "mean", "min", "max", "spread_pts"))
for _, r in spread.iterrows():
    print("%-5s %-22s %6d %9.4f %9.4f %9.4f %10.2f"
          % (r["model"], r["level"], r["n_seeds"], r["mean"], r["lo"], r["hi"],
             r["spread_pts"]))

print("\nPER-CLASS AUC, mean over seeds")
print("%-5s %s" % ("model", "  ".join("%8s" % c for c in names)))
for model in sorted(frame["model"].unique()):
    rows = frame[frame["model"] == model]["name"]
    aucs = np.stack([load(n)[0] for n in rows])
    print("%-5s %s" % (model, "  ".join("%8.4f" % v for v in aucs.mean(axis=0))))
