import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data.hls4ml import EXPECTED_JET_FEATURES, list_h5

h5_names = ["j_" + n for n in EXPECTED_JET_FEATURES]

paths = list_h5(CONFIG.data.particle_train_dir)[:4]
chunks, jf_names = [], None
for path in paths:
    with h5py.File(path, "r") as f:
        jf_names = [n.decode() if isinstance(n, bytes) else n for n in f["jetFeatureNames"][:]]
        idx = [jf_names.index(n) for n in h5_names]
        chunks.append(f["jets"][:][:, idx])
h5_X = np.concatenate(chunks).astype(np.float64)
print(f"h5:     {h5_X.shape} from {len(paths)} files")

from sklearn.datasets import fetch_openml

data = fetch_openml("hls4ml_lhc_jets_hlf", data_home=CONFIG.data.openml_cache)
ml_X = data["data"].values.astype(np.float64)
print(f"openml: {ml_X.shape}")
print(f"openml column order matches EXPECTED_JET_FEATURES: "
      f"{list(data['data'].columns) == EXPECTED_JET_FEATURES}")

print(f"\n{'feature':16s} {'h5 mean':>12s} {'oml mean':>12s} {'h5 std':>12s} "
      f"{'oml std':>12s} {'mean diff/std':>14s}")
worst = 0.0
for i, name in enumerate(EXPECTED_JET_FEATURES):
    a, b = h5_X[:, i], ml_X[:, i]
    scale = max(b.std(), 1e-9)
    rel = abs(a.mean() - b.mean()) / scale
    worst = max(worst, rel)
    print(f"{name:16s} {a.mean():12.4f} {b.mean():12.4f} {a.std():12.4f} "
          f"{b.std():12.4f} {rel:14.4f}")

print(f"\nworst standardised mean gap: {worst:.4f}")
print("same quantities" if worst < 0.1 else "DIFFERENT: do not reuse these columns")
