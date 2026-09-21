import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data import hls4ml
from models import gnn

LOG = []

# every constituent column the dataset stores
ALL_NODE_FEATURES = [
    "j1_px", "j1_py", "j1_pz", "j1_e", "j1_erel", "j1_pt", "j1_ptrel",
    "j1_eta", "j1_etarel", "j1_etarot", "j1_phi", "j1_phirel", "j1_phirot",
    "j1_deltaR", "j1_costheta", "j1_costhetarel",
]


def say(line=""):
    print(line, flush=True)
    LOG.append(str(line))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jets", type=int, default=20000)
    parser.add_argument("--features", type=int, choices=(5, 16), default=5)
    parser.add_argument("--epochs", type=int, default=2, help="timed epochs after a warm-up")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-jets", type=int, default=200000)
    parser.add_argument("--target-epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--log", default=f"{CONFIG.log_dir}/time_gnn.log")
    args = parser.parse_args()

    cfg = CONFIG
    cfg.make_dirs()
    if args.features == 16:
        cfg.data.node_features = list(ALL_NODE_FEATURES)
    if args.batch_size:
        cfg.gnn.batch_size = args.batch_size
    # one file's worth of test jets is plenty for a timing run
    cfg.data.n_test_jets = 10000
    cfg.data.split_index_path = f"artifacts/timing_split_{args.n_jets}_{args.features}f.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    say("=" * 72)
    say("GNN EPOCH TIMING")
    say("=" * 72)
    say(f"device        : {device}")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        say(f"gpu           : {prop.name}, {prop.total_memory / 1024 ** 3:.1f} GB")
    say(f"jets          : {args.n_jets}")
    say(f"node features : {args.features}")
    say(f"batch size    : {cfg.gnn.batch_size}, k = {cfg.gnn.k_neighbors}, "
        f"blocks {cfg.gnn.edgeconv_hidden}")

    start = time.time()
    data = hls4ml.load_particle_level(cfg=cfg, n_jets=args.n_jets, verbose=True)
    say(f"\nload          : {time.time() - start:.1f} s")

    start = time.time()
    train_loader = gnn.make_loader(data, "train", data["norm"], k=cfg.gnn.k_neighbors,
                                   batch_size=cfg.gnn.batch_size, shuffle=True)
    graph_seconds = time.time() - start
    n_train = len(data["y_train"])
    knn_gb = n_train * data["X_train"].shape[1] * cfg.gnn.k_neighbors * 2 / 1024 ** 3
    say(f"knn precompute: {graph_seconds:.1f} s for {n_train} jets "
        f"({knn_gb:.2f} GB as int16)")

    model = gnn.ParticleGNN(n_node_features=len(data["node_features"]),
                            edgeconv_hidden=list(cfg.gnn.edgeconv_hidden),
                            n_classes=len(data["class_names"]),
                            k=cfg.gnn.k_neighbors, f_hidden=cfg.gnn.f_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    say(f"parameters    : {n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.gnn.lr)

    say("\nwarm-up epoch (not counted)")
    start = time.time()
    gnn.run_epoch(model, train_loader, device, optimizer)
    say(f"  {time.time() - start:.1f} s")

    say(f"\n{args.epochs} timed epoch(s), graph precomputed")
    times = []
    for i in range(args.epochs):
        start = time.time()
        loss, acc = gnn.run_epoch(model, train_loader, device, optimizer)
        times.append(time.time() - start)
        say(f"  epoch {i + 1}: {times[-1]:.1f} s   loss {loss:.4f}  acc {acc:.4f}")
    per_epoch = sum(times) / len(times)

    # what the graph precompute buys, measured rather than assumed
    say("\none epoch with the graph rebuilt every forward pass, for comparison")
    rebuild_loader = gnn.make_loader(data, "train", data["norm"], k=None,
                                     batch_size=cfg.gnn.batch_size, shuffle=True)
    start = time.time()
    gnn.run_epoch(model, rebuild_loader, device, optimizer)
    rebuild_seconds = time.time() - start
    say(f"  {rebuild_seconds:.1f} s  ({rebuild_seconds / per_epoch:.2f}x the precomputed epoch)")

    if device.type == "cuda":
        say(f"\npeak gpu memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")

    scale = args.target_jets / n_train
    one_run = per_epoch * scale * args.target_epochs
    say()
    say("=" * 72)
    say("EXTRAPOLATION")
    say("=" * 72)
    say(f"measured        : {per_epoch:.1f} s/epoch at {n_train} training jets")
    say(f"at {args.target_jets} jets   : {per_epoch * scale:.1f} s/epoch (linear in jets)")
    say(f"one run         : {one_run / 3600:.1f} h for {args.target_epochs} epochs")
    say(f"{args.seeds} seeds        : {one_run * args.seeds / 3600:.1f} h")
    say(f"memory at {args.target_jets}: "
        f"{hls4ml.estimate_memory_gb(args.target_jets, len(data['node_features'])):.1f} GB "
        f"for the arrays, plus {knn_gb * scale:.1f} GB for the graph")
    say()
    say("Linear scaling in jet count is the assumption here. Epoch time is "
        "dominated by the per-jet forward and backward pass, so it should hold, "
        "but the memory figure is what decides whether the run fits at all.")

    Path(args.log).write_text("\n".join(LOG) + "\n")
    print(f"\nwritten to {args.log}")


if __name__ == "__main__":
    main()
