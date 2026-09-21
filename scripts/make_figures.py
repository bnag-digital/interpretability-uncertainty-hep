import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

import results
from config import CONFIG, DATASET_NAMES, DEFAULT_DATASET, JET_FEATURES, NODE_FEATURES
from plots import (attribution, concentration, interaction, mechanism, null,
                   performance, reproducibility, style, variance)

# the only models a dataset without a particle view has
JET_PAIRS = {"bdt": ["shap", "lime"],
             "dnn": ["shap", "lime", "saliency", "ig", "smoothgrad"]}


def draw(name, fn):
    """Draw one figure, reporting what happened instead of raising."""
    try:
        fig = fn()
    except Exception as exc:
        print(f"  skipped {name}: {type(exc).__name__}: {exc}", flush=True)
        return None
    path = style.save(fig, name)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_NAMES, default=DEFAULT_DATASET,
                        help="which dataset's artifacts to draw")
    parser.add_argument("--cls-idx", type=int, default=None,
                        help="explain one class instead of averaging over all of them. "
                             "Only affects models whose attributions carry a class axis.")
    parser.add_argument("--figure-dir", default=CONFIG.figure_dir,
                        help="where the png and pdf go. Defaults to the configured "
                             "directory, so a study drawn at a different number of "
                             "seeds can go elsewhere instead of overwriting this one.")
    args = parser.parse_args()
    dataset = args.dataset
    cls_idx = args.cls_idx
    # a second dataset writes its name into every filename, so nothing overwrites
    suffix = "" if dataset == DEFAULT_DATASET else f"_{dataset}"
    if cls_idx is not None:
        suffix += f"_cls{cls_idx}"
    tabular_only = dataset != DEFAULT_DATASET
    pairs = JET_PAIRS if tabular_only else None

    style.setup()
    CONFIG.figure_dir = args.figure_dir
    CONFIG.make_dirs()
    df = results.scan()
    if df.empty:
        raise SystemExit("no artifacts found, so there is nothing to draw")

    print("=" * 72)
    print(f"FIGURES  dataset {dataset}"
          f"{f'  class {cls_idx}' if cls_idx is not None else '  all classes'}")
    print("=" * 72)

    ok, reason = results.can_draw(df, performance.REQUIRES)
    frame = performance.accuracy_frame(df, dataset=dataset) if ok else None
    if frame is not None and not frame.empty:
        draw(f"fig1a_accuracy{suffix}", lambda: performance.plot_accuracy(frame))
        draw(f"fig1b_per_class_auc{suffix}", lambda: performance.plot_per_class_auc(frame))
        draw(f"fig1c_accuracy_vs_size{suffix}",
             lambda: performance.plot_accuracy_vs_size(frame))
    else:
        print(f"  skipped figure 1: {reason or 'no runs on ' + dataset}", flush=True)

    ok, reason = results.can_draw(df, variance.requires(dataset))
    if ok:
        levels = variance.collect_levels(pairs, dataset=dataset, cls_idx=cls_idx)
        verdicts = variance.collect_verdicts(pairs, dataset=dataset, cls_idx=cls_idx)
        draw(f"fig2a_floor_and_ceiling{suffix}", lambda: variance.plot_levels(levels))
        draw(f"fig2b_cross_vs_ceiling{suffix}", lambda: variance.plot_verdicts(verdicts))
        # one panel per dataset, restricted to the models every dataset has
        if dataset == DEFAULT_DATASET:
            matched = variance.collect_verdicts(JET_PAIRS, dataset=dataset,
                                                cls_idx=cls_idx)
            panels = [("hls4ml jets", matched)]
            for other in [d for d in DATASET_NAMES if d != DEFAULT_DATASET]:
                try:
                    rows = variance.collect_verdicts(JET_PAIRS, dataset=other,
                                                     cls_idx=cls_idx)
                except Exception:
                    rows = []
                if rows:
                    panels.append((other.capitalize(), rows))
            if len(panels) > 1:
                draw(f"fig2b_money_two_datasets{suffix}",
                     lambda: variance.plot_verdicts_by_dataset(panels))
            else:
                print("  skipped money plot: only one dataset has artifacts", flush=True)

            # the same plot with the jet panel unrestricted, so GNN and PFN appear
            full = [("hls4ml jets, every model", verdicts)] + panels[1:]
            if len(full) > 1:
                draw(f"fig2b_money_all_models{suffix}",
                     lambda: variance.plot_verdicts_by_dataset(full))

        # the second test, its rows ordered by the verdict panel above it
        consistency = reproducibility.collect_consistency(
            pairs or variance.DEFAULT_PAIRS, dataset=dataset, cls_idx=cls_idx)
        if consistency:
            order = [(r["model"], r["a"], r["b"])
                     for r in sorted(verdicts, key=lambda x: x["cross_tau"])]
            draw(f"fig2c_consistency{suffix}",
                 lambda: reproducibility.plot_consistency(
                     consistency, order=order, show_legend=False))
            contrast = contrast_rows(consistency, verdicts)
            if contrast:
                names = feature_names_of(dataset, contrast[0])
                draw(f"fig2d_displacement{suffix}",
                     lambda r=contrast: reproducibility.plot_displacement_contrast(
                         r, feature_names=names))
        else:
            print("  skipped figures 2c and 2d: no attribution artifacts", flush=True)
    else:
        print(f"  skipped figure 2: {reason}", flush=True)

    # figure 3, the 2x2. Only models sharing a feature space can be compared.
    squares = [("jet", interaction.JET_SQUARE)]
    if not tabular_only:
        squares.append(("particle", interaction.PARTICLE_SQUARE))
    for tag, square in squares:
        try:
            result = interaction.collect_square(square, dataset=dataset, cls_idx=cls_idx)
        except Exception as exc:
            print(f"  skipped figure 3 ({tag}): {type(exc).__name__}: {exc}", flush=True)
            continue
        draw(f"fig3a_interaction_{tag}{suffix}", lambda r=result: interaction.plot_square(r))
        draw(f"fig3b_paired_deltas_{tag}{suffix}",
             lambda r=result: interaction.plot_paired_deltas(r))

    # figure 4 needs nothing on disk, since the null is simulated
    table = null.null_table()
    observed = observed_taus(df, dataset, tabular_only, cls_idx)
    draw(f"fig4a_null_vs_n{suffix}", lambda: null.plot_null_vs_n(table))
    draw(f"fig4b_false_agreement{suffix}", lambda: null.plot_false_agreement(table))
    if observed:
        draw(f"fig4c_observed_against_null{suffix}",
             lambda: null.plot_observed_against_null(table, observed))

    # figure 5
    ok, reason = results.can_draw(df, mechanism.requires(dataset))
    if ok:
        wanted = [("bdt", ["shap", "lime"])]
        if not tabular_only:
            wanted.append(("gnn", ["saliency", "ig", "smoothgrad", "occlusion"]))
        for model, explainers in wanted:
            rows = mechanism.collect_mechanism(model, explainers,
                                               train_level_of(df, model, dataset),
                                               dataset=dataset, cls_idx=cls_idx)
            if not rows:
                print(f"  skipped figure 5 ({model}): no matching artifacts", flush=True)
                continue
            draw(f"fig5a_agreement_vs_tau_{model}{suffix}",
                 lambda r=rows: mechanism.plot_agreement_vs_tau(r))
            draw(f"fig5b_agreement_spread_{model}{suffix}",
                 lambda r=rows: mechanism.plot_agreement_spread(r))
    else:
        print(f"  skipped figure 5: {reason}", flush=True)

    # figure 6
    conc = concentration.collect_concentration(pairs, dataset=dataset, cls_idx=cls_idx)
    if conc:
        draw(f"fig6a_concentration{suffix}", lambda: concentration.plot_concentration(conc))
        draw(f"fig6b_top1_share{suffix}", lambda: concentration.plot_top1_share(conc))
    else:
        print("  skipped figure 6: no attribution artifacts", flush=True)

    # per-feature panels, which say which features are disagreed about
    draw_attribution_panels(df, dataset, suffix, tabular_only, cls_idx)

    print(f"\nfigures in {Path(CONFIG.figure_dir).resolve()}")


def contrast_rows(consistency, verdicts):
    """One pair that fails both tests and one that passes both, from one model."""
    inside = {(r["model"], r["a"], r["b"]) for r in verdicts if not r["below_ceiling"]}
    by_model = {}
    for row in consistency:
        side = "inside" if (row["model"], row["a"], row["b"]) in inside else "clears"
        by_model.setdefault(row["model"], {"inside": [], "clears": []})[side].append(row)

    best = None
    for sides in by_model.values():
        if not sides["inside"] or not sides["clears"]:
            continue
        low = min(sides["inside"], key=lambda r: r["n_unanimous"])
        high = max(sides["clears"], key=lambda r: r["n_unanimous"])
        gap = high["n_unanimous"] - low["n_unanimous"]
        if best is None or gap > best[0]:
            best = (gap, [high, low])
    return best[1] if best else None


def feature_names_of(dataset, row):
    """Axis labels for a displacement panel, where the names are known."""
    if dataset != DEFAULT_DATASET:
        return None
    names = JET_FEATURES if row["model"] in ("bdt", "dnn") else NODE_FEATURES
    return list(names)[:row["n_features"]]


def _levels(df, kind, model, dataset):
    sub = results.select(df, kind=kind, model=model)
    levels = sorted({str(x) for x in sub["level"].dropna().unique()}) if not sub.empty else []
    return [x for x in levels if results.level_dataset(x) == dataset]


def train_level_of(df, model, dataset):
    """The level a model's checkpoints were written at on one dataset."""
    found = _levels(df, "performance", model, dataset)
    with_predictions = set(_levels(df, "prediction", model, dataset))
    preferred = [x for x in found if x in with_predictions]
    if preferred:
        return preferred[0]
    return found[0] if found else results.level_tag("n200000", dataset)


def observed_taus(df, dataset=DEFAULT_DATASET, tabular_only=False, cls_idx=None):
    """Measured ceilings, tagged with the feature count they were measured on."""
    from plots import variance as var

    out = []
    for row in var.collect_levels(JET_PAIRS if tabular_only else None, dataset=dataset,
                                  cls_idx=cls_idx):
        out.append({"label": f"{row['model']}/{row['explainer']}",
                    "model": row["model"], "n_features": row["n_features"],
                    "tau": row["ceiling_tau"]})
    return out


def draw_attribution_panels(df, dataset=DEFAULT_DATASET, suffix="", tabular_only=False,
                            cls_idx=None):
    """SHAP against LIME on the BDT, and the GNN's seed spread per feature."""
    try:
        shap_imp, meta = attribution.importance_of("bdt", "shap", dataset=dataset,
                                                   cls_idx=cls_idx)
        lime_imp, _ = attribution.importance_of("bdt", "lime", dataset=dataset,
                                                cls_idx=cls_idx)
    except FileNotFoundError as exc:
        print(f"  skipped attribution panels: {exc}", flush=True)
        return
    # the sidecars carry the feature count but not the names, so use config
    names = meta.get("feature_names") or list(JET_FEATURES)

    draw(f"fig8a_bdt_shap_vs_lime_shares{suffix}",
         lambda: attribution.plot_share_bars({"shap": shap_imp, "lime": lime_imp}, names,
                                             title="BDT: importance share per feature"))
    draw(f"fig8b_bdt_shap_vs_lime_scatter{suffix}",
         lambda: attribution.plot_share_scatter(shap_imp, lime_imp, names, "shap", "lime"))

    # the same two explainers with the retraining spread drawn on them
    try:
        spread = {m: attribution.shares_across_seeds("bdt", m, dataset=dataset,
                                                     cls_idx=cls_idx)[0]
                  for m in ("shap", "lime")}
    except Exception as exc:
        print(f"  skipped fig8d: {type(exc).__name__}: {exc}", flush=True)
    else:
        draw(f"fig8d_bdt_shap_vs_lime_shares_spread{suffix}",
             lambda: attribution.plot_share_bars_spread(spread, names))

    if tabular_only:
        return
    try:
        stack, _ = results.stack_seeds("gnn", "ig", results.level_tag("ceiling", dataset))
    except Exception as exc:
        print(f"  skipped seed spread panel: {type(exc).__name__}: {exc}", flush=True)
        return
    gnn_names = list(NODE_FEATURES)[:stack.shape[-1]]
    draw("fig8c_gnn_ig_seed_spread",
         lambda: attribution.plot_seed_spread(stack, gnn_names,
                                              title="GNN / IG: share per feature, one "
                                                    "point per model seed"))


if __name__ == "__main__":
    main()
