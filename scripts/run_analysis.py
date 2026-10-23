"""Train a few models and produce all report figures + pattern-analysis tables.

  python scripts/run_analysis.py --data-dir data/elliptic --out results/analysis
  python scripts/run_analysis.py --synthetic --epochs 15 --out results_ci/analysis
"""
import argparse
import json
import os

import numpy as np
import torch

from amlgnn import viz
from amlgnn.config import DataConfig, TrainConfig
from amlgnn.data import build_dataset, make_undirected, per_timestep_counts
from amlgnn.explain import (feature_saliency, illicit_components, neighbor_homophily,
                            structural_stats)
from amlgnn.models import build_model
from amlgnn.train import train_model
from amlgnn.utils import ensure_dir, get_device, set_seed


def make_model(name, in_dim, cfg):
    if name == "temporal_gnn":
        from amlgnn.temporal import TemporalContextGNN
        return TemporalContextGNN(in_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout)
    return build_model(name, in_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=["gcn", "gcn_res", "temporal_gnn"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="results/analysis")
    a = ap.parse_args()
    out = ensure_dir(a.out)
    fig = lambda n: os.path.join(out, n)

    # directed graph for transaction-flow statistics, undirected copy for the GNNs
    directed = build_dataset(DataConfig(data_dir=a.data_dir, synthetic=a.synthetic, to_undirected=False))
    data = make_undirected(directed.clone())

    viz.plot_class_balance(per_timestep_counts(directed), fig("class_balance.png"))
    stats = structural_stats(directed)
    stats.to_csv(fig("structural_stats.csv"), index=False)
    g, sizes = illicit_components(directed)
    with open(fig("illicit_patterns.json"), "w") as f:
        json.dump({"homophily": neighbor_homophily(directed),
                   "n_illicit_components": len(sizes), "largest_components": sizes[:10]}, f, indent=2)
    viz.plot_illicit_subgraph(g, fig("illicit_subgraph.png"))
    print(stats.to_string(index=False))

    cfg = TrainConfig(epochs=a.epochs)
    y = data.y.numpy()
    te = data.test_mask.numpy()
    curves, per_t, trained = {}, {}, {}
    for name in a.models:
        print(f"Training {name}")
        set_seed(cfg.seed)
        model = make_model(name, data.x.size(1), cfg)
        res = train_model(model, data, cfg)
        curves[name] = (y[te], res["probs"][te])
        per_t[name] = res["test_per_timestep"]
        trained[name] = model
        print(f"  {name}: test F1 {res['test']['f1']:.4f} | PR-AUC {res['test']['pr_auc']:.4f}")
        if name == a.models[0]:
            viz.plot_training_curves(res["history"], fig("training_curves.png"))

    viz.plot_pr_curves(curves, fig("pr_curves.png"))
    viz.plot_per_timestep(per_t, fig("per_timestep_f1.png"))

    # explainability on the first model that is a plain GNN
    name = next((n for n in a.models if n != "temporal_gnn"), a.models[0])
    model = trained[name]
    device = next(model.parameters()).device
    d = data.clone().to(device)
    sal = feature_saliency(model, d, d.test_mask & (d.y == 1))
    sal.to_csv(fig("feature_saliency.csv"), index=False)
    viz.plot_saliency(sal, fig("feature_saliency.png"))
    with torch.no_grad():
        model.eval()
        emb = (model.embed(d.x, d.edge_index, d.time) if getattr(model, "needs_time", False)
               else model.embed(d.x, d.edge_index)).cpu().numpy()
    viz.plot_embeddings(emb, y, fig("embeddings_pca.png"))
    print(f"Figures and tables saved to {out}/")


if __name__ == "__main__":
    main()
