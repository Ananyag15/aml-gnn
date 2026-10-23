"""Train + evaluate baselines and GNNs, write result tables.

Examples
  python scripts/run_experiments.py --data-dir data/elliptic
  python scripts/run_experiments.py --synthetic --epochs 5 --models gcn gcn_res --seeds 1 --out results_ci
"""
import argparse
import json
import os

import pandas as pd

from amlgnn.baselines import BASELINES, run_baseline
from amlgnn.config import DataConfig, TrainConfig
from amlgnn.data import build_dataset, summarize
from amlgnn.models import ALL_MODELS, build_model
from amlgnn.train import train_model
from amlgnn.utils import ensure_dir, set_seed

DEFAULT_MODELS = ALL_MODELS + ["temporal_gnn"]


def make_model(name, in_dim, cfg: TrainConfig):
    if name == "temporal_gnn":
        from amlgnn.temporal import TemporalContextGNN   # owned by Member 4
        return TemporalContextGNN(in_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
                                  dropout=cfg.dropout)
    return build_model(name, in_dim, cfg.hidden_dim, cfg.num_layers, cfg.dropout, cfg.heads)


def flat_row(name, seed, res):
    return {
        "model": name, "seed": seed, "best_epoch": res["best_epoch"], "threshold": res["threshold"],
        **{f"test_{k}": v for k, v in res["test"].items()},
        **{f"val_{k}": v for k, v in res["val"].items() if k in ("f1", "pr_auc")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--baselines", nargs="*", default=BASELINES)
    ap.add_argument("--seeds", type=int, default=3, help="number of random seeds")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    out = ensure_dir(a.out)
    data = build_dataset(DataConfig(data_dir=a.data_dir, synthetic=a.synthetic))
    print("Dataset:", summarize(data))

    rows, per_t = [], {}
    for seed in range(42, 42 + a.seeds):
        cfg = TrainConfig(hidden_dim=a.hidden, num_layers=a.layers, lr=a.lr, epochs=a.epochs, seed=seed)
        for b in a.baselines:
            print(f"[seed {seed}] baseline {b}")
            res = run_baseline(b, data, seed)
            rows.append(flat_row(b, seed, res))
            per_t.setdefault(b, res["test_per_timestep"])
        for m in a.models:
            print(f"[seed {seed}] model {m}")
            set_seed(seed)
            res = train_model(make_model(m, data.x.size(1), cfg), data, cfg)
            rows.append(flat_row(m, seed, res))
            per_t.setdefault(m, res["test_per_timestep"])

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "metrics_all_runs.csv"), index=False)
    cols = ["test_precision", "test_recall", "test_f1", "test_pr_auc", "test_roc_auc"]
    summary = df.groupby("model")[cols].agg(["mean", "std"]).round(4)
    summary.to_csv(os.path.join(out, "metrics_summary.csv"))
    with open(os.path.join(out, "per_timestep_f1.json"), "w") as f:
        json.dump(per_t, f, indent=2)
    print("\n=== Test results (illicit class, mean over seeds) ===")
    print(df.groupby("model")[cols].mean().round(4).sort_values("test_f1", ascending=False).to_string())
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
