"""Matplotlib figures for the report (all functions save a PNG and close the figure)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_balance(counts_df, path):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(counts_df["time"], counts_df["licit"], label="licit")
    ax.bar(counts_df["time"], counts_df["illicit"], bottom=counts_df["licit"], label="illicit")
    ax.set(xlabel="time step", ylabel="labelled transactions", title="Class balance per time step")
    ax.legend()
    _save(fig, path)


def plot_training_curves(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ep, [h["loss"] for h in history], label="train loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot(ep, [h["val_pr_auc"] for h in history], color="C1", label="val PR-AUC")
    ax2.set_ylabel("val PR-AUC")
    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.2))
    _save(fig, path)


def plot_pr_curves(curves: dict, path):
    """curves: {name: (y_true, y_prob)} on the test set."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=name)
    ax.set(xlabel="recall", ylabel="precision", title="Precision-recall (test, illicit class)")
    ax.legend()
    _save(fig, path)


def plot_per_timestep(per_t: dict, path):
    """per_t: {model: {time_step: f1}} -> F1 over the test time steps."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for name, d in per_t.items():
        ts = sorted(int(t) for t in d)
        ax.plot(ts, [d[t] if t in d else d[str(t)] for t in ts], marker="o", ms=3, label=name)
    ax.set(xlabel="test time step", ylabel="illicit F1", title="Temporal robustness")
    ax.legend()
    _save(fig, path)


def plot_embeddings(emb, y, path, max_points=4000, seed=0):
    emb, y = np.asarray(emb), np.asarray(y)
    idx = np.where(y >= 0)[0]
    rng = np.random.default_rng(seed)
    if len(idx) > max_points:
        idx = rng.choice(idx, max_points, replace=False)
    z = PCA(n_components=2, random_state=seed).fit_transform(emb[idx])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for lab, name, c in ((0, "licit", "C0"), (1, "illicit", "C3")):
        m = y[idx] == lab
        ax.scatter(z[m, 0], z[m, 1], s=4, alpha=0.5, c=c, label=name)
    ax.set(title="Node embeddings (PCA)")
    ax.legend()
    _save(fig, path)


def plot_illicit_subgraph(g, path, max_nodes=120):
    """Draw the largest illicit connected component (truncated by BFS to max_nodes)."""
    if g.number_of_nodes() == 0:
        return
    comp = max(nx.connected_components(g), key=len)
    start = next(iter(comp))
    keep = [start] + [v for _, v in nx.bfs_edges(g.subgraph(comp), start)]
    sub = g.subgraph(keep[:max_nodes])
    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_spring(sub, ax=ax, node_size=25, node_color="C3", edge_color="0.6", width=0.6)
    ax.set_title(f"Largest illicit component ({sub.number_of_nodes()} of {len(comp)} nodes shown)")
    _save(fig, path)


def plot_saliency(df, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df["feature"][::-1], df["saliency"][::-1])
    ax.set(xlabel="mean |gradient|", title="Features driving the 'illicit' prediction")
    _save(fig, path)
