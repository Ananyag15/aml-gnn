"""Illicit-pattern analysis: what do money-laundering nodes look like structurally and feature-wise?"""
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import degree

from amlgnn.train import forward_pass


def structural_stats(data) -> pd.DataFrame:
    """In/out-degree profile of illicit vs licit nodes (pass a DIRECTED graph to get in != out)."""
    n = data.num_nodes
    src, dst = data.edge_index
    out_deg = degree(src, n).cpu().numpy()
    in_deg = degree(dst, n).cpu().numpy()
    y = data.y.cpu().numpy()
    rows = []
    for name, lab in (("illicit", 1), ("licit", 0)):
        m = y == lab
        rows.append({
            "class": name, "nodes": int(m.sum()),
            "mean_in_degree": in_deg[m].mean(), "mean_out_degree": out_deg[m].mean(),
            "max_in_degree": in_deg[m].max(), "max_out_degree": out_deg[m].max(),
        })
    return pd.DataFrame(rows).round(3)


def neighbor_homophily(data) -> dict:
    """How often do illicit nodes link to other illicit nodes? (labelled endpoints only)"""
    src, dst = data.edge_index
    y = data.y
    both = (y[src] >= 0) & (y[dst] >= 0)
    s, d = y[src][both], y[dst][both]

    def frac(label):
        sel = s == label
        return float((d[sel] == 1).float().mean()) if sel.any() else float("nan")

    return {"P(next hop illicit | source illicit)": frac(1),
            "P(next hop illicit | source licit)": frac(0)}


def illicit_components(data):
    """Connected components of the illicit-only subgraph -> laundering 'rings' / chains."""
    ill = set(torch.where(data.y == 1)[0].tolist())
    g = nx.Graph()
    g.add_nodes_from(ill)
    for s, d in data.edge_index.t().tolist():
        if s in ill and d in ill:
            g.add_edge(s, d)
    sizes = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
    return g, sizes


def feature_saliency(model, data, mask, target: int = 1, top_k: int = 15) -> pd.DataFrame:
    """Mean |d logit_illicit / d feature| over the nodes in `mask` (gradient saliency)."""
    model.eval()
    d = data.clone()
    d.x = data.x.clone().requires_grad_(True)
    logits = forward_pass(model, d)
    logits[mask, target].sum().backward()
    sal = d.x.grad[mask].abs().mean(0).detach().cpu().numpy()
    order = np.argsort(-sal)[:top_k]
    return pd.DataFrame({"feature": [f"f{i}" for i in order], "feature_index": order,
                         "saliency": sal[order]})
