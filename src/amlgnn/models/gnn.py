"""GNN architectures for illicit-transaction detection.

One configurable class covers GCN / GraphSAGE / GAT with three wiring styles:
    skip=None        plain stacked layers (baseline)
    skip="residual"  h_{l+1} = h_l + f(conv(h_l))          (ResNet-style skip connections)
    skip="dense"     classifier sees [h_0, h_1, ..., h_L]  (dense / jumping-knowledge skips)

Every model exposes:
    forward(x, edge_index) -> logits [N, 2]       (class 1 = illicit)
    embed(x, edge_index)   -> node embeddings [N, embed_dim]
    embed_dim, needs_time (False)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


def _make_conv(kind: str, in_dim: int, out_dim: int, heads: int) -> nn.Module:
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "sage":
        return SAGEConv(in_dim, out_dim)
    if kind == "gat":
        assert out_dim % heads == 0, "hidden_dim must be divisible by heads for GAT"
        return GATConv(in_dim, out_dim // heads, heads=heads)
    raise ValueError(f"Unknown conv kind: {kind}")


class GNN(nn.Module):
    needs_time = False

    def __init__(self, kind: str, in_dim: int, hidden_dim: int = 128, out_dim: int = 2,
                 num_layers: int = 2, dropout: float = 0.3, heads: int = 4, skip=None):
        super().__init__()
        assert skip in (None, "residual", "dense")
        self.skip, self.dropout, self.num_layers = skip, dropout, num_layers

        # With skips we first project the input so every layer has equal width.
        self.lin_in = nn.Linear(in_dim, hidden_dim) if skip else None
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            d_in = hidden_dim if (skip or i > 0) else in_dim
            self.convs.append(_make_conv(kind, d_in, hidden_dim, heads))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.embed_dim = hidden_dim * (num_layers + 1) if skip == "dense" else hidden_dim
        self.classifier = nn.Linear(self.embed_dim, out_dim)

    def embed(self, x, edge_index):
        h = x
        if self.lin_in is not None:
            h = F.dropout(F.relu(self.lin_in(h)), self.dropout, self.training)
        feats = [h]
        for conv, norm in zip(self.convs, self.norms):
            z = F.dropout(F.relu(norm(conv(h, edge_index))), self.dropout, self.training)
            h = h + z if self.skip == "residual" else z
            feats.append(z)
        return torch.cat(feats, dim=-1) if self.skip == "dense" else h

    def forward(self, x, edge_index):
        return self.classifier(self.embed(x, edge_index))
