"""Temporal-context GNN (our own design for the "temporal learning" direction).

In Elliptic every transaction exists in exactly ONE time step, so there are no per-node
histories. Instead we model how the *whole network* drifts over time:

  1. a GNN encodes every node using only its own time-step graph,
  2. each time step is summarised by the mean of its node embeddings,
  3. a (uni-directional, hence causal) GRU runs over the sequence of step summaries,
  4. the classifier sees [node embedding || GRU context of that node's time step].

The GRU context lets the model adapt to market regime changes (e.g. the dark-market
shutdown around step 43) without ever looking at future time steps.
"""
import torch
import torch.nn as nn

from amlgnn.models import build_model


class TemporalContextGNN(nn.Module):
    needs_time = True

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, backbone: str = "gcn_res", out_dim: int = 2):
        super().__init__()
        self.encoder = build_model(backbone, in_dim, hidden_dim, num_layers, dropout)
        emb = self.encoder.embed_dim
        self.gru = nn.GRU(emb, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(emb + hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim),
        )
        self.embed_dim = emb + hidden_dim

    def _context(self, h, time):
        steps = torch.unique(time)                         # sorted ascending
        idx = torch.bucketize(time, steps)                 # step index of every node
        sums = torch.zeros(len(steps), h.size(1), device=h.device, dtype=h.dtype).index_add_(0, idx, h)
        counts = torch.bincount(idx, minlength=len(steps)).clamp_min(1).unsqueeze(1).to(h.dtype)
        ctx, _ = self.gru((sums / counts).unsqueeze(0))    # [1, T, hidden]
        return ctx.squeeze(0)[idx]                         # [N, hidden]

    def embed(self, x, edge_index, time):
        h = self.encoder.embed(x, edge_index)
        return torch.cat([h, self._context(h, time)], dim=-1)

    def forward(self, x, edge_index, time):
        return self.head(self.embed(x, edge_index, time))
