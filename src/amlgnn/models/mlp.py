"""Graph-agnostic MLP baseline (ablation: how much does the graph structure help?)."""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    needs_time = False

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 2,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout
        self.embed_dim = hidden_dim
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def embed(self, x, edge_index=None):
        for lin in self.layers:
            x = F.dropout(F.relu(lin(x)), self.dropout, self.training)
        return x

    def forward(self, x, edge_index=None):
        return self.classifier(self.embed(x))
