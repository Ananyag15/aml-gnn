from .gnn import GNN
from .mlp import MLP

# name -> (conv kind, skip style)
MODEL_SPECS = {
    "gcn":        ("gcn", None),
    "sage":       ("sage", None),
    "gat":        ("gat", None),
    "gcn_res":    ("gcn", "residual"),
    "sage_res":   ("sage", "residual"),
    "gat_res":    ("gat", "residual"),
    "gcn_dense":  ("gcn", "dense"),
}
ALL_MODELS = ["mlp"] + list(MODEL_SPECS)


def build_model(name: str, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                dropout: float = 0.3, heads: int = 4, out_dim: int = 2):
    if name == "mlp":
        return MLP(in_dim, hidden_dim, out_dim, num_layers, dropout)
    if name not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{name}'. Choose from {ALL_MODELS}")
    kind, skip = MODEL_SPECS[name]
    return GNN(kind, in_dim, hidden_dim, out_dim, num_layers, dropout, heads, skip)


__all__ = ["GNN", "MLP", "MODEL_SPECS", "ALL_MODELS", "build_model"]
