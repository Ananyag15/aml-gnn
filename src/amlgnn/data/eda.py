"""Dataset summary statistics (used for the report's data section)."""
import pandas as pd
from torch_geometric.data import Data


def summarize(data: Data) -> dict:
    y = data.y
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.x.size(1)),
        "num_time_steps": int(data.time.unique().numel()),
        "illicit": int((y == 1).sum()),
        "licit": int((y == 0).sum()),
        "unknown": int((y == -1).sum()),
        "train_nodes": int(data.train_mask.sum()),
        "val_nodes": int(data.val_mask.sum()),
        "test_nodes": int(data.test_mask.sum()),
    }


def per_timestep_counts(data: Data) -> pd.DataFrame:
    df = pd.DataFrame({"time": data.time.numpy(), "y": data.y.numpy()})
    out = df.groupby("time")["y"].agg(
        illicit=lambda s: int((s == 1).sum()),
        licit=lambda s: int((s == 0).sum()),
        unknown=lambda s: int((s == -1).sum()),
    )
    out["illicit_ratio"] = out["illicit"] / (out["illicit"] + out["licit"]).clip(lower=1)
    return out.reset_index()
