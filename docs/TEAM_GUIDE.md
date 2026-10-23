# Team guide (for the team lead)

## Push order (each member pushes only their zip's files)
1. **Member 1** (data + packaging) -> pushes first, because everything imports `amlgnn.config` / `amlgnn.data`.
2. **Member 2** (models)
3. **Member 3** (training/eval/experiments)
4. **Member 4** (temporal/analysis/CI) -> pushes last; CI goes green once all 4 parts are in.

Parts 2-4 can be developed in parallel locally after unzipping Part 1 as a base; the interfaces are fixed:

| Interface | Contract |
|---|---|
| Dataset | `torch_geometric.data.Data` with `x, edge_index, y (1/0/-1), time, train_mask, val_mask, test_mask` |
| Model | `forward(x, edge_index) -> [N,2]` logits, `embed(...)`, `embed_dim`, `needs_time` flag (True -> `forward(x, edge_index, time)`) |
| Trainer | `train_model(model, data, TrainConfig) -> dict(val, test, threshold, test_per_timestep, history, probs)` |

## Git workflow
```bash
# once, lead:
git init aml-gnn && cd aml-gnn && git remote add origin <repo-url>
# each member:
git clone <repo-url> && cd aml-gnn
git checkout -b member-N-<module>
unzip ../partN_*.zip -d ..          # zip contains the top-level folder aml-gnn/, files land in your clone
git add . && git commit -m "Add <module>" && git push -u origin member-N-<module>
# open a Pull Request -> main, merge in the push order above
```
If the Python code sits in a different folder name, just copy the *contents* of `aml-gnn/` into the repo root.

## Stretch tasks (if time remains)
* DIAM-style node-discrepancy module (Member 2) · hyper-parameter sweep (Member 3)
* GNNExplainer / subgraph case studies (Member 4) · report + slides (Member 1)
