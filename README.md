# Graph Neural Networks for Detecting Money-Laundering Patterns in Transaction Networks

Node classification (illicit vs licit) on the **Elliptic Bitcoin transaction graph** (203k transactions,
234k payment-flow edges, 49 time steps, 165 features) using GCN / GraphSAGE / GAT, skip-connection
variants, a temporal-context model, classic baselines, and an illicit-pattern analysis.

## Project layout
```
src/amlgnn/
  config.py  utils.py                     # Member 1
  data/  (loader, synthetic, split, ...)  # Member 1
  models/ (gnn.py, mlp.py)                # Member 2
  metrics.py  train.py  baselines.py      # Member 3
  temporal.py  explain.py  viz.py         # Member 4
scripts/
  explore_data.py                         # Member 1
  run_experiments.py                      # Member 3
  run_analysis.py                         # Member 4
tests/  .github/workflows/ci.yml          # each member owns the tests of their module; CI = Member 4
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install torch_geometric numpy pandas scikit-learn matplotlib networkx pytest
pip install -e .
```

## Data
Download **Elliptic Data Set** from Kaggle (`ellipticco/elliptic-data-set`) and put the three files in `data/elliptic/`:
`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`.
(The CSVs are git-ignored; do not commit them.) No dataset yet? Add `--synthetic` to any script.

## Run
```bash
pytest -q                                                    # tests
python scripts/explore_data.py                               # dataset statistics
python scripts/run_experiments.py --data-dir data/elliptic   # all models x 3 seeds -> results/
python scripts/run_analysis.py                               # figures + pattern analysis -> results/analysis/
# quick smoke test (what CI runs):
python scripts/run_experiments.py --synthetic --epochs 15 --seeds 1 --out results_ci
```

## Methodology (short)
* **Temporal split** (no random split, avoids look-ahead leakage): steps 1-29 train, 30-34 validation, 35-49 test.
  Unlabelled ("unknown") nodes take part in message passing but never in the loss or metrics.
* **Class imbalance**: class-weighted cross-entropy; the decision threshold is tuned on validation F1.
* **Metrics**: illicit-class precision / recall / F1, PR-AUC (model selection), ROC-AUC, plus F1 per test time step.
* **Models**: MLP (no graph), GCN, GraphSAGE, GAT; each with residual or dense skip connections;
  `temporal_gnn` = GNN + GRU over per-time-step summaries (causal); Random Forest and Logistic Regression baselines.
* **Analysis**: degree profile, illicit-to-illicit link rate, illicit connected components, gradient feature saliency, embedding PCA.

## Relation to the reference papers (read before writing the report!)
The code implements standard, well-known techniques that match the *purpose* listed for each paper:
skip connections / baseline architectures (IEEE Access 2025), a GNN comparison on Bitcoin (IEEE SIU 2026),
and a temporal direction (ChronoWave-GNN, 2026). **It is not a re-implementation of those papers' exact
architectures**, and the DIAM node-discrepancy idea (IEEE TNNLS) is only approximated by the link-rate /
homophily analysis in `explain.py`. Read the papers, cite them, and describe honestly what you reproduced
and what you adapted. Adding a DIAM-style module is a good "future work" or stretch task.

## Team split
| Member | Module | Files |
|---|---|---|
| 1 | Data engineering | `config.py`, `utils.py`, `data/*`, `scripts/explore_data.py`, `tests/test_data.py`, `pyproject.toml`, `requirements.txt`, `.gitignore` |
| 2 | Models | `models/*`, `tests/test_models.py` |
| 3 | Training & evaluation | `metrics.py`, `train.py`, `baselines.py`, `scripts/run_experiments.py`, `tests/test_metrics.py`, `tests/test_training.py` |
| 4 | Temporal, analysis, CI | `temporal.py`, `explain.py`, `viz.py`, `scripts/run_analysis.py`, `tests/test_temporal_explain.py`, `.github/workflows/ci.yml`, `README.md` |

No two members touch the same file, so there are no merge conflicts.
