"""Print dataset statistics:  python scripts/explore_data.py [--synthetic]"""
import argparse

from amlgnn.config import DataConfig
from amlgnn.data import build_dataset, summarize, per_timestep_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/elliptic")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    data = build_dataset(DataConfig(data_dir=a.data_dir, synthetic=a.synthetic))
    for k, v in summarize(data).items():
        print(f"{k:>16}: {v}")
    print(per_timestep_counts(data).to_string(index=False))


if __name__ == "__main__":
    main()
