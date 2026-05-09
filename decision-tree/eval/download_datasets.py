"""
download_datasets.py — downloads and generates benchmark datasets into data/

Run:
    python download_datasets.py

Requirements:
    pip install scikit-learn numpy
"""

import os
import sys
import csv

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"  Saved: {path}  ({len(rows)} rows)")


def download_all():

    try:
        from sklearn.datasets import (
            load_breast_cancer,
            fetch_covtype,
            make_classification
        )

        import urllib.request
        import numpy as np

    except ImportError:
        print("Run: pip install scikit-learn numpy")
        sys.exit(1)

    ensure_dir("data")

    # ============================================================
    # REAL DATASETS
    # ============================================================

    # ---- Breast Cancer ----
    print("  Preparing Breast Cancer dataset...")

    data = load_breast_cancer()

    rows = [
        list(map(float, x)) + [int(y)]
        for x, y in zip(data.data, data.target)
    ]

    write_csv("data/breast_cancer.csv", rows)

    # ---- CoverType ----
    print("  Downloading CoverType dataset (large)...")

    data = fetch_covtype()

    out_path = "data/covertype.csv"

    with open(out_path, "w", newline="") as f:

        writer = csv.writer(f)

        for x, y in zip(data.data, data.target):
            writer.writerow(list(map(float, x)) + [int(y)])

    print(f"  Saved: {out_path}  ({len(data.data)} rows)")

    
    # ---- Banknote Authentication ----
    url = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00267/"
        "data_banknote_authentication.txt"
    )

    try:
        print("  Downloading Banknote dataset from UCI...")

        with urllib.request.urlopen(url, timeout=10) as resp:
            content = resp.read().decode("utf-8")

        rows = []

        for line in content.strip().splitlines():

            parts = [p.strip() for p in line.split(",")]

            if len(parts) == 5:
                rows.append(parts)

        write_csv("data/banknote.csv", rows)

    except Exception as e:

        print(f"  Could not download banknote dataset: {e}")

        print(
            "  Manual download:\n"
            "  https://archive.ics.uci.edu/ml/datasets/banknote+authentication"
        )

    # ============================================================
    # SYNTHETIC SCALABILITY DATASETS
    # ============================================================

    dataset_sizes = [
        100000,
        200000,
        300000,
        500000,
        1000000
    ]

    feature_configs = [8, 16, 32]

    for n_features in feature_configs:

        # informative/redundant scale proportionally
        n_informative = max(2, int(n_features * 0.75))
        n_redundant   = max(1, int(n_features * 0.125))

        for n_samples in dataset_sizes:

            if n_samples < 300000 and n_features == 16:
                continue

            print(
                f"\n  Generating synthetic dataset: "
                f"{n_samples} samples x {n_features} features"
            )

            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_repeated=0,
                n_classes=2,
                random_state=42
            )

            X = X.astype(np.float32)

            out_path = (
                f"data/synthetic_"
                f"{n_samples}_{n_features}f.csv"
            )

            with open(out_path, "w", newline="") as f:

                writer = csv.writer(f)

                for features, label in zip(X, y):
                    writer.writerow(list(features) + [int(label)])

            print(f"  Saved: {out_path}  ({len(X)} rows)")

    print("\nAll datasets ready inside data/")


if __name__ == "__main__":
    download_all()