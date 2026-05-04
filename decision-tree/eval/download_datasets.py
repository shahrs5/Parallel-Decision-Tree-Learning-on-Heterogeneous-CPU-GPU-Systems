"""
download_datasets.py — downloads all 4 benchmark datasets into data/

Run this once before building or running sklearn_compare.py:
    python download_datasets.py

All datasets come from UCI / sklearn built-ins.
No login or API key required.
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
            load_iris, load_wine, load_breast_cancer,  fetch_covtype
        )
        import urllib.request
        import numpy as np
    except ImportError:
        print("Run: pip install scikit-learn numpy")
        sys.exit(1)

    ensure_dir("data")

    # ---- Iris ----
    data = load_iris()
    rows = [list(x) + [int(y)] for x, y in zip(data.data, data.target)]
    write_csv("data/iris.csv", rows)
    
    # ---- Wine ----
    data = load_wine()
    rows = [list(x) + [int(y)] for x, y in zip(data.data, data.target)]
    write_csv("data/wine.csv", rows)

    # ---- Breast Cancer ----
    data = load_breast_cancer()
    rows = [list(x) + [int(y)] for x, y in zip(data.data, data.target)]
    write_csv("data/breast_cancer.csv", rows)

    # ---- CoverType ----
    print("  Downloading Covertype dataset (this may take a moment)...")
    data = fetch_covtype()
    rows = [list(map(float, x)) + [int(y)] 
        for x, y in zip(data.data, data.target)]
    write_csv("data/covertype.csv", rows)

    # ---- Banknote Authentication (UCI) ----
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
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
        print(f"  Could not download banknote from UCI: {e}")
        print("  Manual download: https://archive.ics.uci.edu/ml/datasets/banknote+authentication")
        print("  Save as: data/banknote.csv")
        print("  Format: variance,skewness,curtosis,entropy,class (no header)")

    print("\nAll datasets ready in data/")

if __name__ == "__main__":
    download_all()