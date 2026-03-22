"""
sklearn_compare.py — scikit-learn comparison for Milestone 1

This script trains a Decision Tree on the same 4 datasets using scikit-learn
and prints results in the same format as the C++ output so you can compare
them side by side.

Requirements:
    pip install scikit-learn pandas

Usage:
    python eval/sklearn_compare.py

Make sure your CSV files are in the data/ directory before running.
"""

import time
import os
import sys

try:
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    print("Missing dependencies. Run: pip install scikit-learn pandas numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset configs — must match what main.cpp uses
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "name":       "Iris",
        "path":       "data/iris.csv",
        "max_depth":  5,
        "min_leaf":   1,
        "test_ratio": 0.2,
        "seed":       42,
    },
    {
        "name":       "Wine",
        "path":       "data/wine.csv",
        "max_depth":  5,
        "min_leaf":   1,
        "test_ratio": 0.2,
        "seed":       42,
    },
    {
        "name":       "Breast Cancer",
        "path":       "data/breast_cancer.csv",
        "max_depth":  7,
        "min_leaf":   2,
        "test_ratio": 0.2,
        "seed":       42,
    },
    {
        "name":       "Banknote Auth",
        "path":       "data/banknote.csv",
        "max_depth":  5,
        "min_leaf":   1,
        "test_ratio": 0.2,
        "seed":       42,
    },
]


def run_dataset(cfg):
    path = cfg["path"]
    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        return None

    df = pd.read_csv(path, header=None)
    # Auto-detect header: if first cell is not numeric, skip it
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = pd.read_csv(path)

    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:,  -1].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_ratio"],
        random_state=cfg["seed"],
        shuffle=True,
    )

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_leaf"],
        random_state=cfg["seed"],
    )

    # Train timing
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_ms = (time.perf_counter() - t0) * 1000.0

    # Inference timing
    t1 = time.perf_counter()
    preds = clf.predict(X_test)
    infer_ms = (time.perf_counter() - t1) * 1000.0

    acc = accuracy_score(y_test, preds)

    return {
        "name":      cfg["name"],
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "max_depth": cfg["max_depth"],
        "train_ms":  train_ms,
        "infer_ms":  infer_ms,
        "accuracy":  acc,
        "n_nodes":   clf.tree_.node_count,
    }


def print_separator(char="-", width=90):
    print(char * width)


def main():
    print("=" * 90)
    print("scikit-learn Comparison — Milestone 1")
    print("=" * 90)

    rows = []
    for cfg in DATASETS:
        print(f"\nDataset: {cfg['name']}")
        result = run_dataset(cfg)
        if result:
            print(f"  Samples:          {result['n_samples']}")
            print(f"  Features:         {result['n_features']}")
            print(f"  Train / Test:     {result['n_train']} / {result['n_test']}")
            print(f"  Node count:       {result['n_nodes']}")
            print(f"  Train time (ms):  {result['train_ms']:.4f}")
            print(f"  Infer time (ms):  {result['infer_ms']:.4f}")
            print(f"  Test accuracy:    {result['accuracy']:.4f}")
            rows.append(result)

    # Summary table
    print("\n" + "=" * 90)
    print("SKLEARN COMPARISON SUMMARY")
    print("=" * 90)
    header = (
        f"{'Dataset':<18}"
        f"{'Samples':>8}"
        f"{'Features':>10}"
        f"{'MaxDepth':>10}"
        f"{'TrainTime(ms)':>15}"
        f"{'InferTime(ms)':>15}"
        f"{'Accuracy':>10}"
    )
    print(header)
    print_separator()
    for r in rows:
        print(
            f"{r['name']:<18}"
            f"{r['n_samples']:>8}"
            f"{r['n_features']:>10}"
            f"{r['max_depth']:>10}"
            f"{r['train_ms']:>15.4f}"
            f"{r['infer_ms']:>15.4f}"
            f"{r['accuracy']:>10.4f}"
        )
    print_separator()

    print("""
NOTE FOR REPORT:
  - scikit-learn uses optimized Cython internals — its training time
    may be longer OR shorter depending on overhead at small dataset sizes.
  - Accuracy differences < 5% are expected due to tie-breaking in splits
    and different random shuffle implementations.
  - For the report, focus on: accuracy within acceptable range, and
    that our C++ inference is competitive.
""")


if __name__ == "__main__":
    main()