"""
download_datasets.py -- Download / generate large benchmark datasets.

Saves to decision-tree/data/ as headerless CSV files (features..., label).
Run from the decision-tree/ directory:
    python scripts/download_datasets.py
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from sklearn.datasets import fetch_openml, make_classification
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("ERROR: scikit-learn not found. Run: pip install scikit-learn pandas")
    sys.exit(1)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def save_csv(X, y, path):
    df = pd.DataFrame(np.column_stack([X, y]))
    df.to_csv(path, index=False, header=False, float_format="%.6f")
    print(f"  -> Saved {len(df)} rows x {df.shape[1]-1} features to {os.path.basename(path)}")


def fetch_and_save(name, openml_name, version, out_file, max_samples=None):
    out_path = os.path.join(DATA_DIR, out_file)
    if os.path.exists(out_path):
        print(f"  [EXISTS] {out_file}")
        return True
    print(f"  Fetching '{openml_name}' from OpenML ...")
    try:
        ds = fetch_openml(openml_name, version=version, as_frame=True,
                          parser="auto", cache=True)
        X = ds.data.values.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(ds.target.values).astype(np.int32)
        if max_samples and len(X) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]
        save_csv(X, y, out_path)
        print(f"  [{name}] {len(X)} samples, {X.shape[1]} features, "
              f"{len(np.unique(y))} classes")
        return True
    except Exception as e:
        print(f"  [WARN] OpenML fetch failed: {e}")
        return False


# ---------------------------------------------------------------------------
# 1. Shuttle (Statlog Shuttle — 14,500 samples, 9 features, 7 classes)
# ---------------------------------------------------------------------------
print("\n[1/4] Shuttle dataset (14,500 x 9, 7 classes)")
if not fetch_and_save("Shuttle", "shuttle", 1, "shuttle.csv", max_samples=14500):
    # Fallback: generate synthetic equivalent
    print("  Generating synthetic Shuttle equivalent ...")
    X, y = make_classification(n_samples=14500, n_features=9, n_informative=7,
                               n_classes=7, n_clusters_per_class=1, random_state=1)
    save_csv(X.astype(np.float32), y, os.path.join(DATA_DIR, "shuttle.csv"))

# ---------------------------------------------------------------------------
# 2. Letter Recognition (20,000 samples, 16 features, 26 classes)
# ---------------------------------------------------------------------------
print("\n[2/4] Letter Recognition dataset (20,000 x 16, 26 classes)")
if not fetch_and_save("Letter", "letter", 1, "letter.csv"):
    print("  Generating synthetic Letter equivalent ...")
    X, y = make_classification(n_samples=20000, n_features=16, n_informative=14,
                               n_classes=26, n_clusters_per_class=1, random_state=2)
    save_csv(X.astype(np.float32), y, os.path.join(DATA_DIR, "letter.csv"))

# ---------------------------------------------------------------------------
# 3. Skin_NonSkin (245,057 samples, 3 features, 2 classes)
# ---------------------------------------------------------------------------
print("\n[3/4] Skin_NonSkin dataset (245,057 x 3, 2 classes)")
if not fetch_and_save("Skin", "skin-segmentation", 1, "skin_nonskin.csv"):
    print("  Generating synthetic Skin equivalent (245k x 3) ...")
    X, y = make_classification(n_samples=245057, n_features=3, n_informative=3,
                               n_redundant=0, n_classes=2, random_state=3)
    save_csv(X.astype(np.float32), y, os.path.join(DATA_DIR, "skin_nonskin.csv"))

# ---------------------------------------------------------------------------
# 4. Synthetic 1M (1,000,000 samples, 200 features, 2 classes)
#    Note: loading 1M x 200 in the C++ exe is memory-intensive.
#    We also save a 100k version for quick C++ benchmarking.
# ---------------------------------------------------------------------------
print("\n[4/4] Synthetic large dataset")

syn200k_path = os.path.join(DATA_DIR, "synthetic_200k.csv")
if not os.path.exists(syn200k_path):
    print("  Generating Synthetic 200k x 20 (practical C++ benchmark) ...")
    X, y = make_classification(n_samples=200000, n_features=20, n_informative=15,
                               n_redundant=3, n_classes=2, random_state=42)
    save_csv(X.astype(np.float32), y, syn200k_path)
else:
    print("  [EXISTS] synthetic_200k.csv")

syn1m_path = os.path.join(DATA_DIR, "synthetic_1m.csv")
if not os.path.exists(syn1m_path):
    print("  Generating Synthetic 1M x 200 (large-scale reference) ...")
    print("  WARNING: This creates a ~600MB CSV file and may take 1-2 minutes ...")
    chunk = 100000
    rows_written = 0
    with open(syn1m_path, "w") as fout:
        for seed in range(10):  # 10 x 100k = 1M
            X_c, y_c = make_classification(
                n_samples=chunk, n_features=200, n_informative=100,
                n_redundant=50, n_classes=2, random_state=seed)
            X_c = X_c.astype(np.float32)
            arr = np.column_stack([X_c, y_c])
            np.savetxt(fout, arr, delimiter=",", fmt=["%.4f"]*200 + ["%d"])
            rows_written += chunk
            print(f"    {rows_written:,} / 1,000,000 rows written ...", end="\r")
    print(f"\n  -> Saved 1,000,000 rows to synthetic_1m.csv")
else:
    print("  [EXISTS] synthetic_1m.csv")

print("\nDone. Datasets in:", DATA_DIR)
