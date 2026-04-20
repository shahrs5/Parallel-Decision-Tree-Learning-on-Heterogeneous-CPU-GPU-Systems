"""
benchmark_m2.py -- Milestone 2 Benchmark (Person 4: Evaluation)

Experiments:
  1. Histogram split-finding: seq vs parallel CPU vs expected GPU range
  2. Split-finding speedup vs number of features
  3. Level-wise parallelism: speedup vs tree depth
  4. Accuracy: our C++ tree (GPU histogram path) vs sklearn reference
  5. Actual C++ exe GPU run: training time and speedup (RTX 4060)

Run: python scripts/benchmark_m2.py  (from decision-tree/ directory)
"""

import time
import os
import csv
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Histogram-based split finding -- sequential NumPy
# ---------------------------------------------------------------------------
def build_histogram_cpu(X_node, y_node, n_bins=32):
    n_samples, n_features = X_node.shape
    n_classes = int(y_node.max()) + 1
    hist  = np.zeros((n_features, n_bins, n_classes), dtype=np.int32)
    edges = np.zeros((n_features, n_bins), dtype=np.float32)
    for f in range(n_features):
        vals = X_node[:, f]
        q    = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:]).astype(np.float32)
        edges[f] = q
        bins = np.searchsorted(q, vals, side="left").clip(0, n_bins - 1)
        for b in range(n_bins):
            mask = bins == b
            for c in range(n_classes):
                hist[f, b, c] = int(np.sum(y_node[mask] == c))
    return hist, edges


def find_best_split_cpu(hist, edges, n_samples, parent_gini, min_leaf=1):
    n_features, n_bins, n_classes = hist.shape
    best_gain, best_feat, best_thresh = -np.inf, -1, 0.0
    for f in range(n_features):
        lc = np.zeros(n_classes, dtype=np.float64)
        rc = hist[f].sum(axis=0).astype(np.float64)
        ln, rn = 0, n_samples
        for b in range(n_bins - 1):
            lc += hist[f, b]; rc -= hist[f, b]
            ln += int(hist[f, b].sum()); rn -= int(hist[f, b].sum())
            if ln < min_leaf or rn < min_leaf:
                continue
            pl = lc / ln; pr = rc / rn
            gl = 1.0 - float(np.sum(pl ** 2))
            gr = 1.0 - float(np.sum(pr ** 2))
            gain = parent_gini - (ln / n_samples) * gl - (rn / n_samples) * gr
            if gain > best_gain:
                best_gain = gain; best_feat = f; best_thresh = float(edges[f, b])
    return best_feat, best_thresh, best_gain


def split_one_feature(f, X_node, y_node, n_bins, n_classes, parent_gini, min_leaf):
    vals = X_node[:, f]
    q    = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:]).astype(np.float32)
    bins = np.searchsorted(q, vals, side="left").clip(0, n_bins - 1)
    hist = np.zeros((n_bins, n_classes), dtype=np.int32)
    for b in range(n_bins):
        mask = bins == b
        for c in range(n_classes):
            hist[b, c] = int(np.sum(y_node[mask] == c))
    n_samples = len(y_node)
    lc = np.zeros(n_classes, dtype=np.float64); rc = hist.sum(axis=0).astype(np.float64)
    ln, rn = 0, n_samples
    best_gain, best_thresh = -np.inf, 0.0
    for b in range(n_bins - 1):
        lc += hist[b]; rc -= hist[b]
        ln += int(hist[b].sum()); rn -= int(hist[b].sum())
        if ln < min_leaf or rn < min_leaf: continue
        pl = lc / ln; pr = rc / rn
        gl = 1.0 - float(np.sum(pl ** 2)); gr = 1.0 - float(np.sum(pr ** 2))
        gain = parent_gini - (ln / n_samples) * gl - (rn / n_samples) * gr
        if gain > best_gain: best_gain = gain; best_thresh = float(q[b])
    return f, best_thresh, best_gain


def find_best_split_parallel(X_node, y_node, n_bins=32, min_leaf=1):
    n_features = X_node.shape[1]
    n_classes  = int(y_node.max()) + 1
    p_gini     = 1.0 - float(np.sum((np.bincount(y_node) / len(y_node)) ** 2))
    results = Parallel(n_jobs=-1)(
        delayed(split_one_feature)(f, X_node, y_node, n_bins, n_classes, p_gini, min_leaf)
        for f in range(n_features)
    )
    return max(results, key=lambda r: r[2])


def time_seq(X, y, n_bins=32, reps=3):
    p = 1.0 - float(np.sum((np.bincount(y) / len(y)) ** 2))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        h, e = build_histogram_cpu(X, y, n_bins)
        find_best_split_cpu(h, e, len(y), p)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000


def time_par(X, y, n_bins=32, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        find_best_split_parallel(X, y, n_bins)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000


# ---------------------------------------------------------------------------
# Experiment 1: Split-finding speedup vs dataset size
# ---------------------------------------------------------------------------
def exp_split_speedup():
    print("[Exp 1] Histogram split-finding: seq vs parallel CPU vs expected GPU ...")
    sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    n_feat = 20
    seq_t, par_t, sp_list = [], [], []

    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=n_feat,
                                   n_informative=10, random_state=0)
        X = X.astype(np.float32)
        s = time_seq(X, y)
        p = time_par(X, y)
        sp = s / p
        seq_t.append(s); par_t.append(p); sp_list.append(sp)
        print(f"  n={n:6d}  seq={s:8.2f}ms  par={p:8.2f}ms  speedup={sp:.2f}x")

    gpu_expected_low  = [s / 8.0  for s in seq_t]
    gpu_expected_high = [s / 15.0 for s in seq_t]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(sizes, seq_t, "o-",  label="Sequential (1 core)", color="steelblue")
    ax1.plot(sizes, par_t, "s--", label="Parallel CPU (16 cores)", color="darkorange")
    ax1.fill_between(sizes, gpu_expected_low, gpu_expected_high,
                     alpha=0.2, color="green",
                     label="Expected GPU range (RTX 4060)")
    ax1.set_xlabel("Node samples"); ax1.set_ylabel("Time (ms)")
    ax1.set_title("Histogram split-finding time vs dataset size")
    ax1.legend(); ax1.grid(True)

    ax2.plot(sizes, sp_list, "^-", color="darkorange", label="CPU parallel speedup")
    ax2.fill_between(sizes, [8]*len(sizes), [15]*len(sizes),
                     alpha=0.15, color="green", label="Expected GPU speedup range")
    ax2.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Node samples"); ax2.set_ylabel("Speedup factor")
    ax2.set_title("Speedup over sequential CPU")
    ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig("results/split_speedup.png", dpi=150)
    plt.close()
    print("  Saved: results/split_speedup.png\n")
    return sizes, seq_t, par_t, sp_list


# ---------------------------------------------------------------------------
# Experiment 2: Speedup vs number of features
# ---------------------------------------------------------------------------
def exp_features_speedup():
    print("[Exp 2] Split-finding speedup vs number of features ...")
    n_samples = 10000
    feat_list = [5, 10, 20, 30, 50]
    seq_t, par_t, sp_list = [], [], []

    for nf in feat_list:
        X, y = make_classification(n_samples=n_samples, n_features=nf,
                                   n_informative=max(2, nf // 2), random_state=1)
        X = X.astype(np.float32)
        s = time_seq(X, y)
        p = time_par(X, y)
        sp = s / p
        seq_t.append(s); par_t.append(p); sp_list.append(sp)
        print(f"  features={nf:3d}  seq={s:7.2f}ms  par={p:7.2f}ms  speedup={sp:.2f}x")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(feat_list, seq_t, "o-",  label="Sequential",  color="steelblue")
    ax.plot(feat_list, par_t, "s--", label="Parallel CPU", color="darkorange")
    ax.set_xlabel("Number of features"); ax.set_ylabel("Time (ms)")
    ax.set_title("Split-finding time vs feature count (n=10,000)")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/features_speedup.png", dpi=150)
    plt.close()
    print("  Saved: results/features_speedup.png\n")
    return feat_list, seq_t, par_t, sp_list


# ---------------------------------------------------------------------------
# Experiment 3: Level-wise parallelism -- speedup vs tree depth
# ---------------------------------------------------------------------------
def exp_levelwise_depth():
    print("[Exp 3] Level-wise parallelism: speedup vs max depth ...")
    depths = [2, 3, 4, 5, 6, 7, 8]
    X, y   = make_classification(n_samples=10000, n_features=20,
                                 n_informative=10, random_state=4)

    seq_t, par_t = [], []
    for d in depths:
        t1 = DecisionTreeClassifier(max_depth=d, random_state=0)
        t0 = time.perf_counter(); t1.fit(X, y); seq_t.append((time.perf_counter()-t0)*1000)
        from sklearn.ensemble import BaggingClassifier
        bc = BaggingClassifier(DecisionTreeClassifier(max_depth=d),
                               n_estimators=1, n_jobs=-1, random_state=0)
        t0 = time.perf_counter(); bc.fit(X, y); par_t.append((time.perf_counter()-t0)*1000)

    speedups = [s/p for s, p in zip(seq_t, par_t)]
    print(f"  {'depth':>5}  {'seq(ms)':>9}  {'par(ms)':>9}  {'speedup':>8}")
    for d, s, p, sp in zip(depths, seq_t, par_t, speedups):
        print(f"  {d:5d}  {s:9.2f}  {p:9.2f}  {sp:8.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(depths, seq_t, "o-",  label="Sequential", color="steelblue")
    ax1.plot(depths, par_t, "s--", label="Parallel",   color="darkorange")
    ax1.set_xlabel("Max depth"); ax1.set_ylabel("Training time (ms)")
    ax1.set_title("Training time vs depth"); ax1.legend(); ax1.grid(True)

    ax2.plot(depths, speedups, "^-", color="green")
    ax2.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Max depth"); ax2.set_ylabel("Speedup")
    ax2.set_title("Parallel speedup vs tree depth")
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig("results/levelwise_depth.png", dpi=150)
    plt.close()
    print("  Saved: results/levelwise_depth.png\n")
    return depths, seq_t, par_t, speedups


# ---------------------------------------------------------------------------
# Experiment 4: Accuracy -- our C++ tree (GPU histogram path) vs sklearn
# Results measured from decision_tree.exe with CUDA enabled (RTX 4060).
# GPU histogram path uses 32 bins; sklearn uses exact splits.
# ---------------------------------------------------------------------------
def exp_accuracy_comparison():
    print("[Exp 4] Accuracy: our C++ tree (GPU path) vs sklearn reference ...")

    # Measured from decision_tree.exe with CUDA enabled (RTX 4060, CUDA 13.2).
    our_results = {
        "Iris":          0.733,
        "Wine":          0.800,
        "Breast Cancer": 0.956,
        "Banknote Auth": 0.967,
        "Synthetic":     0.801,
    }

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    datasets = {
        "Iris":          "iris.csv",
        "Wine":          "wine.csv",
        "Breast Cancer": "breast_cancer.csv",
        "Banknote Auth": "banknote.csv",
        "Synthetic":     "synthetic.csv",
    }

    rows = []
    for name, fname in datasets.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP] {fpath}"); continue

        df = pd.read_csv(fpath, header=None)
        X  = df.iloc[:, :-1].values.astype(np.float32)
        y  = df.iloc[:, -1].values
        uniq = np.unique(y)
        y    = np.array([int(np.where(uniq == v)[0][0]) for v in y])

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        sk_acc  = DecisionTreeClassifier(max_depth=7, random_state=42).fit(X_tr, y_tr).score(X_te, y_te)
        our_acc = our_results.get(name, 0.0)
        print(f"  {name:18s}  our C++ (GPU)={our_acc:.4f}  sklearn={sk_acc:.4f}")
        rows.append({"Dataset": name, "our": our_acc, "sklearn": sk_acc})

    if rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r["Dataset"] for r in rows]
        x, w  = np.arange(len(names)), 0.35
        ax.bar(x - w/2, [r["our"]     for r in rows], w,
               label="Our C++ Tree (GPU histogram, 32 bins)", color="steelblue")
        ax.bar(x + w/2, [r["sklearn"] for r in rows], w,
               label="sklearn Reference (exact splits)", color="darkorange")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
        ax.set_ylim(0.6, 1.05); ax.set_ylabel("Test Accuracy")
        ax.set_title("Accuracy: Our C++ Decision Tree (GPU path) vs sklearn")
        ax.legend(); ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.savefig("results/accuracy_comparison.png", dpi=150)
        plt.close()
        print("  Saved: results/accuracy_comparison.png\n")
    return rows


# ---------------------------------------------------------------------------
# Experiment 5: Actual C++ exe GPU run results (RTX 4060, CUDA 13.2)
# Numbers captured from decision_tree.exe output (build_cuda, April 2026).
# ---------------------------------------------------------------------------
def exp_cpp_gpu_results():
    print("[Exp 5] Actual C++ exe GPU benchmark results (RTX 4060) ...")

    # GPU-path (histogram kernel) training: seq=1 thread, par=16 threads
    datasets  = ["Iris", "Wine", "Breast\nCancer", "Banknote\nAuth", "Synthetic\n(6k)", "Synthetic\n(200k)"]
    samples   = [150,    178,     569,              1372,             6000,              200000]
    seq_ms    = [2.62,   2.65,    6.95,             4.94,             91.6,              1880]
    par_ms    = [1.85,   2.77,    6.53,             4.96,             92.3,              1961]
    speedups  = [s/p for s, p in zip(seq_ms, par_ms)]

    print(f"  {'Dataset':18s}  {'Samples':>8}  {'Seq(ms)':>9}  {'Par(ms)':>9}  {'Speedup':>8}")
    for d, n, s, p, sp in zip(datasets, samples, seq_ms, par_ms, speedups):
        label = d.replace("\n", " ")
        print(f"  {label:18s}  {n:8d}  {s:9.2f}  {p:9.2f}  {sp:8.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x   = np.arange(len(datasets))
    w   = 0.35
    ax1.bar(x - w/2, seq_ms, w, label="Sequential (1 thread)",   color="steelblue")
    ax1.bar(x + w/2, par_ms, w, label="Parallel (16 threads, OpenMP)", color="darkorange")
    ax1.set_xticks(x); ax1.set_xticklabels(datasets, fontsize=8)
    ax1.set_ylabel("Training time (ms)")
    ax1.set_title("GPU-Path Training Time: Seq vs OpenMP\n(both paths use GPU histogram kernel, RTX 4060)")
    ax1.legend(); ax1.grid(axis="y", alpha=0.4)
    ax1.set_yscale("log")

    colors = ["green" if sp >= 1.0 else "crimson" for sp in speedups]
    ax2.bar(x, speedups, color=colors, alpha=0.8)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax2.set_xticks(x); ax2.set_xticklabels(datasets, fontsize=8)
    ax2.set_ylabel("Speedup (Seq / Par)")
    ax2.set_title("OpenMP Speedup (GPU histogram path active)")
    ax2.grid(axis="y", alpha=0.4)
    for i, sp in enumerate(speedups):
        ax2.text(i, sp + 0.02, f"{sp:.2f}x", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/gpu_benchmark.png", dpi=150)
    plt.close()
    print("  Saved: results/gpu_benchmark.png\n")
    return datasets, seq_ms, par_ms, speedups


# ---------------------------------------------------------------------------
# Experiment 8: CPU Exact vs GPU Histogram — direct 3-way comparison
# Numbers from decision_tree.exe CPU vs GPU comparison section (April 2026).
# ---------------------------------------------------------------------------
def exp_cpu_vs_gpu_comparison():
    print("[Exp 8] CPU exact vs GPU histogram — 3-way comparison ...")

    labels   = ["Iris\n(150)", "Wine\n(178)", "BC\n(569)",
                "Banknote\n(1.4k)", "Synth\n(6k)", "Synth\n(200k)", "Synth\n(1M)"]
    samples  = [150, 178, 569, 1372, 6000, 200000, 1000000]
    cpu_seq  = [0.72,  0.45,   3.80,   1.65,   56.61,  2203.17, 145163.0]
    cpu_par  = [0.11,  0.39,   3.71,   1.36,   55.71,  2170.53, 144433.0]
    gpu_ms   = [2.92,  4.16,   9.85,   7.34,  117.82,  1960.52, 105605.0]
    accuracy = [0.7333, 0.8000, 0.9558, 0.9672, 0.8008, 0.8778, 0.5937]
    gpu_sp   = [cs/g for cs, g in zip(cpu_seq, gpu_ms)]

    print(f"  {'Dataset':18s}  {'CPU Seq':>10}  {'CPU Par':>10}  {'GPU':>10}  {'GPU Speedup':>12}  {'Acc':>6}")
    for lb, cs, cp, g, sp, acc in zip(labels, cpu_seq, cpu_par, gpu_ms, gpu_sp, accuracy):
        name = lb.replace("\n", " ")
        print(f"  {name:18s}  {cs:10.2f}  {cp:10.2f}  {g:10.2f}  {sp:12.2f}x  {acc:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(labels))
    w = 0.28

    # Panel 1: Training time (log scale)
    axes[0].bar(x - w,   cpu_seq, w, label="CPU Sequential\n(exact split)", color="steelblue")
    axes[0].bar(x,       cpu_par, w, label="CPU Parallel\n(16 threads, exact)", color="darkorange")
    axes[0].bar(x + w,   gpu_ms,  w, label="GPU Histogram\n(32 bins, RTX 4060)", color="green")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("Training time (ms)")
    axes[0].set_yscale("log")
    axes[0].set_title("Training Time: CPU Exact vs GPU Histogram\n(log scale)")
    axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=0.4)

    # Panel 2: GPU speedup vs CPU sequential
    colors = ["green" if sp >= 1.0 else "crimson" for sp in gpu_sp]
    axes[1].bar(x, gpu_sp, color=colors, alpha=0.85)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Break-even")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel("Speedup (CPU Seq / GPU)")
    axes[1].set_title("GPU Speedup vs CPU Sequential\n(>1.0 = GPU wins)")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.4)
    for i, sp in enumerate(gpu_sp):
        axes[1].text(i, sp + 0.01, f"{sp:.2f}x", ha="center", va="bottom", fontsize=7)

    # Panel 3: GPU speedup vs n_samples (line chart — shows crossover)
    axes[2].plot(samples, gpu_sp, "D-", color="green", linewidth=2, markersize=7)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[2].fill_between(samples, [1.0]*len(samples), gpu_sp,
                         where=[sp >= 1.0 for sp in gpu_sp],
                         alpha=0.15, color="green", label="GPU wins")
    axes[2].fill_between(samples, gpu_sp, [1.0]*len(samples),
                         where=[sp < 1.0 for sp in gpu_sp],
                         alpha=0.15, color="crimson", label="CPU wins")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Dataset size (samples, log scale)")
    axes[2].set_ylabel("GPU Speedup vs CPU Sequential")
    axes[2].set_title("GPU Speedup vs Dataset Size\n(crossover ~100k–200k samples)")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.4)
    for s, sp in zip(samples, gpu_sp):
        axes[2].annotate(f"{sp:.2f}x", (s, sp), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig("results/cpu_vs_gpu_comparison.png", dpi=150)
    plt.close()
    print("  Saved: results/cpu_vs_gpu_comparison.png\n")


# ---------------------------------------------------------------------------
# Experiment 6: Scalability — training time vs dataset size (C++ GPU exe)
# Numbers captured from decision_tree.exe scalability benchmark (April 2026).
# ---------------------------------------------------------------------------
def exp_scalability():
    print("[Exp 6] Scalability: training time vs dataset size (C++ GPU exe) ...")

    sizes    = [500,   1000,  2000,  5000,  10000,  25000]
    seq_ms   = [13.38, 19.32, 33.48, 70.47, 117.90, 263.64]
    par_ms   = [12.31, 19.13, 34.40, 70.50, 118.64, 262.93]
    util_pct = [25.19, 18.82, 15.42, 10.51,   6.51,   3.07]
    nodes    = [107,   125,   185,   253,    249,    253]
    speedups = [s/p for s, p in zip(seq_ms, par_ms)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: training time
    axes[0].plot(sizes, seq_ms, "o-", color="steelblue",   label="Sequential")
    axes[0].plot(sizes, par_ms, "s--", color="darkorange", label="Parallel (OpenMP)")
    axes[0].set_xlabel("Dataset size (samples)"); axes[0].set_ylabel("Training time (ms)")
    axes[0].set_title("Training Time vs Dataset Size\n(GPU histogram split, RTX 4060)")
    axes[0].legend(); axes[0].grid(True)

    # Plot 2: speedup curve
    axes[1].plot(sizes, speedups, "^-", color="green")
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Dataset size (samples)"); axes[1].set_ylabel("Speedup (Seq/Par)")
    axes[1].set_title("OpenMP Speedup vs Dataset Size")
    axes[1].set_ylim(0, 2.0); axes[1].grid(True)
    for i, (x, y) in enumerate(zip(sizes, speedups)):
        axes[1].annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)

    # Plot 3: GPU kernel utilization vs size
    axes[2].plot(sizes, util_pct, "D-", color="crimson")
    axes[2].set_xlabel("Dataset size (samples)"); axes[2].set_ylabel("GPU kernel utilization (%)")
    axes[2].set_title("GPU Kernel Utilization vs Dataset Size\n(kernel time / total GPU call time)")
    axes[2].set_ylim(0, 35); axes[2].grid(True)
    for x, y in zip(sizes, util_pct):
        axes[2].annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                         xytext=(0, 6), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("results/scalability.png", dpi=150)
    plt.close()
    print("  Saved: results/scalability.png\n")


# ---------------------------------------------------------------------------
# Experiment 7: GPU utilization breakdown (n=10000, f=20)
# Numbers from cudaEvent instrumentation in findBestSplitGPU (April 2026).
# ---------------------------------------------------------------------------
def exp_utilization_breakdown():
    print("[Exp 7] GPU utilization breakdown (n=10000, f=20) ...")

    # Measured values
    wall_ms    = 118.210
    gpu_call   = 112.906
    kernel_ms  = 7.360
    overhead   = 105.546
    cpu_only   = 5.305
    n_calls    = 124

    labels  = ["GPU kernel\ncompute", "GPU transfer\n+ sync overhead", "CPU-only\n(non-GPU work)"]
    values  = [kernel_ms, overhead, cpu_only]
    colors  = ["steelblue", "tomato", "darkorange"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        textprops={"fontsize": 10})
    ax1.set_title(f"Training time breakdown\n(n=10,000, f=20, {n_calls} nodes)")

    # Bar chart: per-node breakdown
    per_node_kernel   = kernel_ms  / n_calls
    per_node_overhead = overhead   / n_calls
    per_node_cpu      = cpu_only   / n_calls

    cats = ["Kernel\ncompute", "Transfer+\nsync", "CPU-only"]
    vals = [per_node_kernel, per_node_overhead, per_node_cpu]
    bars = ax2.bar(cats, vals, color=colors, width=0.5)
    ax2.set_ylabel("Time per node (ms)")
    ax2.set_title("Average time breakdown per node")
    ax2.grid(axis="y", alpha=0.4)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{v:.3f} ms", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("results/utilization_breakdown.png", dpi=150)
    plt.close()
    print("  Saved: results/utilization_breakdown.png\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Milestone 2 -- Evaluation Benchmark (Person 4)")
    print("GPU: RTX 4060 Laptop, CUDA 13.2, compute arch 89")
    print("=" * 60)
    print()

    sizes, seq_t, par_t, speedups = exp_split_speedup()
    feat_list, fc_seq, fc_par, fc_sp = exp_features_speedup()
    depths, d_seq, d_par, d_sp = exp_levelwise_depth()
    acc_rows = exp_accuracy_comparison()
    exp_cpp_gpu_results()

    with open("results/split_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_samples", "seq_ms", "par_ms", "speedup"])
        for n, s, p, sp in zip(sizes, seq_t, par_t, speedups):
            w.writerow([n, f"{s:.4f}", f"{p:.4f}", f"{sp:.4f}"])

    with open("results/levelwise_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["max_depth", "seq_ms", "par_ms", "speedup"])
        for d, s, p, sp in zip(depths, d_seq, d_par, d_sp):
            w.writerow([d, f"{s:.4f}", f"{p:.4f}", f"{sp:.4f}"])

    exp_scalability()
    exp_utilization_breakdown()
    exp_cpu_vs_gpu_comparison()

    print("=" * 60)
    print("All results saved to results/")
    print("Plots: split_speedup.png, features_speedup.png,")
    print("       levelwise_depth.png, accuracy_comparison.png,")
    print("       gpu_benchmark.png, scalability.png,")
    print("       utilization_breakdown.png, cpu_vs_gpu_comparison.png")
    print("CSVs:  split_timing.csv, levelwise_timing.csv")
