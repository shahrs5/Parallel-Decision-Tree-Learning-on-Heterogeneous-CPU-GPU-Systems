"""
benchmark_m3.py -- Milestone 3 Benchmark (Ensemble Learning + GPU Inference)

Experiments:
  1. Random Forest training time: CPU serial vs CPU parallel vs number of trees
  2. Speedup vs number of trees (OpenMP efficiency)
  3. Inference throughput: CPU serial vs CPU parallel vs GPU batch
  4. Accuracy vs number of trees (vs sklearn baseline)

Run from the decision-tree/ directory:
    python scripts/benchmark_m3.py
"""

import os
import sys
import csv
import subprocess
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import (load_breast_cancer, load_iris, load_wine,
                               make_classification)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# Script lives in decision-tree/scripts/; all other dirs are siblings.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Locate the compiled executable (works on Windows and Linux).
# CMake target name is 'decision_tree', so the binary is decision_tree[.exe].
# ---------------------------------------------------------------------------
def find_exe():
    # Prefer build_msvc (MSVC + CUDA) over older build dirs.
    # Paths are relative to decision-tree/ (parent of this script's dir).
    dt_dir = os.path.join(_SCRIPT_DIR, "..")
    candidates = [
        os.path.join(dt_dir, "build_msvc", "decision_tree.exe"),
        os.path.join(dt_dir, "build_msvc", "decision_tree"),
        os.path.join(dt_dir, "build",      "decision_tree.exe"),
        os.path.join(dt_dir, "build",      "decision_tree"),
        os.path.join(dt_dir, "build_cuda", "decision_tree.exe"),
        os.path.join(dt_dir, "build_cuda", "decision_tree"),
        os.path.join(dt_dir, "build",      "Release", "decision_tree.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

EXE = find_exe()

# ---------------------------------------------------------------------------
# Ensure small UCI datasets exist as CSV files for the C++ executable.
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data")

def ensure_datasets():
    """Generate breast_cancer, iris, wine CSVs if missing."""
    loaders = [
        ("breast_cancer", load_breast_cancer),
        ("iris",          load_iris),
        ("wine",          load_wine),
    ]
    for name, loader in loaders:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if not os.path.exists(path):
            print(f"  Generating {name}.csv ...")
            ds = loader()
            X, y = ds.data.astype(np.float32), ds.target.astype(np.int32)
            arr = np.column_stack([X, y])
            np.savetxt(path, arr, delimiter=",", fmt=["%.6f"] * X.shape[1] + ["%d"])
            print(f"    -> {len(X)} rows saved to {path}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_cmd(args, timeout=300):
    """Run a subprocess from the exe's directory so ../data/ resolves."""
    if EXE is None:
        return None
    exe_abs = os.path.abspath(EXE)
    exe_dir = os.path.dirname(exe_abs)
    try:
        result = subprocess.run(
            [exe_abs] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=exe_dir
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def parse_rf_result(out):
    """Parse '--benchmark-rf-*' output: 'time_ms acc' -> (float, float)."""
    if out is None:
        return -1.0, -1.0
    parts = out.split()
    if len(parts) >= 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    return -1.0, -1.0


def parse_infer_result(out):
    """Parse '--benchmark-infer-*' output: 'time_ms' -> float."""
    if out is None:
        return -1.0
    try:
        return float(out.split()[0])
    except (ValueError, IndexError):
        return -1.0


# ---------------------------------------------------------------------------
# Experiment 1 + 2: Training time & speedup vs n_trees
# ---------------------------------------------------------------------------
def benchmark_training(dataset, n_trees_list, max_depth=7, min_leaf=2):
    results = []
    for n in n_trees_list:
        cpu_t, cpu_a = parse_rf_result(
            run_cmd(["--benchmark-rf-cpu", dataset, str(n),
                     str(max_depth), str(min_leaf)]))
        gpu_t, gpu_a = parse_rf_result(
            run_cmd(["--benchmark-rf-gpu", dataset, str(n),
                     str(max_depth), str(min_leaf)]))
        results.append((n, cpu_t, cpu_a, gpu_t, gpu_a))
    return results


def benchmark_speedup(dataset, n_trees_list, max_depth=7, min_leaf=2):
    """Call --benchmark-rf-speedup which outputs seq/par per n_trees."""
    results = []
    out = run_cmd(["--benchmark-rf-speedup", dataset])
    if out is None:
        return [(n, -1.0, -1.0, -1.0) for n in n_trees_list]
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                n   = int(parts[0])
                seq = float(parts[1])
                par = float(parts[2])
                acc = float(parts[3])
                results.append((n, seq, par, acc))
            except ValueError:
                pass
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Inference throughput
# ---------------------------------------------------------------------------
def benchmark_inference(dataset, n_samples_list):
    results = []
    for n in n_samples_list:
        seq_t = parse_infer_result(
            run_cmd(["--benchmark-infer-seq", dataset, str(n)]))
        par_t = parse_infer_result(
            run_cmd(["--benchmark-infer-par", dataset, str(n)]))
        gpu_t = parse_infer_result(
            run_cmd(["--benchmark-infer-gpu", dataset, str(n)]))
        results.append((n, seq_t, par_t, gpu_t))
    return results


# ---------------------------------------------------------------------------
# Experiment 4: sklearn comparison
# ---------------------------------------------------------------------------
def sklearn_comparison(X_train, y_train, X_test, y_test, n_trees_list):
    results = []
    for n in n_trees_list:
        rf = RandomForestClassifier(n_estimators=n, max_depth=7,
                                    min_samples_leaf=2, random_state=42,
                                    n_jobs=-1)
        t0 = time.time()
        rf.fit(X_train, y_train)
        train_ms = (time.time() - t0) * 1000
        acc = rf.score(X_test, y_test)
        results.append((n, train_ms, acc))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Milestone 3 Benchmark: Random Forest Performance")
    print("=" * 60)

    if EXE is None:
        print("\n[WARN] C++ executable not found. Searched for decision_tree[.exe] "
              "in build/ and build_cuda/.\n"
              "       Run cmake + build first, then re-run this script.\n"
              "       sklearn-only plots will still be generated.")
    else:
        print(f"  Using executable: {EXE}")

    ensure_datasets()

    # Load breast_cancer for sklearn side.
    bc = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        bc.data, bc.target, test_size=0.2, random_state=42)

    n_trees_list   = [1, 5, 10, 20, 50]
    n_samples_list = [100, 500, 1000, 5000, 10000]
    dataset        = "breast_cancer"

    # ------------------------------------------------------------------
    # 1. Training time vs n_trees (C++ + sklearn)
    # ------------------------------------------------------------------
    print("\n[1/4] Training time vs number of trees ...")
    train_results  = benchmark_training(dataset, n_trees_list)
    sklearn_results = sklearn_comparison(X_train, y_train, X_test, y_test,
                                         n_trees_list)

    with open(os.path.join(RESULTS_DIR, "rf_training.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_trees", "cpu_time_ms", "cpu_acc",
                    "gpu_time_ms", "gpu_acc",
                    "sklearn_time_ms", "sklearn_acc"])
        for i, (n, ct, ca, gt, ga) in enumerate(train_results):
            st, sa = sklearn_results[i][1], sklearn_results[i][2]
            w.writerow([n, ct, ca, gt, ga, st, sa])

    n_arr      = [r[0] for r in train_results]
    cpu_times  = [r[1] for r in train_results]
    gpu_times  = [r[3] for r in train_results]
    sk_times   = [r[1] for r in sklearn_results]
    cpu_accs   = [r[2] for r in train_results]
    gpu_accs   = [r[4] for r in train_results]
    sk_accs    = [r[2] for r in sklearn_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if any(t > 0 for t in cpu_times):
        ax.plot(n_arr, cpu_times, label="C++ CPU parallel", marker="o")
    if any(t > 0 for t in gpu_times):
        ax.plot(n_arr, gpu_times, label="C++ GPU", marker="s")
    ax.plot(n_arr, sk_times, label="sklearn (n_jobs=-1)", marker="^")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Training Time (ms)")
    ax.set_title("RF Training Time vs Number of Trees")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    if any(a > 0 for a in cpu_accs):
        ax.plot(n_arr, cpu_accs, label="C++ CPU", marker="o")
    if any(a > 0 for a in gpu_accs):
        ax.plot(n_arr, gpu_accs, label="C++ GPU", marker="s")
    ax.plot(n_arr, sk_accs, label="sklearn", marker="^")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Accuracy")
    ax.set_title("RF Accuracy vs Number of Trees")
    ax.set_ylim(0.8, 1.0)
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rf_training_accuracy.png"), dpi=150)
    plt.close()
    print("  -> results/rf_training_accuracy.png")

    # ------------------------------------------------------------------
    # 2. Speedup vs n_trees (OpenMP efficiency)
    # ------------------------------------------------------------------
    print("\n[2/4] Speedup vs number of trees (OpenMP) ...")
    speedup_results = benchmark_speedup(dataset, n_trees_list)

    if speedup_results and speedup_results[0][1] > 0:
        with open(os.path.join(RESULTS_DIR, "rf_speedup.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_trees", "seq_ms", "par_ms", "speedup", "acc"])
            for (n, seq, par, acc) in speedup_results:
                sp = seq / par if par > 0 else 0.0
                w.writerow([n, seq, par, sp, acc])

        sn   = [r[0] for r in speedup_results]
        sseq = [r[1] for r in speedup_results]
        spar = [r[2] for r in speedup_results]
        ssp  = [r[1] / r[2] if r[2] > 0 else 0 for r in speedup_results]
        # Ideal speedup: if n_trees fit perfectly on all cores
        ideal = [min(n, 16) for n in sn]  # 16 logical cores

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(sn, sseq, label="Sequential (1 thread)", marker="o")
        ax.plot(sn, spar, label="Parallel (all threads)", marker="s")
        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Training Time (ms)")
        ax.set_title("Sequential vs Parallel Training Time")
        ax.legend(); ax.grid(True)

        ax = axes[1]
        ax.plot(sn, ssp,   label="Actual speedup",  marker="o")
        ax.plot(sn, ideal, label="Ideal (min(n,16))", marker="^", linestyle="--")
        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("Speedup")
        ax.set_title("Speedup vs Number of Trees (OpenMP)")
        ax.legend(); ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "rf_speedup.png"), dpi=150)
        plt.close()
        print("  -> results/rf_speedup.png")
    else:
        print("  [SKIP] No speedup data (C++ executable not available or failed)")

    # ------------------------------------------------------------------
    # 3. Inference throughput: CPU serial vs CPU parallel vs GPU
    # ------------------------------------------------------------------
    print("\n[3/4] Inference throughput ...")
    infer_results = benchmark_inference(dataset, n_samples_list)

    with open(os.path.join(RESULTS_DIR, "rf_inference.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_samples", "seq_ms", "par_ms", "gpu_ms"])
        for r in infer_results:
            w.writerow(r)

    in_n   = [r[0] for r in infer_results]
    in_seq = [r[1] for r in infer_results]
    in_par = [r[2] for r in infer_results]
    in_gpu = [r[3] for r in infer_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if any(t > 0 for t in in_seq):
        ax.plot(in_n, in_seq, label="CPU serial",   marker="o")
    if any(t > 0 for t in in_par):
        ax.plot(in_n, in_par, label="CPU parallel", marker="s")
    if any(t > 0 for t in in_gpu):
        ax.plot(in_n, in_gpu, label="GPU batch",    marker="^")
    ax.set_xlabel("Batch Size (samples)")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Throughput vs Batch Size")
    ax.legend(); ax.grid(True)

    ax = axes[1]
    gpu_sp  = [s / g if g > 0 else 0 for s, g in zip(in_seq, in_gpu)]
    par_sp  = [s / p if p > 0 else 0 for s, p in zip(in_seq, in_par)]
    if any(x > 0 for x in gpu_sp):
        ax.plot(in_n, gpu_sp, label="GPU vs serial",     marker="^")
    if any(x > 0 for x in par_sp):
        ax.plot(in_n, par_sp, label="CPU-par vs serial", marker="s")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Batch Size (samples)")
    ax.set_ylabel("Speedup over CPU serial")
    ax.set_title("Inference Speedup")
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rf_inference_time.png"), dpi=150)
    plt.close()
    print("  -> results/rf_inference_time.png")

    # ------------------------------------------------------------------
    # 4. Summary print
    # ------------------------------------------------------------------
    print("\n[4/4] Summary")
    print(f"{'N_Trees':>8} {'CPU_ms':>10} {'GPU_ms':>10} "
          f"{'CPU_acc':>9} {'sk_acc':>9}")
    print("-" * 52)
    for i, (n, ct, ca, gt, ga) in enumerate(train_results):
        st, sa = sklearn_results[i][1], sklearn_results[i][2]
        ct_s = f"{ct:.1f}" if ct > 0 else "N/A"
        gt_s = f"{gt:.1f}" if gt > 0 else "N/A"
        ca_s = f"{ca:.4f}" if ca > 0 else "N/A"
        print(f"{n:>8} {ct_s:>10} {gt_s:>10} {ca_s:>9} {sa:>9.4f}")

    print(f"\nAll benchmarks complete. Results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
