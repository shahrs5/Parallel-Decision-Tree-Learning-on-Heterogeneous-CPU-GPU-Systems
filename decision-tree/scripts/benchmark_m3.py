"""
benchmark_m3.py -- Milestone 3 Benchmark (Ensemble Learning)

Experiments:
  1. Random Forest training time: CPU vs GPU vs number of trees
  2. Inference throughput: serial vs parallel batch
  3. Accuracy vs number of trees
  4. Comparison with sklearn RandomForest

Run: python scripts/benchmark_m3.py  (from decision-tree/ directory)
"""

import time
import os
import csv
import subprocess
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)


def run_cpp_benchmark(dataset_name, n_trees_list, max_depth=7, min_leaf=2):
    """Run C++ benchmark for different n_trees, return times and accuracies."""
    results = []
    for n_trees in n_trees_list:
        # Run CPU version
        cmd_cpu = [
            "./build/decision-tree.exe",
            "--benchmark-rf-cpu",
            dataset_name,
            str(n_trees),
            str(max_depth),
            str(min_leaf)
        ]
        try:
            result_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=300)
            cpu_time = float(result_cpu.stdout.strip().split()[0])
            cpu_acc = float(result_cpu.stdout.strip().split()[1])
        except:
            cpu_time, cpu_acc = -1, -1

        # Run GPU version if available
        cmd_gpu = [
            "./build/decision-tree.exe",
            "--benchmark-rf-gpu",
            dataset_name,
            str(n_trees),
            str(max_depth),
            str(min_leaf)
        ]
        try:
            result_gpu = subprocess.run(cmd_gpu, capture_output=True, text=True, timeout=300)
            gpu_time = float(result_gpu.stdout.strip().split()[0])
            gpu_acc = float(result_gpu.stdout.strip().split()[1])
        except:
            gpu_time, gpu_acc = -1, -1

        results.append((n_trees, cpu_time, cpu_acc, gpu_time, gpu_acc))
    return results


def benchmark_inference_throughput(dataset_name, n_samples_list):
    """Benchmark inference throughput for different batch sizes."""
    results = []
    for n_samples in n_samples_list:
        # Serial inference
        cmd_seq = [
            "./build/decision-tree.exe",
            "--benchmark-infer-seq",
            dataset_name,
            str(n_samples)
        ]
        try:
            result_seq = subprocess.run(cmd_seq, capture_output=True, text=True, timeout=60)
            seq_time = float(result_seq.stdout.strip())
        except:
            seq_time = -1

        # Parallel inference
        cmd_par = [
            "./build/decision-tree.exe",
            "--benchmark-infer-par",
            dataset_name,
            str(n_samples)
        ]
        try:
            result_par = subprocess.run(cmd_par, capture_output=True, text=True, timeout=60)
            par_time = float(result_par.stdout.strip())
        except:
            par_time = -1

        results.append((n_samples, seq_time, par_time))
    return results


def sklearn_comparison(X_train, y_train, X_test, y_test, n_trees_list):
    """Compare with sklearn RandomForest."""
    results = []
    for n_trees in n_trees_list:
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=7, min_samples_leaf=2, random_state=42)
        start = time.time()
        rf.fit(X_train, y_train)
        train_time = (time.time() - start) * 1000  # ms
        acc = rf.score(X_test, y_test)
        results.append((n_trees, train_time, acc))
    return results


def main():
    print("Milestone 3 Benchmark: Random Forest Performance")

    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_trees_list = [1, 5, 10, 20, 50]

    # 1. Training time vs n_trees
    print("1. Training time vs number of trees")
    cpp_results = run_cpp_benchmark("breast_cancer", n_trees_list)
    sklearn_results = sklearn_comparison(X_train, y_train, X_test, y_test, n_trees_list)

    with open("results/rf_training.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_trees", "cpu_time_ms", "cpu_acc", "gpu_time_ms", "gpu_acc", "sklearn_time_ms", "sklearn_acc"])
        for i, (n_trees, cpu_t, cpu_a, gpu_t, gpu_a) in enumerate(cpp_results):
            sk_t, sk_a = sklearn_results[i][1], sklearn_results[i][2]
            writer.writerow([n_trees, cpu_t, cpu_a, gpu_t, gpu_a, sk_t, sk_a])

    # Plot training time
    n_trees = [r[0] for r in cpp_results]
    cpu_times = [r[1] for r in cpp_results if r[1] > 0]
    gpu_times = [r[3] for r in cpp_results if r[3] > 0]
    sk_times = [r[1] for r in sklearn_results]

    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, cpu_times, label="C++ CPU", marker="o")
    if gpu_times:
        plt.plot(n_trees, gpu_times, label="C++ GPU", marker="s")
    plt.plot(n_trees, sk_times, label="sklearn", marker="^")
    plt.xlabel("Number of Trees")
    plt.ylabel("Training Time (ms)")
    plt.title("Random Forest Training Time vs Number of Trees")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/rf_training_time.png")
    plt.close()

    # 2. Inference throughput
    print("2. Inference throughput")
    n_samples_list = [100, 500, 1000, 5000, 10000]
    infer_results = benchmark_inference_throughput("breast_cancer", n_samples_list)

    with open("results/rf_inference.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_samples", "seq_time_ms", "par_time_ms"])
        for r in infer_results:
            writer.writerow(r)

    # Plot inference
    n_samples = [r[0] for r in infer_results]
    seq_times = [r[1] for r in infer_results if r[1] > 0]
    par_times = [r[2] for r in infer_results if r[2] > 0]

    plt.figure(figsize=(10, 6))
    plt.plot(n_samples, seq_times, label="Serial", marker="o")
    plt.plot(n_samples, par_times, label="Parallel", marker="s")
    plt.xlabel("Number of Samples")
    plt.ylabel("Inference Time (ms)")
    plt.title("Random Forest Inference Time vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/rf_inference_time.png")
    plt.close()

    # 3. Accuracy vs n_trees
    print("3. Accuracy vs number of trees")
    cpu_accs = [r[2] for r in cpp_results if r[2] > 0]
    gpu_accs = [r[4] for r in cpp_results if r[4] > 0]
    sk_accs = [r[2] for r in sklearn_results]

    plt.figure(figsize=(10, 6))
    plt.plot(n_trees, cpu_accs, label="C++ CPU", marker="o")
    if gpu_accs:
        plt.plot(n_trees, gpu_accs, label="C++ GPU", marker="s")
    plt.plot(n_trees, sk_accs, label="sklearn", marker="^")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.title("Random Forest Accuracy vs Number of Trees")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/rf_accuracy.png")
    plt.close()

    print("Benchmarks completed. Results saved in results/ directory.")


if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\aqibs\OneDrive\Documents\PDC\Parallel-Decision-Tree-Learning-on-Heterogeneous-CPU-GPU-Systems\decision-tree\scripts\benchmark_m3.py