"""
run_m3_report_benchmarks.py
Generates all tables and figures needed for the M3 report.
Run from decision-tree/ directory.
"""

import os, sys, subprocess, csv, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_HERE, "..", "data")
RESULTS_DIR = os.path.join(_HERE, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- locate exe ----
def find_exe():
    dt = os.path.join(_HERE, "..")
    for name in ["build_msvc", "build", "build_cuda"]:
        for ext in ["decision_tree.exe", "decision_tree"]:
            p = os.path.join(dt, name, ext)
            if os.path.isfile(p):
                return os.path.abspath(p)
    return None

EXE = find_exe()
EXE_DIR = os.path.dirname(EXE) if EXE else None

def run(args, timeout=600):
    if EXE is None: return None
    try:
        r = subprocess.run([EXE]+args, capture_output=True, text=True,
                           timeout=timeout, cwd=EXE_DIR)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception as e:
        print(f"  [ERR] {args[0]}: {e}")
        return None

def parse2(out):
    if out is None: return -1.0, -1.0
    p = out.split()
    try: return float(p[0]), float(p[1])
    except: return -1.0, -1.0

def parse1(out):
    if out is None: return -1.0
    try: return float(out.split()[0])
    except: return -1.0

# ---- datasets for multi-dataset table ----
DATASETS = [
    ("Iris",          "iris",          5,  1),
    ("Wine",          "wine",          5,  1),
    ("Breast Cancer", "breast_cancer", 7,  2),
    ("Banknote",      "banknote",      5,  1),
    ("Synthetic 6k",  "synthetic",     8,  2),
]
N_TREES_LIST  = [1, 5, 10, 20, 50]
BATCH_SIZES   = [100, 500, 1000, 5000, 10000, 50000]

# ============================================================
# E1: Multi-dataset accuracy: single tree vs RF-10 vs sklearn
# ============================================================
print("="*60)
print("E1: Accuracy — single tree vs RF-10 vs sklearn RF")
print("="*60)
acc_table = []   # (dataset, n_samples, n_feat, acc_1tree, acc_rf10_cpu, acc_rf10_gpu, acc_sklearn)

for name, ds, depth, leaf in DATASETS:
    path = os.path.join(DATA_DIR, ds+".csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {ds}.csv not found"); continue
    print(f"  {name} ...", end=" ", flush=True)

    # single tree (n_trees=1)
    t1, a1 = parse2(run(["--benchmark-rf-cpu", ds, "1", str(depth), str(leaf)]))
    # RF-10 CPU
    t10c, a10c = parse2(run(["--benchmark-rf-cpu", ds, "10", str(depth), str(leaf)]))
    # RF-10 GPU
    t10g, a10g = parse2(run(["--benchmark-rf-gpu", ds, "10", str(depth), str(leaf)]))

    # sklearn RF-10
    data = np.loadtxt(path, delimiter=",")
    X, y = data[:, :-1].astype(np.float32), data[:, -1].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sk = RandomForestClassifier(n_estimators=10, max_depth=depth,
                                min_samples_leaf=leaf, random_state=42, n_jobs=-1)
    sk.fit(Xtr, ytr); sk_acc = sk.score(Xte, yte)

    n_samples, n_feat = len(X), X.shape[1]
    acc_table.append((name, n_samples, n_feat, a1, a10c, a10g, sk_acc,
                      t1, t10c, t10g))
    print(f"tree={a1:.3f}  rf10={a10c:.3f}  gpu={a10g:.3f}  sk={sk_acc:.3f}")

# ============================================================
# E2: Training time vs n_trees (breast_cancer focus)
# ============================================================
print("\n"+"="*60)
print("E2: Training time vs n_trees (Breast Cancer)")
print("="*60)
train_rows = []   # (n_trees, cpu_seq_ms, cpu_par_ms, gpu_ms, acc)
speedup_rows = [] # (n_trees, seq_ms, par_ms, acc)

sp_out = run(["--benchmark-rf-speedup", "breast_cancer"])
if sp_out:
    for line in sp_out.splitlines():
        p = line.split()
        if len(p) >= 4:
            try:
                nt, seq, par, acc = int(p[0]), float(p[1]), float(p[2]), float(p[3])
                speedup_rows.append((nt, seq, par, acc))
                print(f"  n={nt:3d}  seq={seq:.1f}ms  par={par:.1f}ms  sp={seq/par:.2f}x")
            except: pass

for nt in N_TREES_LIST:
    tc, ac = parse2(run(["--benchmark-rf-cpu", "breast_cancer", str(nt), "7", "2"]))
    tg, ag = parse2(run(["--benchmark-rf-gpu", "breast_cancer", str(nt), "7", "2"]))
    train_rows.append((nt, tc, tg, ac))
    print(f"  n={nt:3d}  cpu={tc:.1f}ms  gpu={tg:.1f}ms  acc={ac:.3f}")

# ============================================================
# E3: Inference throughput
# ============================================================
print("\n"+"="*60)
print("E3: Inference throughput vs batch size")
print("="*60)
infer_rows = []
for bs in BATCH_SIZES:
    ts = parse1(run(["--benchmark-infer-seq", "breast_cancer", str(bs)]))
    tp = parse1(run(["--benchmark-infer-par", "breast_cancer", str(bs)]))
    tg = parse1(run(["--benchmark-infer-gpu", "breast_cancer", str(bs)]))
    infer_rows.append((bs, ts, tp, tg))
    sp_g = ts/tg if tg > 0 else 0
    sp_p = ts/tp if tp > 0 else 0
    print(f"  batch={bs:6d}  seq={ts:.2f}  par={tp:.2f}  gpu={tg:.2f}  sp_gpu={sp_g:.2f}x")

# ============================================================
# Save CSVs
# ============================================================
with open(os.path.join(RESULTS_DIR, "m3_accuracy.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset","n_samples","n_features","acc_1tree","acc_rf10_cpu","acc_rf10_gpu","acc_sklearn","t_1tree_ms","t_rf10_cpu_ms","t_rf10_gpu_ms"])
    for row in acc_table: w.writerow(row)

with open(os.path.join(RESULTS_DIR, "m3_training.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_trees","cpu_time_ms","gpu_time_ms","accuracy"])
    for row in train_rows: w.writerow(row)

with open(os.path.join(RESULTS_DIR, "m3_speedup.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_trees","seq_ms","par_ms","speedup","acc"])
    for (nt, s, p, a) in speedup_rows:
        w.writerow([nt, s, p, s/p if p>0 else 0, a])

with open(os.path.join(RESULTS_DIR, "m3_inference.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["batch_size","seq_ms","par_ms","gpu_ms"])
    for row in infer_rows: w.writerow(row)

# ============================================================
# PLOTS
# ============================================================
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.4})

# ---- Figure 1: Training time and accuracy vs n_trees ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Random Forest — Training Performance vs Number of Trees\n(Breast Cancer, depth=7)", fontsize=12)

ns  = [r[0] for r in train_rows]
ct  = [r[1] for r in train_rows]
gt  = [r[2] for r in train_rows]
acc = [r[3] for r in train_rows]

ax = axes[0]
ax.plot(ns, ct, marker="o", label="CPU parallel (16 threads)")
ax.plot(ns, gt, marker="s", linestyle="--", label="GPU (sequential trees)")
ax.set_xlabel("Number of Trees"); ax.set_ylabel("Training Time (ms)")
ax.set_title("Training Time vs Number of Trees"); ax.legend()

ax = axes[1]
ax.plot(ns, acc, marker="o", color="green", label="RF accuracy (CPU)")
ax.axhline(acc_table[2][3] if len(acc_table) > 2 else 0.956,
           color="gray", linestyle="--", linewidth=1.2, label="Single tree (M2 baseline)")
ax.set_xlabel("Number of Trees"); ax.set_ylabel("Accuracy (20% test split)")
ax.set_title("Accuracy vs Number of Trees"); ax.legend(); ax.set_ylim(0.88, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "m3_training_accuracy.png"), dpi=150)
plt.close(); print("\n-> m3_training_accuracy.png")

# ---- Figure 2: Speedup vs n_trees ----
if speedup_rows:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("OpenMP Speedup — Parallel vs Sequential Training\n(Breast Cancer)", fontsize=12)

    sn  = [r[0] for r in speedup_rows]
    ss  = [r[1] for r in speedup_rows]
    sp  = [r[2] for r in speedup_rows]
    ssp = [s/p if p > 0 else 0 for s, p in zip(ss, sp)]
    ideal = [min(n, 16) for n in sn]

    ax = axes[0]
    ax.plot(sn, ss, marker="o", label="Sequential (1 thread)")
    ax.plot(sn, sp, marker="s", label="Parallel (16 threads)")
    ax.set_xlabel("Number of Trees"); ax.set_ylabel("Training Time (ms)")
    ax.set_title("Sequential vs Parallel Training Time"); ax.legend()

    ax = axes[1]
    ax.plot(sn, ssp,   marker="o", label="Actual speedup")
    ax.plot(sn, ideal, marker="^", linestyle="--", color="gray", label="Ideal (min(n,16)×)")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel("Number of Trees"); ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Number of Trees (OpenMP)"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "m3_speedup.png"), dpi=150)
    plt.close(); print("-> m3_speedup.png")

# ---- Figure 3: Inference throughput ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Inference Throughput — CPU vs GPU\n(10-tree RF, Breast Cancer)", fontsize=12)

bn  = [r[0] for r in infer_rows]
its = [r[1] for r in infer_rows]
itp = [r[2] for r in infer_rows]
itg = [r[3] for r in infer_rows]

ax = axes[0]
ax.plot(bn, its, marker="o", label="CPU serial (1 thread)")
ax.plot(bn, itp, marker="s", label="CPU parallel (16 threads)")
ax.plot(bn, [g if g > 0 else None for g in itg], marker="^", label="GPU batch kernel")
ax.set_xlabel("Batch Size"); ax.set_ylabel("Inference Time (ms)")
ax.set_title("Inference Time vs Batch Size"); ax.legend()

ax = axes[1]
gpu_sp = [s/g if g > 0 else None for s, g in zip(its, itg)]
par_sp = [s/p if p > 0 else None for s, p in zip(its, itp)]
ax.plot(bn, gpu_sp, marker="^", label="GPU / CPU-serial")
ax.plot(bn, par_sp, marker="s", label="CPU-par / CPU-serial")
ax.axhline(1.0, color="red", linestyle=":", linewidth=1, label="Breakeven")
ax.set_xlabel("Batch Size"); ax.set_ylabel("Speedup over CPU serial")
ax.set_title("Inference Speedup"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "m3_inference.png"), dpi=150)
plt.close(); print("-> m3_inference.png")

# ---- Figure 4: Multi-dataset accuracy comparison ----
if acc_table:
    fig, ax = plt.subplots(figsize=(11, 5))
    ds_names = [r[0] for r in acc_table]
    a1t  = [r[3] for r in acc_table]
    a10c = [r[4] for r in acc_table]
    a10g = [r[5] for r in acc_table]
    ask  = [r[6] for r in acc_table]

    x = np.arange(len(ds_names)); w = 0.2
    ax.bar(x - 1.5*w, a1t,  w, label="Single tree (M2)")
    ax.bar(x - 0.5*w, a10c, w, label="RF-10 CPU")
    ax.bar(x + 0.5*w, a10g, w, label="RF-10 GPU")
    ax.bar(x + 1.5*w, ask,  w, label="sklearn RF-10")
    ax.set_xticks(x); ax.set_xticklabels(ds_names, rotation=12)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.7, 1.0)
    ax.set_title("Accuracy: Single Tree (M2) vs Random Forest (M3)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "m3_accuracy_comparison.png"), dpi=150)
    plt.close(); print("-> m3_accuracy_comparison.png")

print("\n=== All benchmarks complete ===")

# Print summary tables for copy-paste into report
print("\n--- E1: Accuracy table ---")
print(f"{'Dataset':<18} {'N':>6} {'F':>4} {'1-Tree':>8} {'RF-10C':>8} {'RF-10G':>8} {'sklearn':>8}")
print("-"*64)
for (nm, n, f, a1, a10c, a10g, ask, t1, t10c, t10g) in acc_table:
    print(f"{nm:<18} {n:>6} {f:>4} {a1:>8.4f} {a10c:>8.4f} {a10g:>8.4f} {ask:>8.4f}")

print("\n--- E2: Training time vs n_trees (BC) ---")
print(f"{'N_Trees':>8} {'CPU (ms)':>10} {'GPU (ms)':>10} {'Accuracy':>10}")
print("-"*42)
for (nt, tc, tg, ac) in train_rows:
    print(f"{nt:>8} {tc:>10.1f} {tg:>10.1f} {ac:>10.4f}")

print("\n--- E3: Inference (BC, 10-tree RF) ---")
print(f"{'Batch':>8} {'Seq (ms)':>10} {'Par (ms)':>10} {'GPU (ms)':>10} {'GPU sp':>8}")
print("-"*52)
for (bs, ts, tp, tg) in infer_rows:
    sp = ts/tg if tg > 0 else 0
    print(f"{bs:>8} {ts:>10.2f} {tp:>10.2f} {tg:>10.2f} {sp:>8.2f}x")
