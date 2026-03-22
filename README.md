# Parallel Decision Tree Learning on Heterogeneous CPU–GPU Systems

---

## 📁 Project Structure

```text
decision-tree/
├── CMakeLists.txt          # C++17 build; CUDA disabled by default (enable in M2)
├── data/                   # Place datasets here (.csv)
├── src/
│   ├── main.cpp            # Verification tests for M1
│   ├── data_loader.h       # CSV loader (header-only)
│   ├── tree/
│   │   ├── node.h          # Node struct (array-of-structs layout)
│   │   ├── decision_tree.h # DecisionTree class interface
│   │   └── decision_tree.cpp  # Gini impurity, training, prediction
│   └── gpu/
│       └── split_kernel.cu # GPU split-finding kernels (M2 stub)
└── eval/
    ├── metrics.h
    ├── download_datasets.py
    └── sklearn_compare.py
```

---

## 1. Project Overview

A CART-style binary decision tree classifier implemented in **C++17**, progressively parallelised across milestones using CPU threading and CUDA.

### Milestones

| Milestone | Focus                                                    | Status     |
| --------- | -------------------------------------------------------- | ---------- |
| **M1**    | Sequential baseline (CART, Gini, CSV loader, benchmarks) | ✅ Complete |
| **M2**    | GPU split finding (histograms, level-wise expansion)     | ⏳ Planned  |
| **M3**    | Parallel random forest + inference optimisation          | ⏳ Planned  |

---

## 2. Repository Structure

| File                         | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| `CMakeLists.txt`             | Build config; enable CUDA with `-DENABLE_CUDA=ON` |
| `src/main.cpp`               | Entry point: tests + dataset benchmarks           |
| `src/data_loader.h`          | Header-only CSV loader                            |
| `src/tree/node.h`            | Node structure (array-of-structs)                 |
| `src/tree/decision_tree.h`   | Class interface                                   |
| `src/tree/decision_tree.cpp` | Training + prediction logic                       |
| `src/gpu/split_kernel.cu`    | GPU kernels (M2)                                  |
| `eval/metrics.h`             | Accuracy helper                                   |
| `eval/sklearn_compare.py`    | Comparison with sklearn                           |
| `eval/download_datasets.py`  | Dataset downloader                                |
| `data/`                      | Dataset storage                                   |

---

## 3. Algorithm

### CART Training

* Searches all features and thresholds
* Uses midpoints between sorted values
* Maximises **Gini gain**

### Gini Impurity

```
Gini = 1 − Σ p_k²
```

* 0 → pure node
* 0.5 → max impurity (binary)

### Stopping Conditions

* Node becomes pure
* Samples < `2 × min_samples_leaf`
* Max depth reached
* No positive split gain

### Node Storage

* Stored in `std::vector<Node>`
* Indexed children (no pointers)
* GPU-friendly design

---

## 4. Build & Run

### CPU (Milestone 1)

```bash
cd decision-tree
cmake -B build && cmake --build build
./build/decision_tree
```

### With CUDA (Milestone 2+)

```bash
cmake -B build -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
```

**GPU Architectures:**

* 75 → RTX 20xx
* 86 → RTX 30xx
* 89 → RTX 40xx

### Datasets

```bash
python eval/download_datasets.py
python eval/sklearn_compare.py
```

---

## 5. Datasets

| Dataset       | Samples | Features | Classes | Max Depth | Min Leaf | Purpose           |
| ------------- | ------- | -------- | ------- | --------- | -------- | ----------------- |
| Iris          | 150     | 4        | 3       | 5         | 1        | Simple baseline   |
| Wine          | 178     | 13       | 3       | 5         | 1        | Feature scaling   |
| Breast Cancer | 569     | 30       | 2       | 7         | 2        | High-dim dataset  |
| Banknote Auth | 1372    | 4        | 2       | 5         | 1        | Large sample test |

---

## 6. Results

| Dataset       | Impl    | Train (ms) | Infer (ms) | Accuracy | Nodes |
| ------------- | ------- | ---------- | ---------- | -------- | ----- |
| Iris          | C++     | 7.15       | 0.0064     | 0.9667   | 17    |
| Iris          | sklearn | 1.60       | 0.150      | 1.000    | 9     |
| Wine          | C++     | 57.68      | 0.0053     | 0.9143   | 19    |
| Wine          | sklearn | 1.70       | 0.184      | 0.9444   | —     |
| Breast Cancer | C++     | 1892.57    | 0.0170     | 0.9204   | 31    |
| Breast Cancer | sklearn | 7.60       | 0.184      | 0.9298   | —     |
| Banknote      | C++     | 1135.82    | 0.0379     | 0.9672   | 37    |
| Banknote      | sklearn | 4.43       | 0.181      | 0.9673   | —     |

---

## 7. Analysis

### Training Time

* Scales with **N × F × V**
* High features (Breast Cancer) → slow
* High samples (Banknote) → slow

### Key Insight

* Your bottleneck is **split search loop**
* sklearn is faster due to optimized backend

### Inference

* C++ is **20–400× faster**
* Reason: no Python overhead

### Accuracy

* Within **1–3% of sklearn**
* Differences due to tie-breaking

### Why GPU Matters

* Split search = fully parallel problem
* Histogram binning reduces complexity
* CUDA will directly attack bottleneck

---

## 8. Team Roles

* Student 1 → Sequential tree (M1 ✅)
* Student 2 → GPU kernels (M2)
* Student 3 → CPU–GPU coordination (M2)
* Student 4 → Ensemble + optimisation (M3)

---

### Notes

* This version focuses on **readability + structure**
* Tables replace messy ASCII formatting
* Clean sections improve GitHub presentation

---