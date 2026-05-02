# M3 Change Log

Running log of every file added or modified for Milestone 3 (Random Forest).
Newest entries at the bottom of each section.

---

## Step 1 — RandomForest skeleton + bootstrap sampling (serial)

**Goal:** stand up the ensemble class end-to-end with the simplest possible
implementation. Serial training (no tree-level OpenMP yet), no feature
subsampling yet, basic majority-vote inference. Get it compiling and producing
correct predictions before we layer parallelism on top.

### New files

- **`decision-tree/src/tree/random_forest.h`**
  - Declares `class RandomForest`.
  - Public API: `train()`, `predict()`, `predictBatch()`, `n_trees()`.
  - Holds `std::vector<std::unique_ptr<DecisionTree>>`.
    - Why `unique_ptr`: `DecisionTree` owns raw GPU pointers (`d_X_`, `d_y_`)
      with a destructor that frees them. The default-generated move/copy
      semantics would alias those pointers and cause a double-free. Wrapping
      in `unique_ptr` sidesteps the problem without modifying `DecisionTree`.
  - Private static `bootstrapSample()` helper.

- **`decision-tree/src/tree/random_forest.cpp`**
  - Constructor validates `n_trees >= 1`.
  - `train()`:
    - Creates `n_trees_` `DecisionTree` instances.
    - Pre-generates one deterministic seed per tree from the master seed,
      so bootstrap samples are reproducible across runs.
    - Serial loop: bootstrap-sample → `tree->train(X_boot, y_boot)`.
    - Tree-level OpenMP is intentionally deferred to a later step.
  - `bootstrapSample()`:
    - `n` draws with replacement using `uniform_int_distribution`.
    - Builds new `X_boot`, `y_boot` vectors. (Refactor to pass index lists
      directly into `DecisionTree::train` is a possible later optimisation,
      but copy cost is negligible vs training cost and keeps the existing
      `train()` signature untouched.)
  - `predict()`:
    - Loop trees, count votes in `std::map<int,int>`, return arg-max.
  - `predictBatch()`:
    - Serial loop over samples calling `predict()`. Parallel batch inference
      will come in step 5.

### Modified files

- **`decision-tree/CMakeLists.txt`**
  - Added `src/tree/random_forest.cpp` to `SOURCES`.

- **`decision-tree/src/main.cpp`**
  - Added a small RF smoke test after the existing unit tests:
    trains a 5-tree forest on Breast Cancer, prints accuracy. Confirms the
    new class compiles, links, and produces sensible output. Will be replaced
    with full benchmarks (teammate B's ticket) later.

### What is *not* yet done in step 1

- Per-split feature subsampling (step 3).
- Tree-level OpenMP (step 4).
- Parallel batch inference (step 5).
- The nested-OMP guard inside `trainLevelWise` is not yet needed because
  step 1 trains trees serially.

### Verification

Build: CPU + OpenMP (Ninja + MSVC 19.44, Release).  Result: **clean build**;
only the pre-existing `data_loader.h:72` C4834 warning carried over from M2.

Smoke test on Breast Cancer (5 trees, depth 7, min_leaf 2):
- Train time: 33.2 ms (serial)
- Test accuracy: **0.9558**
- `[PASS] RandomForest accuracy >= 0.85 on Breast Cancer`

Note: accuracy matches the single-tree M2 number because every tree currently
sees the same feature pool — there is no per-split feature subsampling yet,
so the trees lack diversity. Step 3 will introduce that.

---

## Step 3 — Per-split feature subsampling

**Goal:** at every split, evaluate only a random subset of features (default
`sqrt(F)`), the standard mechanism that makes a Random Forest a *random*
forest. Without this, an ensemble of bootstrap-trained trees is just bagging,
and the trees are too correlated for the majority vote to help.

### Modified files

- **`decision-tree/src/tree/decision_tree.h`**
  - Constructor now takes optional `int feature_subsample = -1` and
    `unsigned seed = 42`. `-1` keeps the M2 single-tree behavior (use all
    features), `>0` activates per-split subsampling.
  - New private members: `feature_subsample_`, `tree_seed_`.
  - `findBestSplitForNode` gains a `const std::vector<int>& feature_subset`
    parameter. Empty vector means "evaluate every feature" — the M2 path is
    unchanged.

- **`decision-tree/src/tree/decision_tree.cpp`**
  - Constructor stores the new parameters.
  - `trainLevelWise` now owns a `std::mt19937 tree_rng(tree_seed_)` and, at
    the start of each level, **serially** generates one feature subset per
    pending node via partial Fisher-Yates shuffle. Generating subsets in
    serial code (before the OMP parallel-for) keeps the RNG sequence
    deterministic regardless of thread scheduling and avoids races.
  - `findBestSplitForNode`: the feature loop now iterates over the supplied
    subset (or all features when empty). Result slots for non-evaluated
    features stay at `-inf` and are filtered out by the existing reduction.
  - Legacy recursive `buildNode` now passes an empty subset to keep the M2
    code path identical.

- **`decision-tree/src/gpu/split_kernel.cuh`**
  - `findBestSplitGPU` adds two parameters: `const int* feature_subset` and
    `int n_subset`. Pass `nullptr / 0` to consider all features (M2 default).

- **`decision-tree/src/gpu/split_kernel.cu`**
  - Host-side argmax restricted to the supplied feature subset when given.
  - **Known suboptimality**: the kernel still launches one block per feature
    and computes histograms for every feature, then we throw most away. With
    `sqrt(F)` subsampling at F=30 that wastes ~80% of GPU compute. Future
    optimisation: launch only `n_subset` blocks indexed via a device-side
    subset array. Comment-flagged in the kernel.

- **`decision-tree/src/tree/random_forest.h`**
  - Constructor takes `int feature_subsample = -1` (sqrt(F) default), 0 for
    bagging-only, or an explicit positive count.

- **`decision-tree/src/tree/random_forest.cpp`**
  - Resolves the effective subsample size once `n_features` is known:
    `-1 → max(1, sqrt(F))`, `0 or >=F → use all`, otherwise the supplied
    value. The resolved value is forwarded into each `DecisionTree` along
    with that tree's own seed.

- **`decision-tree/src/main.cpp`**
  - Smoke test extended to compare three configurations on Breast Cancer:
    single tree, 10-tree forest with all features (bagging), 10-tree forest
    with sqrt(F) subsampling (full RF).

### Verification

Build: clean (only the pre-existing C4834 warning).

Breast Cancer test split, 10 trees, depth 7, min_leaf 2:

| Configuration | Train (ms) | Test acc |
|---|---|---|
| Single tree | 13.8 | 0.9469 |
| Forest, all features (bagging) | 36.8 | 0.9381 |
| Forest, sqrt(F)=5 features | **22.3** | **0.9735** |

Two observations worth recording for the report:

1. **Bagging alone hurts accuracy** (0.9381 < 0.9469). Bootstrap-only trees
   pick the same dominant features, so they make correlated mistakes and the
   majority vote amplifies them.
2. **Feature subsampling fixes both accuracy and speed.** Decorrelating the
   trees brings accuracy to 0.9735, *and* training drops to 22 ms because
   each split only sorts/sweeps ~5 features instead of 30 — roughly 6× less
   per-split work, which more than offsets the 10× tree count vs the single
   tree baseline.

---

## Steps 4 + 5 — Tree-level OpenMP training and parallel batch inference

**Goal:** training trees in an RF is embarrassingly parallel (independent
bootstrap samples, independent feature subsets), so we put the OpenMP
parallel-for at the *tree* level instead of the per-tree internal level.
At the same time, parallelise `predictBatch` over samples since per-sample
inference is independent.

### GPU vs CPU under tree-level parallelism — design call

We force the **CPU exact path** for every tree when `RandomForest::train`
runs. Two reasons:
1. Multiple host threads launching CUDA kernels would serialise on the
   single device anyway — there is one queue, not 16 — so tree-level OMP
   buys nothing on GPU.
2. The timing globals in `split_kernel.cu` (`g_total_kernel_ms`,
   `g_n_gpu_calls`) are not thread-safe; concurrent updates would race
   and produce nonsense numbers.

The single-tree GPU path is *unchanged* — the M2 benchmarks in `main.cpp`
do not go through `RandomForest`. Within a forest each tree sets
`use_gpu_=false` before its `train()` call, and we skip the device upload
entirely so each tree pays zero CUDA cost.

### Modified files

- **`decision-tree/src/tree/decision_tree.cpp`**
  - `train()`: the `cudaMalloc + cudaMemcpy` upload is now gated on
    `use_gpu_`. CPU-forced trees skip it. Necessary for tree-level OMP to
    not pay 10× upload cost for nothing.
  - `trainLevelWise`: node-level pragma now also disables when
    `omp_in_parallel()` is true. With tree-level OMP active above us,
    the inner pragma was either no-op or oversubscribing cores.

- **`decision-tree/src/tree/random_forest.cpp`**
  - Each `DecisionTree` is created and immediately has `setUseGPU(false)`
    called on it.
  - The per-tree training loop is wrapped in
    `#pragma omp parallel for schedule(dynamic) if(n_trees_ > 1)`. Dynamic
    schedule because tree training time varies with bootstrap composition.
  - `predictBatch`: sample-level parallel-for with
    `if(n >= 64)` guard so tiny batches don't pay OMP overhead.

- **`decision-tree/src/main.cpp`**
  - Smoke test extended to:
    - Time RF training serially (`omp_set_num_threads(1)`) vs parallel
      (max threads), report both accuracy and speedup ratio.
    - Time `predictBatch` over 22,600 samples (replicated test set) under
      both thread counts.
    - Assert parallel == serial accuracy and parallel == serial
      predictions (vector equality).

### Verification

Build: clean. Smoke-test results (Breast Cancer, 10 trees, depth 7):

```
Single tree              : 10.62 ms,  acc=0.9469
Forest (bagging,    10)  :  6.50 ms,  acc=0.9381
Forest (sqrt(F),seq,10)  :  5.44 ms,  acc=0.9735
Forest (sqrt(F),par,10)  :  4.76 ms,  acc=0.9735  speedup=1.14x
Batch inference (22600 samples):
    serial   : 2.98 ms
    parallel : 3.19 ms  speedup=0.94x
Predictions match (par vs seq): YES
[PASS] RF parallel accuracy >= 0.85 on Breast Cancer
[PASS] Bagging accuracy >= 0.85
[PASS] Parallel and serial RF train -> same accuracy
[PASS] Parallel batch inference matches serial
```

**Correctness checks all pass** — most importantly, parallel == serial
predictions, vector-equal. The mechanism is right.

**Speedup numbers are honest**: Breast Cancer is too small to amortise
OpenMP fork/join overhead. Each tree trains in ~0.5 ms; with 16 threads
the parallel-region setup cost is roughly the same magnitude as the work.
Inference hits the same wall — the entire 22,600-sample batch finishes
in 3 ms (≈7 M predictions/sec, well into cache) so spawning 16 threads
costs more than they save.

The places this *will* show real speedup are the larger M2 datasets:
synthetic 200k / 1M, where each tree takes 100s of ms to seconds.
Those measurements belong to teammate B's benchmark suite.

### Notes for teammate B (benchmarks)

- Use `synthetic_200k.csv` and `synthetic_1m.csv` for training-time
  speedup curves. Vary `n_trees` ∈ {1, 2, 4, 8, 10}; sweep
  `omp_set_num_threads` ∈ {1, 2, 4, 8, 16}.
- Inference throughput plot: replicate test set up to ≥1 M samples to
  push past OMP overhead.
- Accuracy curve vs sklearn `RandomForestClassifier(n_estimators=10,
  max_features='sqrt')` — should be within 1–2% on every dataset.

### Notes for teammate A (compact node layout)

- The `predict()` path currently traverses `nodes_` (a `std::vector<Node>`)
  via `unique_ptr<DecisionTree>` indirection. Two cache-miss layers per
  prediction. Converting to an SoA layout
  (`feature_index[]`, `threshold[]`, `left[]`, `right[]`, `label[]`)
  packed inside `RandomForest` should give a measurable boost — the M3
  smoke test above is already inference-bottlenecked by cache, not OMP.

