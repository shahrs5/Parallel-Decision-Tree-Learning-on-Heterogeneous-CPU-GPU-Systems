# Project 26: Parallel Decision Tree Learning on Heterogeneous CPU–GPU Systems

---

## Problem Statement

Decision tree learning is a widely used machine learning technique, but its training process involves **irregular control flow, data-dependent branching, and expensive split evaluation**, making it difficult to parallelize efficiently—especially on GPUs. While GPUs excel at dense numerical workloads, decision trees require careful **CPU–GPU coordination** to achieve meaningful speedups.

The goal of this project is to **design and implement a heterogeneous decision tree training pipeline** that leverages both CPU and GPU, explicitly addressing challenges in **task decomposition, memory management, and irregular parallel computation**. Students will explore how classical machine learning algorithms must be restructured to perform efficiently on modern heterogeneous systems.

---

## Project Overview

Students will implement a **parallel decision tree classifier** that:

* trains on large numerical datasets,
* offloads computationally heavy split evaluation to the GPU,
* manages tree structure and control flow on the CPU,
* and evaluates performance gains from heterogeneous execution.

The project emphasizes **parallel algorithm design**, **CPU–GPU workload partitioning**, and **empirical performance analysis**, rather than feature completeness.

---

## Scope Constraints (to Ensure Feasibility)

To keep the project tractable within 2 months:

* Numerical features only (no categorical features).
* One impurity metric (Gini or entropy).
* One pruning strategy (or none, with justification).
* One GPU-based split-finding method (histogram-based).
* Random forest size capped (e.g., ≤ 10 trees).

---

## Milestone 1: Sequential Decision Tree Baseline

**Objective:** Implement a correct and efficient sequential decision tree as a baseline for parallelization.

### Tasks

* Implement CART-style decision tree training:

  * recursive node splitting
  * Gini impurity (or entropy)
* Implement efficient split evaluation:

  * pre-sorting features or
  * histogram-based approximation on CPU
* Limit tree depth or minimum samples per leaf.
* Train and evaluate on benchmark datasets (e.g., UCI datasets).
* Compare accuracy with a reference implementation (scikit-learn).

**PDC Focus:** understanding algorithmic structure and identifying parallelizable components.

**Deliverable:**
Working sequential tree with runtime and accuracy measurements.

---

## Milestone 2: Parallel Tree Construction with GPU-Accelerated Split Finding

**Objective:** Parallelize split evaluation and integrate GPU acceleration into tree training.

### Tasks

* Implement **level-wise tree expansion**:

  * all nodes at the same depth processed in parallel.
* Offload split evaluation to the GPU:

  * compute feature histograms in parallel
  * evaluate impurity reduction per feature/bin.
* Keep tree structure management on CPU.
* Minimize CPU–GPU data transfers:

  * keep feature matrices resident on GPU
  * transfer only split statistics back to CPU.
* Support batch processing if data exceeds GPU memory.

**PDC Focus:**

* task parallelism
* heterogeneous execution
* memory transfer optimization
* irregular workload handling

**Deliverable:**
Hybrid CPU–GPU decision tree training with measurable speedup.

---

## Milestone 3: Parallel Ensemble Learning and Inference Optimization

**Objective:** Extend the system to a small ensemble and optimize inference performance.

### Tasks

* Implement a **limited random forest**:

  * train multiple trees in parallel on bootstrap samples.
* Parallelize inference:

  * evaluate multiple samples concurrently
  * optionally batch inference on GPU.
* Optimize tree representation:

  * compact node arrays
  * cache-friendly layouts.
* Benchmark:

  * training time (CPU vs CPU–GPU)
  * inference throughput
  * speedup vs number of trees.

**PDC Focus:**

* data parallelism
* ensemble-level parallelism
* throughput-oriented optimization

**Deliverable:**
Parallel ensemble system with performance evaluation.

---

## Evaluation & Analysis Requirements

Students must analyze:

* speedup from GPU acceleration
* scalability with dataset size
* CPU vs GPU utilization
* impact of level-wise parallelism on performance
* trade-offs between exact and approximate split finding

Results should include:

* runtime plots
* speedup curves
* brief discussion of bottlenecks

---

## Suggested Team Division

* **Student 1:** Sequential tree + data preprocessing
* **Student 2:** GPU split-finding kernels + optimization
* **Student 3:** CPU–GPU coordination + parallel tree construction
* **Student 4:** Ensemble methods, inference optimization, evaluation

---

## Expected Learning Outcomes

By completing this project, students will:

* understand why tree-based models are difficult to parallelize
* design heterogeneous CPU–GPU algorithms
* optimize irregular computations on GPUs
* reason about performance vs algorithmic fidelity
* gain hands-on experience with real ML systems engineering

---