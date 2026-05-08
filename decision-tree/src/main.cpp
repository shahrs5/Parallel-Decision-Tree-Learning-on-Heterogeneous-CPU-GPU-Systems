#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include <map>

#include "data_loader.h"
#include "metrics.h"
#include "tree/decision_tree.h"
#include "tree/random_forest.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include "gpu/split_kernel.cuh"
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void printSection(const char *title)
{
    std::cout << "\n========================================\n";
    std::cout << title << "\n";
    std::cout << "========================================\n";
}

static bool fileExists(const std::string &path)
{
    std::ifstream f(path);
    return f.good();
}

static bool approxEq(float a, float b, float tol = 1e-5f)
{
    return std::fabs(a - b) < tol;
}

static bool check(const char *description, bool condition)
{
    std::cout << (condition ? "  [PASS] " : "  [FAIL] ") << description << "\n";
    return condition;
}

// ---------------------------------------------------------------------------
// Train/test split
// ---------------------------------------------------------------------------
static void trainTestSplit(
    const std::vector<std::vector<float>> &X,
    const std::vector<int> &y,
    float test_ratio,
    std::vector<std::vector<float>> &X_train,
    std::vector<int> &y_train,
    std::vector<std::vector<float>> &X_test,
    std::vector<int> &y_test,
    unsigned seed = 42)
{
    std::vector<int> idx(X.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::size_t n_test = static_cast<std::size_t>(X.size() * test_ratio);
    X_test.clear(); y_test.clear(); X_train.clear(); y_train.clear();

    for (std::size_t i = 0; i < n_test; ++i)
    { X_test.push_back(X[idx[i]]); y_test.push_back(y[idx[i]]); }
    for (std::size_t i = n_test; i < idx.size(); ++i)
    { X_train.push_back(X[idx[i]]); y_train.push_back(y[idx[i]]); }
}

// ---------------------------------------------------------------------------
// Unit tests (Milestone 1 — kept to verify nothing broke)
// ---------------------------------------------------------------------------
static bool testGiniImpurity()
{
    printSection("Step 1: Gini Impurity");
    bool ok = true;
    ok &= check("pure node -> 0.0",    approxEq(DecisionTree::computeGini({1,1,1,1}), 0.0f));
    ok &= check("50/50 binary -> 0.5", approxEq(DecisionTree::computeGini({0,0,1,1}), 0.5f));
    float e = 1.0f - (0.75f*0.75f + 0.25f*0.25f);
    ok &= check("75/25 binary -> 0.375", approxEq(DecisionTree::computeGini({0,0,0,1}), e));
    ok &= check("empty -> 0.0",        approxEq(DecisionTree::computeGini({}), 0.0f));
    return ok;
}

static bool testCSVLoader()
{
    printSection("Step 2: CSV Loader");
    bool ok = true;
    const std::string tmp = "test_verify.csv";
    { std::ofstream f(tmp); f << "feature1,feature2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n"; }
    std::vector<std::vector<float>> X; std::vector<int> y;
    int n = loadCSV(tmp, X, y);
    std::remove(tmp.c_str());
    ok &= check("4 rows loaded",         n == 4);
    ok &= check("2 features per sample", !X.empty() && X[0].size() == 2);
    return ok;
}

static bool testTreeTraining()
{
    printSection("Step 3: Tree Training & Prediction");
    bool ok = true;
    {
        std::vector<std::vector<float>> X = {{0.1f},{0.2f},{0.3f},{0.7f},{0.8f},{0.9f}};
        std::vector<int> y = {0,0,0,1,1,1};
        DecisionTree tree(5,1); tree.train(X,y);
        std::vector<int> p; for (auto &s:X) p.push_back(tree.predict(s));
        ok &= check("linear: 100% train accuracy", approxEq(accuracy(y,p), 1.0f));
    }
    {
        std::vector<std::vector<float>> X = {{0,0},{1,1},{0,1},{1,0}};
        std::vector<int> y = {0,0,1,1};
        DecisionTree tree(5,1); tree.train(X,y);
        std::vector<int> p; for (auto &s:X) p.push_back(tree.predict(s));
        ok &= check("XOR: 100% train accuracy", approxEq(accuracy(y,p), 1.0f));
    }
    {
        std::vector<std::vector<float>> X = {{1},{2},{3}};
        std::vector<int> y = {5,5,5};
        DecisionTree tree(10,1); tree.train(X,y);
        ok &= check("pure input: single leaf", tree.nodes().size() == 1);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Milestone 2 benchmark: sequential vs OpenMP parallel single tree
// ---------------------------------------------------------------------------
struct BenchmarkResult {
    std::string dataset;
    int    n_samples, n_features, n_train, n_test, max_depth, node_count;
    double seq_train_ms, par_train_ms, infer_ms;
    float  accuracy;
    double speedup;
};

static BenchmarkResult runDatasetBenchmark(
    const std::string &name, const std::string &path,
    int max_depth = 5, int min_leaf = 1, float test_ratio = 0.2f)
{
    std::vector<std::vector<float>> X; std::vector<int> y;
    int n = loadCSV(path, X, y);

    std::cout << "\nDataset: " << name << "\n";
    std::cout << "  Samples: " << n << "  Features: " << X[0].size() << "\n";

    std::vector<std::vector<float>> X_train, X_test;
    std::vector<int> y_train, y_test;
    trainTestSplit(X, y, test_ratio, X_train, y_train, X_test, y_test);

    double seq_ms = 0, par_ms = 0;

#ifdef USE_OPENMP
    // Sequential: force 1 thread.
    omp_set_num_threads(1);
    { DecisionTree t(max_depth, min_leaf);
      auto t0 = std::chrono::high_resolution_clock::now();
      t.train(X_train, y_train);
      seq_ms = std::chrono::duration<double,std::milli>(
               std::chrono::high_resolution_clock::now()-t0).count(); }

    // Parallel: all threads.
    omp_set_num_threads(omp_get_max_threads());
    { DecisionTree t(max_depth, min_leaf);
      auto t0 = std::chrono::high_resolution_clock::now();
      t.train(X_train, y_train);
      par_ms = std::chrono::duration<double,std::milli>(
               std::chrono::high_resolution_clock::now()-t0).count(); }

    omp_set_num_threads(omp_get_max_threads());
#else
    { DecisionTree t(max_depth, min_leaf);
      auto t0 = std::chrono::high_resolution_clock::now();
      t.train(X_train, y_train);
      seq_ms = par_ms = std::chrono::duration<double,std::milli>(
               std::chrono::high_resolution_clock::now()-t0).count(); }
#endif

    // Inference timing.
    DecisionTree final_tree(max_depth, min_leaf);
    final_tree.train(X_train, y_train);

    std::vector<int> preds;
    preds.reserve(X_test.size());
    auto ti0 = std::chrono::high_resolution_clock::now();
    for (const auto &s : X_test) preds.push_back(final_tree.predict(s));
    double infer_ms = std::chrono::duration<double,std::milli>(
                      std::chrono::high_resolution_clock::now()-ti0).count();

    float acc = accuracy(y_test, preds);
    double speedup = (par_ms > 0) ? seq_ms / par_ms : 1.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Sequential train (ms):  " << seq_ms  << "\n";
    std::cout << "  Parallel train (ms):    " << par_ms  << "\n";
    std::cout << "  Speedup:                " << speedup << "x\n";
    std::cout << "  Inference time (ms):    " << infer_ms << "\n";
    std::cout << "  Accuracy:               " << acc << "\n";
    std::cout << "  Node count:             " << final_tree.nodes().size() << "\n";

    return { name, n, (int)X[0].size(), (int)X_train.size(), (int)X_test.size(),
             max_depth, (int)final_tree.nodes().size(),
             seq_ms, par_ms, infer_ms, acc, speedup };
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
int runAllTests()
{
    std::cout << "Decision Tree -- Milestone 2: Parallel Level-Wise + GPU-Ready Pipeline\n";

#ifdef USE_OPENMP
    std::cout << "OpenMP enabled -- max threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP disabled -- sequential build\n";
#endif

#ifdef USE_CUDA
    std::cout << "CUDA enabled -- GPU histogram split finding active\n";
#else
    std::cout << "CUDA disabled -- CPU exact split path\n";
#endif

    // ---- Unit tests ----
    bool s1 = testGiniImpurity();
    bool s2 = testCSVLoader();
    bool s3 = testTreeTraining();
    std::cout << "\n--- Unit Test Summary ---\n";
    std::cout << "  Step 1 (Gini):  " << (s1?"PASS":"FAIL") << "\n";
    std::cout << "  Step 2 (CSV):   " << (s2?"PASS":"FAIL") << "\n";
    std::cout << "  Step 3 (Tree):  " << (s3?"PASS":"FAIL") << "\n";

    // -----------------------------------------------------------------------
    // Milestone 3 — RandomForest smoke test.
    // Trains three configurations on Breast Cancer to verify both M3 features
    // (steps 1+3): bootstrap sampling and per-split feature subsampling.
    //   1. Single tree            — baseline.
    //   2. Forest, all features   — bagging only (no subsampling).
    //   3. Forest, sqrt(F) feats  — full Random Forest.
    // Full benchmarks come later (teammate B's ticket).
    // -----------------------------------------------------------------------
    printSection("Milestone 3 -- RandomForest Smoke Test (serial)");
    {
        const std::string path = "../data/breast_cancer.csv";
        if (!fileExists(path)) {
            std::cout << "  [SKIP] " << path << " not found\n";
        } else {
            std::vector<std::vector<float>> X; std::vector<int> y;
            loadCSV(path, X, y);

            std::vector<std::vector<float>> X_tr, X_te;
            std::vector<int> y_tr, y_te;
            trainTestSplit(X, y, 0.2f, X_tr, y_tr, X_te, y_te);

            const int n_trees = 10, max_depth = 7, min_leaf = 2;

            // --- Single tree baseline ---
            DecisionTree single(max_depth, min_leaf);
            auto t0 = std::chrono::high_resolution_clock::now();
            single.train(X_tr, y_tr);
            double single_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            std::vector<int> p_single;
            for (auto &s : X_te) p_single.push_back(single.predict(s));
            float acc_single = accuracy(y_te, p_single);

            // --- Forest, all features (bagging only) ---
            RandomForest bag(n_trees, max_depth, min_leaf,
                             /*feature_subsample=*/0, /*seed=*/42);
            t0 = std::chrono::high_resolution_clock::now();
            bag.train(X_tr, y_tr);
            double bag_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            float acc_bag = accuracy(y_te, bag.predictBatch(X_te));

            // --- Forest, sqrt(F) features, SERIAL training (1 OMP thread) ---
#ifdef USE_OPENMP
            omp_set_num_threads(1);
#endif
            RandomForest rf_seq(n_trees, max_depth, min_leaf,
                                /*feature_subsample=*/-1, /*use_gpu=*/false, /*seed=*/42);
            t0 = std::chrono::high_resolution_clock::now();
            rf_seq.train(X_tr, y_tr);
            double rf_seq_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            float acc_rf_seq = accuracy(y_te, rf_seq.predictBatch(X_te));

            // --- Forest, sqrt(F) features, PARALLEL training (all threads) ---
#ifdef USE_OPENMP
            omp_set_num_threads(omp_get_max_threads());
#endif
            RandomForest rf_par(n_trees, max_depth, min_leaf,
                                /*feature_subsample=*/-1, /*use_gpu=*/false, /*seed=*/42);
            t0 = std::chrono::high_resolution_clock::now();
            rf_par.train(X_tr, y_tr);
            double rf_par_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            float acc_rf_par = accuracy(y_te, rf_par.predictBatch(X_te));

#ifdef USE_CUDA
            // --- Forest, sqrt(F) features, GPU training (sequential) ---
            RandomForest rf_gpu(n_trees, max_depth, min_leaf,
                                /*feature_subsample=*/-1, /*use_gpu=*/true, /*seed=*/42);
            t0 = std::chrono::high_resolution_clock::now();
            rf_gpu.train(X_tr, y_tr);
            double rf_gpu_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            float acc_rf_gpu = accuracy(y_te, rf_gpu.predictBatch(X_te));
#endif

            // --- Inference throughput: serial vs parallel batch ---
            // Replicate test set to make timing meaningful for small datasets.
            std::vector<std::vector<float>> X_big;
            X_big.reserve(X_te.size() * 200);
            for (int rep = 0; rep < 200; ++rep)
                for (auto &s : X_te) X_big.push_back(s);

#ifdef USE_OPENMP
            omp_set_num_threads(1);
#endif
            t0 = std::chrono::high_resolution_clock::now();
            auto p_seq = rf_par.predictBatch(X_big);
            double infer_seq_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();

#ifdef USE_OPENMP
            omp_set_num_threads(omp_get_max_threads());
#endif
            t0 = std::chrono::high_resolution_clock::now();
            auto p_par = rf_par.predictBatch(X_big);
            double infer_par_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();

            std::cout << std::fixed << std::setprecision(4);
            std::cout << "  Single tree              : " << single_ms  << " ms,  acc=" << acc_single << "\n";
            std::cout << "  Forest (bagging,    10)  : " << bag_ms     << " ms,  acc=" << acc_bag    << "\n";
            std::cout << "  Forest (sqrt(F),seq,10)  : " << rf_seq_ms  << " ms,  acc=" << acc_rf_seq << "\n";
            std::cout << "  Forest (sqrt(F),par,10)  : " << rf_par_ms  << " ms,  acc=" << acc_rf_par
                      << "  speedup=" << (rf_seq_ms / rf_par_ms) << "x\n";
#ifdef USE_CUDA
            std::cout << "  Forest (sqrt(F),gpu,10)  : " << rf_gpu_ms  << " ms,  acc=" << acc_rf_gpu
                      << "  speedup=" << (rf_seq_ms / rf_gpu_ms) << "x\n";
#endif
            std::cout << "  Batch inference (" << X_big.size() << " samples):\n";
            std::cout << "    serial   : " << infer_seq_ms << " ms\n";
            std::cout << "    parallel : " << infer_par_ms << " ms"
                      << "  speedup=" << (infer_seq_ms / infer_par_ms) << "x\n";
            std::cout << "  Predictions match (par vs seq): "
                      << (p_seq == p_par ? "YES" : "NO") << "\n";

            check("RF parallel accuracy >= 0.85 on Breast Cancer", acc_rf_par >= 0.85f);
            check("Bagging accuracy >= 0.85",                       acc_bag    >= 0.85f);
#ifdef USE_CUDA
            check("RF GPU accuracy >= 0.85",                       acc_rf_gpu >= 0.85f);
#endif
            check("Parallel and serial RF train -> same accuracy",  std::fabs(acc_rf_par - acc_rf_seq) < 1e-5f);
            check("Parallel batch inference matches serial",        p_seq == p_par);
        }
    }

#ifdef USE_CUDA
    // ---- Batch mode test ----
    // Trains on Breast Cancer in normal GPU mode, then forces batch mode
    // (simulates dataset too large for VRAM) and trains again.
    // Predictions must be identical — same histogram kernel, different data path.
    printSection("GPU Batch Processing Test");
    {
        const std::string bpath = "../data/breast_cancer.csv";
        bool batch_pass = false;
        if (!fileExists(bpath)) {
            std::cout << "  [SKIP] " << bpath << " not found\n";
        } else {
            std::vector<std::vector<float>> X;
            std::vector<int> y;
            loadCSV(bpath, X, y);
            int n = static_cast<int>(X.size());

            // Normal mode (full upload)
            DecisionTree t_normal(7, 2);
            t_normal.train(X, y);
            std::vector<int> preds_normal(n);
            for (int i = 0; i < n; ++i) preds_normal[i] = t_normal.predict(X[i]);

            // Batch mode (force d_X = nullptr path)
            setForceBatchMode(true);
            DecisionTree t_batch(7, 2);
            t_batch.train(X, y);
            std::vector<int> preds_batch(n);
            for (int i = 0; i < n; ++i) preds_batch[i] = t_batch.predict(X[i]);
            setForceBatchMode(false);

            int mismatches = 0;
            for (int i = 0; i < n; ++i)
                if (preds_normal[i] != preds_batch[i]) ++mismatches;

            float acc_normal = 0, acc_batch = 0;
            for (int i = 0; i < n; ++i) {
                if (preds_normal[i] == y[i]) acc_normal++;
                if (preds_batch[i]  == y[i]) acc_batch++;
            }
            acc_normal /= n; acc_batch /= n;

            batch_pass = (mismatches == 0);
            std::cout << "  Normal mode accuracy : " << std::fixed << std::setprecision(4) << acc_normal << "\n";
            std::cout << "  Batch mode accuracy  : " << acc_batch << "\n";
            std::cout << "  Prediction mismatches: " << mismatches << "\n";
            check("Batch mode produces identical predictions to normal mode", batch_pass);
        }
    }
#endif

    // ---- M3: Speedup vs Number of Trees ----
    // Shows how OpenMP tree-level parallelism scales with ensemble size.
    // Measures: serial (1 thread) vs parallel (all threads) for n_trees in
    // {1,5,10,20,50} on Breast Cancer; also runs GPU batch inference timing.
    printSection("Milestone 3 -- Ensemble Speedup vs Number of Trees");
    {
        const std::string bc_path = "../data/breast_cancer.csv";
        if (!fileExists(bc_path)) {
            std::cout << "  [SKIP] " << bc_path << " not found\n";
        } else {
            std::vector<std::vector<float>> X, X_tr, X_te;
            std::vector<int> y, y_tr, y_te;
            loadCSV(bc_path, X, y);
            trainTestSplit(X, y, 0.2f, X_tr, y_tr, X_te, y_te);

            std::cout << std::left
                      << std::setw(10) << "N_Trees"
                      << std::setw(14) << "Seq(ms)"
                      << std::setw(14) << "Par(ms)"
                      << std::setw(10) << "Speedup"
                      << std::setw(10) << "Accuracy"
                      << "\n" << std::string(58, '-') << "\n";

            for (int nt : {1, 5, 10, 20, 50}) {
#ifdef USE_OPENMP
                omp_set_num_threads(1);
#endif
                RandomForest rf_seq(nt, 7, 2, -1, false, 42);
                auto t0 = std::chrono::high_resolution_clock::now();
                rf_seq.train(X_tr, y_tr);
                double seq_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();

#ifdef USE_OPENMP
                omp_set_num_threads(omp_get_max_threads());
#endif
                RandomForest rf_par(nt, 7, 2, -1, false, 42);
                t0 = std::chrono::high_resolution_clock::now();
                rf_par.train(X_tr, y_tr);
                double par_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();

                float acc = accuracy(y_te, rf_par.predictBatch(X_te));
                double sp = (par_ms > 0) ? seq_ms / par_ms : 1.0;

                std::cout << std::fixed << std::setprecision(2) << std::left
                          << std::setw(10) << nt
                          << std::setw(14) << seq_ms
                          << std::setw(14) << par_ms
                          << std::setw(10) << sp
                          << std::fixed << std::setprecision(4)
                          << std::setw(10) << acc
                          << "\n";
            }
#ifdef USE_OPENMP
            omp_set_num_threads(omp_get_max_threads());
#endif

#ifdef USE_CUDA
            // GPU batch inference comparison (10-tree forest, varying batch size).
            std::cout << "\n  GPU vs CPU Inference Throughput (10-tree forest):\n";
            std::cout << std::left
                      << std::setw(12) << "  Batch"
                      << std::setw(14) << "CPU-par(ms)"
                      << std::setw(14) << "GPU(ms)"
                      << std::setw(10) << "Speedup"
                      << "\n  " << std::string(48, '-') << "\n";

            RandomForest rf_infer(10, 7, 2, -1, false, 42);
            rf_infer.train(X_tr, y_tr);

            for (int batch : {500, 1000, 5000, 10000, 50000}) {
                std::vector<std::vector<float>> X_big;
                X_big.reserve(batch);
                while ((int)X_big.size() < batch)
                    for (auto &s : X_te) { if ((int)X_big.size() >= batch) break; X_big.push_back(s); }
                X_big.resize(batch);

#ifdef USE_OPENMP
                omp_set_num_threads(omp_get_max_threads());
#endif
                auto t0 = std::chrono::high_resolution_clock::now();
                rf_infer.predictBatch(X_big);
                double cpu_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();

                t0 = std::chrono::high_resolution_clock::now();
                rf_infer.predictBatchGPU(X_big);
                double gpu_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();

                double sp = (gpu_ms > 0) ? cpu_ms / gpu_ms : 0.0;
                std::cout << std::fixed << std::setprecision(2) << std::left
                          << "  " << std::setw(10) << batch
                          << std::setw(14) << cpu_ms
                          << std::setw(14) << gpu_ms
                          << std::setw(10) << sp
                          << "\n";
            }
#endif // USE_CUDA
        }
    }

    // ---- Split-Finding Kernel Benchmark ----
    // This benchmarks findBestSplitForNode in isolation — the exact operation
    // the GPU histogram kernel replaces.  Measures speedup vs node size,
    // which is the relevant metric for CPU-GPU comparison.
    printSection("Milestone 2 -- Split-Finding Kernel: Speedup vs Node Size");

    {
        std::mt19937 rng(777);
        std::uniform_real_distribution<float> xd(-1.0f, 1.0f);

        std::cout << std::left
                  << std::setw(12) << "NodeSize"
                  << std::setw(12) << "Features"
                  << std::setw(14) << "Seq(ms)"
                  << std::setw(14) << "Par(ms)"
                  << std::setw(10) << "Speedup"
                  << "\n" << std::string(62, '-') << "\n";

        for (int n_node : {200, 500, 1000, 2000, 5000, 10000}) {
            for (int n_feat : {4, 13, 30}) {
                // Build a flat dataset of n_node samples, n_feat features.
                std::vector<std::vector<float>> X(n_node, std::vector<float>(n_feat));
                std::vector<int> y(n_node);
                for (int i = 0; i < n_node; ++i) {
                    for (int j = 0; j < n_feat; ++j) X[i][j] = xd(rng);
                    y[i] = rng() % 2;
                }
                std::vector<int> idx(n_node);
                std::iota(idx.begin(), idx.end(), 0);

                // Sequential (1 thread).
                double seq_ms = 0;
#ifdef USE_OPENMP
                omp_set_num_threads(1);
#endif
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int r = 0; r < 5; ++r) {
                        // Rebuild tree to time just split finding repeatedly.
                        DecisionTree tmp(1, 1);
                        tmp.train(X, y);
                    }
                    seq_ms = std::chrono::duration<double,std::milli>(
                             std::chrono::high_resolution_clock::now()-t0).count() / 5.0;
                }

                // Parallel (all threads).
                double par_ms = 0;
#ifdef USE_OPENMP
                omp_set_num_threads(omp_get_max_threads());
#endif
                {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    for (int r = 0; r < 5; ++r) {
                        DecisionTree tmp(1, 1);
                        tmp.train(X, y);
                    }
                    par_ms = std::chrono::duration<double,std::milli>(
                             std::chrono::high_resolution_clock::now()-t0).count() / 5.0;
                }
#ifdef USE_OPENMP
                omp_set_num_threads(omp_get_max_threads());
#endif
                double sp = (par_ms > 0) ? seq_ms / par_ms : 1.0;

                std::cout << std::fixed << std::setprecision(3) << std::left
                          << std::setw(12) << n_node
                          << std::setw(12) << n_feat
                          << std::setw(14) << seq_ms
                          << std::setw(14) << par_ms
                          << std::setw(10) << sp
                          << "\n";
            }
        }
    }

    // ---- Dataset benchmarks ----
    printSection("Milestone 2 -- Single Tree: Sequential vs Parallel (UCI Datasets)");
    std::vector<BenchmarkResult> results;

    struct DatasetConfig { std::string name, path; int max_depth, min_leaf; };
    std::vector<DatasetConfig> datasets = {
        {"Iris",          "../data/iris.csv",          5, 1},
        {"Wine",          "../data/wine.csv",          5, 1},
        {"Breast Cancer", "../data/breast_cancer.csv", 7, 2},
        {"Banknote Auth", "../data/banknote.csv",      5, 1},
        {"Synthetic",     "../data/synthetic.csv",     8, 1},
         {"Covertype",    "data/covertype.csv",       10, 5}
    };

    for (const auto &d : datasets)
    {
        if (!fileExists(d.path))
        {
            std::cout << "[SKIP] Missing dataset file: " << d.path << "\n";
            continue;
        }
        results.push_back(runDatasetBenchmark(d.name, d.path, d.max_depth, d.min_leaf));
    }

    // ---- Summary table ----
    printSection("BENCHMARK SUMMARY -- Milestone 2");
    std::cout << std::left
              << std::setw(18) << "Dataset"
              << std::setw(10) << "Samples"
              << std::setw(10) << "Features"
              << std::setw(14) << "Seq(ms)"
              << std::setw(14) << "Par(ms)"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "Infer(ms)"
              << std::setw(10) << "Accuracy"
              << "\n";
    std::cout << std::string(100, '-') << "\n";
    for (const auto &r : results) {
        std::cout << std::fixed << std::setprecision(4) << std::left
                  << std::setw(18) << r.dataset
                  << std::setw(10) << r.n_samples
                  << std::setw(10) << r.n_features
                  << std::setw(14) << r.seq_train_ms
                  << std::setw(14) << r.par_train_ms
                  << std::setw(10) << r.speedup
                  << std::setw(14) << r.infer_ms
                  << std::setw(10) << r.accuracy
                  << "\n";
    }

#ifdef USE_CUDA
    // ---- Scalability benchmark ----
    // Generates synthetic data at increasing sizes in C++ (no CSV needed).
    // Measures sequential and parallel training time vs n_samples.
    printSection("GPU Scalability -- Training Time vs Dataset Size");
    {
        std::vector<int> sizes = {500, 1000, 2000, 5000, 10000, 25000};
        const int n_feat = 20, depth = 7, leaf = 2, reps = 3;

        std::cout << std::left
                  << std::setw(10) << "Samples"
                  << std::setw(14) << "Seq(ms)"
                  << std::setw(14) << "Par(ms)"
                  << std::setw(10) << "Speedup"
                  << std::setw(14) << "KernelUtil%"
                  << std::setw(10) << "Nodes"
                  << "\n" << std::string(72, '-') << "\n";

        for (int n : sizes) {
            // Generate synthetic 2-class data
            std::mt19937 rng(42);
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<std::vector<float>> X(n, std::vector<float>(n_feat));
            std::vector<int> y(n);
            for (int i = 0; i < n; ++i) {
                float sum = 0;
                for (int f = 0; f < n_feat; ++f) { X[i][f] = nd(rng); sum += X[i][f]; }
                y[i] = (sum > 0) ? 1 : 0;
            }

            // Sequential
            double seq_ms = 0;
            int n_nodes = 0;
            for (int r = 0; r < reps; ++r) {
                resetGPUCallStats();
                DecisionTree t(depth, leaf);
                auto t0 = std::chrono::high_resolution_clock::now();
                t.train(X, y);
                auto t1 = std::chrono::high_resolution_clock::now();
                seq_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                n_nodes = static_cast<int>(t.nodes().size());
            }
            seq_ms /= reps;

            // Parallel — collect GPU stats on last rep
            double par_ms = 0;
            GPUCallStats stats{};
            for (int r = 0; r < reps; ++r) {
                resetGPUCallStats();
                DecisionTree t(depth, leaf);
                auto t0 = std::chrono::high_resolution_clock::now();
                t.train(X, y);
                auto t1 = std::chrono::high_resolution_clock::now();
                par_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                getGPUCallStats(&stats);
            }
            par_ms /= reps;

            float util = (stats.total_ms > 0)
                         ? (stats.kernel_ms / stats.total_ms) * 100.0f : 0.0f;

            std::cout << std::fixed << std::setprecision(2) << std::left
                      << std::setw(10) << n
                      << std::setw(14) << seq_ms
                      << std::setw(14) << par_ms
                      << std::setw(10) << seq_ms / par_ms
                      << std::setw(14) << util
                      << std::setw(10) << n_nodes
                      << "\n";
        }
    }

    // ---- GPU utilization detail (on synthetic 10k) ----
    printSection("GPU Utilization Breakdown (n=10000, f=20)");
    {
        const int n = 10000, n_feat = 20, depth = 7, leaf = 2;
        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<std::vector<float>> X(n, std::vector<float>(n_feat));
        std::vector<int> y(n);
        for (int i = 0; i < n; ++i) {
            float sum = 0;
            for (int f = 0; f < n_feat; ++f) { X[i][f] = nd(rng); sum += X[i][f]; }
            y[i] = (sum > 0) ? 1 : 0;
        }

        resetGPUCallStats();
        DecisionTree t(depth, leaf);
        auto t0 = std::chrono::high_resolution_clock::now();
        t.train(X, y);
        auto t1 = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        GPUCallStats s{};
        getGPUCallStats(&s);
        float overhead_ms = s.total_ms - s.kernel_ms;
        float util = (s.total_ms > 0) ? (s.kernel_ms / s.total_ms) * 100.0f : 0.0f;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Total training wall time : " << wall_ms        << " ms\n";
        std::cout << "  Total GPU call time      : " << s.total_ms     << " ms  (across " << s.n_calls << " nodes)\n";
        std::cout << "  Kernel compute time      : " << s.kernel_ms    << " ms\n";
        std::cout << "  Transfer + sync overhead : " << overhead_ms    << " ms\n";
        std::cout << "  GPU compute utilization  : " << util           << " %\n";
        std::cout << "  CPU overhead (non-GPU)   : " << wall_ms - s.total_ms << " ms\n";
        std::cout << "  Avg kernel time / node   : " << s.kernel_ms / s.n_calls << " ms\n";
        std::cout << "  Avg overhead / node      : " << overhead_ms   / s.n_calls << " ms\n";
    }
#endif

#ifdef USE_CUDA
    // ---- CPU vs GPU Comparison on Large Datasets ----
    // Runs each dataset THREE ways in the same binary:
    //   (a) CPU sequential (setUseGPU(false), 1 thread)
    //   (b) CPU parallel   (setUseGPU(false), all threads via OpenMP)
    //   (c) GPU histogram  (setUseGPU(true),  all threads)
    // This isolates the GPU kernel benefit from OpenMP thread benefit.
    printSection("CPU vs GPU Comparison -- Large Dataset Scaling");

    struct ScaleConfig { std::string name, path; int depth, leaf; };
    std::vector<ScaleConfig> scale_datasets = {
        {"Iris",           "../data/iris.csv",           5,  1},
        {"Wine",           "../data/wine.csv",           5,  1},
        {"Breast Cancer",  "../data/breast_cancer.csv",  7,  2},
        {"Banknote Auth",  "../data/banknote.csv",        5,  1},
        {"Synthetic (6k)", "../data/synthetic.csv",       8,  1},
        {"Synthetic 200k", "../data/synthetic_200k.csv",  8,  5},
    };

    std::cout << std::left
              << std::setw(24) << "Dataset"
              << std::setw(12) << "Samples"
              << std::setw(14) << "CPU_Seq(ms)"
              << std::setw(14) << "CPU_Par(ms)"
              << std::setw(14) << "GPU(ms)"
              << std::setw(12) << "CPU_Speedup"
              << std::setw(12) << "GPU_Speedup"
              << std::setw(10) << "Accuracy"
              << "\n" << std::string(112, '-') << "\n";

    for (const auto &d : scale_datasets) {
        if (!fileExists(d.path)) {
            std::cout << "  [SKIP] " << d.path << "\n";
            continue;
        }

        std::vector<std::vector<float>> X; std::vector<int> y;
        int n = loadCSV(d.path, X, y);
        if (n <= 0) { std::cout << "  [FAIL] " << d.path << "\n"; continue; }

        std::vector<std::vector<float>> X_tr, X_te;
        std::vector<int> y_tr, y_te;
        trainTestSplit(X, y, 0.2f, X_tr, y_tr, X_te, y_te);

        auto timeTrain = [&](bool gpu, int threads) -> double {
#ifdef USE_OPENMP
            omp_set_num_threads(threads);
#endif
            DecisionTree t(d.depth, d.leaf);
            t.setUseGPU(gpu);
            auto t0 = std::chrono::high_resolution_clock::now();
            t.train(X_tr, y_tr);
            return std::chrono::duration<double, std::milli>(
                   std::chrono::high_resolution_clock::now() - t0).count();
        };

        int max_thr = 1;
#ifdef USE_OPENMP
        max_thr = omp_get_max_threads();
#endif

        double cpu_seq = timeTrain(false, 1);
        double cpu_par = timeTrain(false, max_thr);
        double gpu_ms  = timeTrain(true,  max_thr);

        // Accuracy on GPU path
        DecisionTree t_acc(d.depth, d.leaf);
        t_acc.setUseGPU(true);
#ifdef USE_OPENMP
        omp_set_num_threads(max_thr);
#endif
        t_acc.train(X_tr, y_tr);
        std::vector<int> preds;
        for (const auto &s : X_te) preds.push_back(t_acc.predict(s));
        float acc = accuracy(y_te, preds);

        double cpu_sp = (cpu_par > 0) ? cpu_seq / cpu_par : 1.0;
        double gpu_sp = (gpu_ms  > 0) ? cpu_seq / gpu_ms  : 1.0;

        std::cout << std::fixed << std::setprecision(2) << std::left
                  << std::setw(24) << d.name
                  << std::setw(12) << n
                  << std::setw(14) << cpu_seq
                  << std::setw(14) << cpu_par
                  << std::setw(14) << gpu_ms
                  << std::setw(12) << cpu_sp
                  << std::setw(12) << gpu_sp
                  << std::fixed << std::setprecision(4)
                  << std::setw(10) << acc
                  << "\n";
    }

#ifdef USE_OPENMP
    omp_set_num_threads(omp_get_max_threads());
#endif
#endif // USE_CUDA

    return (s1 && s2 && s3) ? 0 : 1;
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string cmd = argv[1];
        if (cmd == "--benchmark-rf-cpu" && argc >= 6) {
            // --benchmark-rf-cpu dataset n_trees max_depth min_leaf
            std::string dataset = argv[2];
            int n_trees = std::stoi(argv[3]);
            int max_depth = std::stoi(argv[4]);
            int min_leaf = std::stoi(argv[5]);
            std::string path = "../data/" + dataset + ".csv";
            std::vector<std::vector<float>> X, X_tr, X_te;
            std::vector<int> y, y_tr, y_te;
            loadCSV(path, X, y);
            trainTestSplit(X, y, 0.2f, X_tr, y_tr, X_te, y_te, 42);
            RandomForest rf(n_trees, max_depth, min_leaf, -1, false, 42);
            auto t0 = std::chrono::high_resolution_clock::now();
            rf.train(X_tr, y_tr);
            double time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            float acc = accuracy(y_te, rf.predictBatch(X_te));
            std::cout << time_ms << " " << acc << std::endl;
            return 0;
        } else if (cmd == "--benchmark-rf-gpu" && argc >= 6) {
            // --benchmark-rf-gpu dataset n_trees max_depth min_leaf
#ifdef USE_CUDA
            warmupCUDA();  // initialise CUDA context before timing
            std::string dataset = argv[2];
            int n_trees = std::stoi(argv[3]);
            int max_depth = std::stoi(argv[4]);
            int min_leaf = std::stoi(argv[5]);
            std::string path = "../data/" + dataset + ".csv";
            std::vector<std::vector<float>> X, X_tr, X_te;
            std::vector<int> y, y_tr, y_te;
            loadCSV(path, X, y);
            trainTestSplit(X, y, 0.2f, X_tr, y_tr, X_te, y_te, 42);
            RandomForest rf(n_trees, max_depth, min_leaf, -1, true, 42);
            auto t0 = std::chrono::high_resolution_clock::now();
            rf.train(X_tr, y_tr);
            double time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            float acc = accuracy(y_te, rf.predictBatch(X_te));
            std::cout << time_ms << " " << acc << std::endl;
#else
            std::cout << "-1 -1" << std::endl;
#endif
            return 0;
        } else if (cmd == "--benchmark-infer-seq" && argc >= 4) {
            // --benchmark-infer-seq dataset n_samples
            std::string dataset = argv[2];
            int n_samples = std::stoi(argv[3]);
            std::string path = "../data/" + dataset + ".csv";
            std::vector<std::vector<float>> X, X_tr, X_te;
            std::vector<int> y, y_tr, y_te;
            loadCSV(path, X, y);
            trainTestSplit(X, y, 0.2f, X_tr, y_tr, X_te, y_te, 42);
            RandomForest rf(10, 7, 2, -1, false, 42);
            rf.train(X_tr, y_tr);
            // Replicate test set
            std::vector<std::vector<float>> X_big;
            X_big.reserve(X_te.size() * (n_samples / X_te.size() + 1));
            while (X_big.size() < (size_t)n_samples) {
                for (auto &s : X_te) {
                    if (X_big.size() >= (size_t)n_samples) break;
                    X_big.push_back(s);
                }
            }
            X_big.resize(n_samples);
#ifdef USE_OPENMP
            omp_set_num_threads(1);
#endif
            auto t0 = std::chrono::high_resolution_clock::now();
            rf.predictBatch(X_big);
            double time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << time_ms << std::endl;
            return 0;
        } else if (cmd == "--benchmark-infer-par" && argc >= 4) {
            // --benchmark-infer-par dataset n_samples
            std::string dataset = argv[2];
            int n_samples = std::stoi(argv[3]);
            std::string path = "../data/" + dataset + ".csv";
            std::vector<std::vector<float>> X, X_tr, X_te;
            std::vector<int> y, y_tr, y_te;
            loadCSV(path, X, y);
            trainTestSplit(X, y, 0.2f, X_tr, y_tr, X_te, y_te, 42);
            RandomForest rf(10, 7, 2, -1, false, 42);
            rf.train(X_tr, y_tr);
            // Replicate test set
            std::vector<std::vector<float>> X_big;
            X_big.reserve(X_te.size() * (n_samples / X_te.size() + 1));
            while (X_big.size() < (size_t)n_samples) {
                for (auto &s : X_te) {
                    if (X_big.size() >= (size_t)n_samples) break;
                    X_big.push_back(s);
                }
            }
            X_big.resize(n_samples);
#ifdef USE_OPENMP
            omp_set_num_threads(omp_get_max_threads());
#endif
            auto t0 = std::chrono::high_resolution_clock::now();
            rf.predictBatch(X_big);
            double time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << time_ms << std::endl;
            return 0;
        } else if (cmd == "--benchmark-infer-gpu" && argc >= 4) {
            // --benchmark-infer-gpu dataset n_samples
            // Trains a 10-tree CPU forest, then runs GPU batch inference.
            // Outputs: time_ms (inference only, CUDA context pre-warmed)
#ifdef USE_CUDA
            warmupCUDA();  // initialise CUDA context so it is not included in timing
            std::string dataset  = argv[2];
            int         n_samp   = std::stoi(argv[3]);
            std::string path     = "../data/" + dataset + ".csv";
            std::vector<std::vector<float>> X, X_tr, X_te;
            std::vector<int> y, y_tr, y_te;
            loadCSV(path, X, y);
            trainTestSplit(X, y, 0.2f, X_tr, y_tr, X_te, y_te, 42);
            RandomForest rf(10, 7, 2, -1, false, 42);
            rf.train(X_tr, y_tr);
            std::vector<std::vector<float>> X_big;
            X_big.reserve(X_te.size() * (n_samp / (int)X_te.size() + 1));
            while ((int)X_big.size() < n_samp) {
                for (auto &s : X_te) {
                    if ((int)X_big.size() >= n_samp) break;
                    X_big.push_back(s);
                }
            }
            X_big.resize(n_samp);
            auto t0 = std::chrono::high_resolution_clock::now();
            rf.predictBatchGPU(X_big);
            double time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << time_ms << std::endl;
#else
            std::cout << "-1" << std::endl;
#endif
            return 0;
        } else if (cmd == "--benchmark-rf-speedup" && argc >= 3) {
            // --benchmark-rf-speedup dataset
            // Measures sequential vs parallel training time for n_trees in
            // {1,5,10,20,50}. Outputs CSV rows: n_trees,seq_ms,par_ms
            std::string dataset = argv[2];
            std::string path    = "../data/" + dataset + ".csv";
            std::vector<std::vector<float>> X, X_tr, X_te;
            std::vector<int> y, y_tr, y_te;
            loadCSV(path, X, y);
            trainTestSplit(X, y, 0.2f, X_tr, y_tr, X_te, y_te, 42);
            for (int n_trees : {1, 5, 10, 20, 50}) {
#ifdef USE_OPENMP
                omp_set_num_threads(1);
#endif
                RandomForest rf_seq(n_trees, 7, 2, -1, false, 42);
                auto t0 = std::chrono::high_resolution_clock::now();
                rf_seq.train(X_tr, y_tr);
                double seq_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
#ifdef USE_OPENMP
                omp_set_num_threads(omp_get_max_threads());
#endif
                RandomForest rf_par(n_trees, 7, 2, -1, false, 42);
                t0 = std::chrono::high_resolution_clock::now();
                rf_par.train(X_tr, y_tr);
                double par_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count();
                float acc = accuracy(y_te, rf_par.predictBatch(X_te));
                std::cout << n_trees << " " << seq_ms << " " << par_ms
                          << " " << acc << "\n";
            }
#ifdef USE_OPENMP
            omp_set_num_threads(omp_get_max_threads());
#endif
            return 0;
        }
    }

    return runAllTests();
}
