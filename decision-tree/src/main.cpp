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

#ifdef USE_OPENMP
#include <omp.h>
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

static BenchmarkResult runBenchmark(
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
// main
// ---------------------------------------------------------------------------
int main()
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

    struct DatasetConfig { std::string name, path; int depth, leaf; };
    std::vector<DatasetConfig> datasets = {
        {"Iris",          "../data/iris.csv",          5, 1},
        {"Wine",          "../data/wine.csv",          5, 1},
        {"Breast Cancer", "../data/breast_cancer.csv", 7, 2},
        {"Banknote Auth", "../data/banknote.csv",      5, 1}
        {"Synthetic",     "../data/synthetic.csv",     8, 1}
    };

    std::vector<BenchmarkResult> results;
    for (const auto &d : datasets) {
        if (!fileExists(d.path)) { std::cout << "[SKIP] " << d.path << "\n"; continue; }
        results.push_back(runBenchmark(d.name, d.path, d.depth, d.leaf));
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

    return (s1 && s2 && s3) ? 0 : 1;
}
