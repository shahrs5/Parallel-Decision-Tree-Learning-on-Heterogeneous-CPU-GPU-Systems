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

#include "data_loader.h"
#include "metrics.h"
#include "tree/decision_tree.h"

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
// train/test split — deterministic shuffle with fixed seed
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

    X_test.clear();
    y_test.clear();
    X_train.clear();
    y_train.clear();

    for (std::size_t i = 0; i < n_test; ++i)
    {
        X_test.push_back(X[idx[i]]);
        y_test.push_back(y[idx[i]]);
    }
    for (std::size_t i = n_test; i < idx.size(); ++i)
    {
        X_train.push_back(X[idx[i]]);
        y_train.push_back(y[idx[i]]);
    }
}

// ---------------------------------------------------------------------------
// runDatasetBenchmark — loads a CSV, trains, measures, prints results
// ---------------------------------------------------------------------------

struct BenchmarkResult
{
    std::string dataset;
    int n_samples;
    int n_features;
    int n_train;
    int n_test;
    int max_depth;
    int min_leaf;
    int node_count;
    double train_ms;
    double infer_ms;
    float accuracy;
};

static BenchmarkResult runDatasetBenchmark(
    const std::string &dataset_name,
    const std::string &filepath,
    int max_depth = 5,
    int min_leaf = 1,
    float test_ratio = 0.2f)
{
    // --- Load ---
    std::vector<std::vector<float>> X;
    std::vector<int> y;
    int n = loadCSV(filepath, X, y);

    std::cout << "\nDataset: " << dataset_name << "\n";
    std::cout << "Rows loaded: " << n << "\n";

    // --- Split ---
    std::vector<std::vector<float>> X_train, X_test;
    std::vector<int> y_train, y_test;
    trainTestSplit(X, y, test_ratio, X_train, y_train, X_test, y_test);

    std::cout << "Total samples: " << n << "\n";
    std::cout << "Features: " << X[0].size() << "\n";
    std::cout << "Train samples: " << X_train.size() << "\n";
    std::cout << "Test samples: " << X_test.size() << "\n";
    std::cout << "Max depth: " << max_depth << "\n";
    std::cout << "Min samples per leaf: " << min_leaf << "\n";

    // --- Train ---
    DecisionTree tree(max_depth, min_leaf);
    auto t0 = std::chrono::high_resolution_clock::now();
    tree.train(X_train, y_train);
    auto t1 = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- Infer ---
    std::vector<int> preds;
    preds.reserve(X_test.size());
    auto t2 = std::chrono::high_resolution_clock::now();
    for (const auto &s : X_test)
        preds.push_back(tree.predict(s));
    auto t3 = std::chrono::high_resolution_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    float acc = accuracy(y_test, preds);

    std::cout << "Node count: " << tree.nodes().size() << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Training time (ms): " << train_ms << "\n";
    std::cout << "Inference time (ms): " << infer_ms << "\n";
    std::cout << "Test accuracy: " << acc << "\n";

    return BenchmarkResult{
        dataset_name,
        n,
        static_cast<int>(X[0].size()),
        static_cast<int>(X_train.size()),
        static_cast<int>(X_test.size()),
        max_depth, min_leaf,
        static_cast<int>(tree.nodes().size()),
        train_ms, infer_ms, acc};
}

// ---------------------------------------------------------------------------
// Unit tests (unchanged from Milestone 1 verification)
// ---------------------------------------------------------------------------

static bool testGiniImpurity()
{
    printSection("Step 1: Gini Impurity");
    bool all_pass = true;
    {
        std::vector<int> l = {1, 1, 1, 1};
        float g = DecisionTree::computeGini(l);
        all_pass &= check("pure node → 0.0", approxEq(g, 0.0f));
    }
    {
        std::vector<int> l = {0, 0, 1, 1};
        float g = DecisionTree::computeGini(l);
        all_pass &= check("50/50 binary → 0.5", approxEq(g, 0.5f));
    }
    {
        std::vector<int> l = {0, 0, 0, 1};
        float exp = 1.0f - (0.75f * 0.75f + 0.25f * 0.25f);
        all_pass &= check("75/25 binary → 0.375", approxEq(DecisionTree::computeGini(l), exp));
    }
    {
        std::vector<int> l = {};
        all_pass &= check("empty → 0.0", approxEq(DecisionTree::computeGini(l), 0.0f));
    }
    return all_pass;
}

static bool testCSVLoader()
{
    printSection("Step 2: CSV Loader");
    bool all_pass = true;
    const std::string tmp = "test_verify.csv";
    {
        std::ofstream f(tmp);
        f << "feature1,feature2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n";
    }
    std::vector<std::vector<float>> X;
    std::vector<int> y;
    int n = loadCSV(tmp, X, y);
    std::remove(tmp.c_str());
    all_pass &= check("4 rows loaded", n == 4);
    all_pass &= check("2 features per sample", !X.empty() && X[0].size() == 2);
    return all_pass;
}

static bool testTreeTraining()
{
    printSection("Step 3: Tree Training & Prediction");
    bool all_pass = true;
    {
        std::vector<std::vector<float>> X = {{0.1f}, {0.2f}, {0.3f}, {0.7f}, {0.8f}, {0.9f}};
        std::vector<int> y = {0, 0, 0, 1, 1, 1};
        DecisionTree tree(5, 1);
        tree.train(X, y);
        std::vector<int> p;
        for (auto &s : X)
            p.push_back(tree.predict(s));
        all_pass &= check("linear: 100% training accuracy", approxEq(accuracy(y, p), 1.0f));
    }
    {
        std::vector<std::vector<float>> X = {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
        std::vector<int> y = {0, 0, 1, 1};
        DecisionTree tree(5, 1);
        tree.train(X, y);
        std::vector<int> p;
        for (auto &s : X)
            p.push_back(tree.predict(s));
        all_pass &= check("XOR: 100% training accuracy", approxEq(accuracy(y, p), 1.0f));
    }
    {
        std::vector<std::vector<float>> X = {{1}, {2}, {3}};
        std::vector<int> y = {5, 5, 5};
        DecisionTree tree(10, 1);
        tree.train(X, y);
        all_pass &= check("pure input: single leaf node", tree.nodes().size() == 1);
    }
    return all_pass;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    std::cout << "Decision Tree — Milestone 1\n";

    // ----- Unit tests -----
    bool step1 = testGiniImpurity();
    bool step2 = testCSVLoader();
    bool step3 = testTreeTraining();
    std::cout << "\n--- Unit Test Summary ---\n";
    std::cout << "  Step 1 (Gini):  " << (step1 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Step 2 (CSV):   " << (step2 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Step 3 (Tree):  " << (step3 ? "PASS" : "FAIL") << "\n";

    // ----- Dataset benchmarks -----
    // Adjust paths to where you placed your CSVs inside data/
    printSection("Milestone 1 - Dataset Benchmarks");

    std::vector<BenchmarkResult> results;

    struct DatasetConfig
    {
        std::string name;
        std::string path;
        int max_depth;
        int min_leaf;
    };

    std::vector<DatasetConfig> datasets = {
        {"Iris", "../data/iris.csv", 5, 1},
        {"Wine", "../data/wine.csv", 5, 1},
        {"Breast Cancer", "../data/breast_cancer.csv", 7, 2},
        {"Banknote Auth", "../data/banknote.csv", 5, 1}};

    for (const auto &d : datasets)
    {
        if (!fileExists(d.path))
        {
            std::cout << "[SKIP] Missing dataset file: " << d.path << "\n";
            continue;
        }
        results.push_back(runDatasetBenchmark(d.name, d.path, d.max_depth, d.min_leaf));
    }
    if (results.empty())
    {
        std::cerr << "No benchmark datasets found in data/.\n";
        return 1;
    }
    // ----- Summary table -----
    printSection("BENCHMARK SUMMARY");
    std::cout << std::left
              << std::setw(18) << "Dataset"
              << std::setw(10) << "Samples"
              << std::setw(10) << "Features"
              << std::setw(10) << "MaxDepth"
              << std::setw(14) << "TrainTime(ms)"
              << std::setw(14) << "InferTime(ms)"
              << std::setw(10) << "Accuracy"
              << "\n";
    std::cout << std::string(86, '-') << "\n";
    for (const auto &r : results)
    {
        std::cout << std::fixed << std::setprecision(4)
                  << std::left
                  << std::setw(18) << r.dataset
                  << std::setw(10) << r.n_samples
                  << std::setw(10) << r.n_features
                  << std::setw(10) << r.max_depth
                  << std::setw(14) << r.train_ms
                  << std::setw(14) << r.infer_ms
                  << std::setw(10) << r.accuracy
                  << "\n";
    }

    return (step1 && step2 && step3) ? 0 : 1;
}