#include <cassert>
#include <cmath>
#include <cstdio>       // std::remove
#include <fstream>
#include <iostream>
#include <vector>

#include "data_loader.h"
#include "metrics.h"
#include "tree/decision_tree.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void printSection(const char* title) {
    std::cout << "\n=== " << title << " ===\n";
}

// Checks whether two floats are equal within a small tolerance.
static bool approxEq(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) < tol;
}

// Prints PASS/FAIL and returns whether the check passed.
static bool check(const char* description, bool condition) {
    std::cout << (condition ? "  [PASS] " : "  [FAIL] ") << description << "\n";
    return condition;
}

// ---------------------------------------------------------------------------
// Step 1 — Gini impurity
//
// Verifies the formula against analytically known values.
// ---------------------------------------------------------------------------

static bool testGiniImpurity() {
    printSection("Step 1: Gini Impurity");
    bool all_pass = true;

    // Pure node: all the same class → Gini = 0.
    {
        std::vector<int> labels = {1, 1, 1, 1};
        float g = DecisionTree::computeGini(labels);
        std::cout << "  computeGini({1,1,1,1}) = " << g << "\n";
        all_pass &= check("pure node → 0.0", approxEq(g, 0.0f));
    }

    // 50/50 binary split → Gini = 1 - (0.5² + 0.5²) = 0.5.
    {
        std::vector<int> labels = {0, 0, 1, 1};
        float g = DecisionTree::computeGini(labels);
        std::cout << "  computeGini({0,0,1,1}) = " << g << "\n";
        all_pass &= check("50/50 binary → 0.5", approxEq(g, 0.5f));
    }

    // 75/25 binary → Gini = 1 - (0.75² + 0.25²) = 0.375.
    {
        std::vector<int> labels = {0, 0, 0, 1};
        float expected = 1.0f - (0.75f * 0.75f + 0.25f * 0.25f);
        float g        = DecisionTree::computeGini(labels);
        std::cout << "  computeGini({0,0,0,1}) = " << g
                  << "  (expected " << expected << ")\n";
        all_pass &= check("75/25 binary → 0.375", approxEq(g, expected));
    }

    // 3 equal classes → Gini = 1 - 3*(1/3)² = 2/3 ≈ 0.6667.
    {
        std::vector<int> labels  = {0, 1, 2};
        float expected = 1.0f - 3.0f * (1.0f / 9.0f);
        float g        = DecisionTree::computeGini(labels);
        std::cout << "  computeGini({0,1,2}) = " << g
                  << "  (expected " << expected << ")\n";
        all_pass &= check("3-class equal → 0.6667", approxEq(g, expected));
    }

    // Empty input → Gini = 0 by convention.
    {
        std::vector<int> labels = {};
        float g = DecisionTree::computeGini(labels);
        std::cout << "  computeGini({}) = " << g << "\n";
        all_pass &= check("empty → 0.0", approxEq(g, 0.0f));
    }

    return all_pass;
}

// ---------------------------------------------------------------------------
// Step 2 — CSV loader
//
// Writes a small CSV to a temp file, loads it, and checks every value.
// ---------------------------------------------------------------------------

static bool testCSVLoader() {
    printSection("Step 2: CSV Loader");
    bool all_pass = true;

    const std::string tmp = "test_verify.csv";

    // Create a minimal CSV with a header row and 4 data rows.
    {
        std::ofstream f(tmp);
        f << "feature1,feature2,label\n"  // header (should be skipped)
          << "1.0,2.0,0\n"
          << "3.0,4.0,1\n"
          << "5.0,6.0,0\n"
          << "7.0,8.0,1\n";
    }

    std::vector<std::vector<float>> X;
    std::vector<int>                y;
    int n = loadCSV(tmp, X, y);
    std::remove(tmp.c_str()); // clean up temp file

    std::cout << "  Rows loaded: " << n << "\n";
    all_pass &= check("4 rows loaded",          n == 4);
    all_pass &= check("2 features per sample",  !X.empty() && X[0].size() == 2);

    if (!X.empty() && X[0].size() == 2) {
        all_pass &= check("X[0] = {1.0, 2.0}",
                          approxEq(X[0][0], 1.0f) && approxEq(X[0][1], 2.0f));
        all_pass &= check("y[0] = 0",  y[0] == 0);
        all_pass &= check("X[3] = {7.0, 8.0}",
                          approxEq(X[3][0], 7.0f) && approxEq(X[3][1], 8.0f));
        all_pass &= check("y[3] = 1",  y[3] == 1);
    }

    // Edge case: CSV without a header row should also load correctly.
    {
        const std::string tmp2 = "test_no_header.csv";
        {
            std::ofstream f(tmp2);
            f << "10.0,20.0,2\n"
              << "30.0,40.0,3\n";
        }
        std::vector<std::vector<float>> X2;
        std::vector<int>                y2;
        int n2 = loadCSV(tmp2, X2, y2);
        std::remove(tmp2.c_str());
        all_pass &= check("no-header CSV: 2 rows",      n2 == 2);
        all_pass &= check("no-header CSV: y[1] = 3",    !y2.empty() && y2[1] == 3);
    }

    return all_pass;
}

// ---------------------------------------------------------------------------
// Step 3 — Decision tree training and prediction
//
// Uses a tiny linearly-separable dataset and a small XOR-like dataset.
// ---------------------------------------------------------------------------

static bool testTreeTraining() {
    printSection("Step 3: Tree Training & Prediction");
    bool all_pass = true;

    // --- 3a: Linearly separable (depth 1 should suffice) ---
    {
        //  x < 0.5 → label 0,   x ≥ 0.5 → label 1
        std::vector<std::vector<float>> X = {
            {0.1f}, {0.2f}, {0.3f},
            {0.7f}, {0.8f}, {0.9f},
        };
        std::vector<int> y = {0, 0, 0, 1, 1, 1};

        DecisionTree tree(/*max_depth=*/5, /*min_samples_leaf=*/1);
        tree.train(X, y);

        std::cout << "  Linear dataset — node count: " << tree.nodes().size() << "\n";
        all_pass &= check("linear: root is not a leaf",  !tree.nodes()[0].is_leaf);

        std::vector<int> preds;
        for (const auto& s : X) preds.push_back(tree.predict(s));
        float acc = accuracy(y, preds);
        std::cout << "  Linear dataset — training accuracy: " << acc << "\n";
        all_pass &= check("linear: 100% training accuracy", approxEq(acc, 1.0f));
    }

    // --- 3b: XOR-like (needs depth ≥ 2, tests multi-level splits) ---
    {
        //  (0,0)→0,  (1,1)→0,  (0,1)→1,  (1,0)→1
        std::vector<std::vector<float>> X = {
            {0.0f, 0.0f},
            {1.0f, 1.0f},
            {0.0f, 1.0f},
            {1.0f, 0.0f},
        };
        std::vector<int> y = {0, 0, 1, 1};

        DecisionTree tree(/*max_depth=*/5, /*min_samples_leaf=*/1);
        tree.train(X, y);

        std::cout << "  XOR dataset — node count: " << tree.nodes().size() << "\n";
        all_pass &= check("XOR: tree has more than 1 node", tree.nodes().size() > 1);

        std::vector<int> preds;
        for (const auto& s : X) preds.push_back(tree.predict(s));
        float acc = accuracy(y, preds);
        std::cout << "  XOR dataset — training accuracy: " << acc << "\n";
        all_pass &= check("XOR: 100% training accuracy", approxEq(acc, 1.0f));
    }

    // --- 3c: Pure node should produce a single-node (leaf) tree ---
    {
        std::vector<std::vector<float>> X = {{1.0f}, {2.0f}, {3.0f}};
        std::vector<int>                y = {5, 5, 5}; // all same label

        DecisionTree tree(/*max_depth=*/10, /*min_samples_leaf=*/1);
        tree.train(X, y);

        all_pass &= check("pure input: single leaf node",  tree.nodes().size() == 1);
        all_pass &= check("pure input: root is leaf",       tree.nodes()[0].is_leaf);
        all_pass &= check("pure input: predicts 5",         tree.predict({1.5f}) == 5);
    }

    // --- 3d: max_depth=0 forces root to be a leaf ---
    {
        std::vector<std::vector<float>> X = {{0.0f}, {1.0f}};
        std::vector<int>                y = {0, 1};

        DecisionTree tree(/*max_depth=*/0, /*min_samples_leaf=*/1);
        tree.train(X, y);

        all_pass &= check("max_depth=0: root is leaf",  tree.nodes()[0].is_leaf);
    }

    return all_pass;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "Decision Tree — Milestone 1 Verification\n";

    bool step1 = testGiniImpurity();
    bool step2 = testCSVLoader();
    bool step3 = testTreeTraining();

    std::cout << "\n--- Summary ---\n";
    std::cout << "  Step 1 (Gini):   " << (step1 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Step 2 (CSV):    " << (step2 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Step 3 (Tree):   " << (step3 ? "PASS" : "FAIL") << "\n";

    bool all_pass = step1 && step2 && step3;
    std::cout << "\nOverall: " << (all_pass ? "ALL PASS" : "SOME FAILURES") << "\n";
    return all_pass ? 0 : 1;
}
