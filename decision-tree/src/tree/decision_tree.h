#pragma once

#include <string>
#include <vector>
#include <utility>
#include "node.h"

// ---------------------------------------------------------------------------
// DecisionTree — CART-style binary decision tree classifier.
//
// Milestone 1: sequential training with Gini impurity (unchanged).
// Milestone 2 (Person 3 — CPU-GPU Integration):
//   - trainLevelWise() processes all nodes at the same depth in parallel
//     via OpenMP when compiled with -DUSE_OPENMP (or -fopenmp).
//   - When compiled with -DUSE_CUDA the inner findBestSplitForNode() call
//     is replaced by findBestSplitGPU() from src/gpu/split_kernel.cuh.
//     Feature data is uploaded once and stays GPU-resident; only the tiny
//     (feature, threshold) result is transferred back per node.
// Milestone 3: DecisionTree is reused as-is inside RandomForest.
// ---------------------------------------------------------------------------
class DecisionTree
{
public:
    explicit DecisionTree(int max_depth = 10,
                          int min_samples_leaf = 1);

    ~DecisionTree();

    void train(const std::vector<std::vector<float>> &X,
               const std::vector<int> &y);

    int predict(const std::vector<float> &sample) const;

    const std::vector<Node> &nodes() const { return nodes_; }

    static float computeGini(const std::vector<int> &labels);
    static int   majorityLabel(const std::vector<int> &labels);

private:
    struct PendingNode {
        int node_idx;
        std::vector<int> sample_indices;
        int depth;
    };

    int max_depth_;
    int min_samples_leaf_;
    std::vector<Node> nodes_;

    // Flattened row-major feature matrix kept for GPU upload (Person 3).
    // Populated in train(); reused across nodes so we upload once.
    std::vector<float> X_flat_;   // [n_samples * n_features]
    int n_samples_  = 0;
    int n_features_ = 0;

#ifdef USE_CUDA
    float* d_X_ = nullptr;   // GPU-resident feature matrix
    int*   d_y_ = nullptr;   // GPU-resident label array
#endif

    int  createEmptyNode();

    bool findBestSplitForNode(const std::vector<std::vector<float>> &X,
                              const std::vector<int> &sample_indices,
                              const std::vector<int> &y,
                              int   &best_feat,
                              float &best_thresh) const;

    void trainLevelWise(const std::vector<std::vector<float>> &X,
                        const std::vector<int> &y);

    int buildNode(const std::vector<std::vector<float>> &X,
                  const std::vector<int> &sample_indices,
                  const std::vector<int> &y,
                  int depth);
};
