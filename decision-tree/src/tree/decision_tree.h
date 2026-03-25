#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "node.h"

// ---------------------------------------------------------------------------
// DecisionTree — CART-style binary decision tree classifier.
//
// Milestone 1 (this file): sequential training with Gini impurity.
// Milestone 2: split evaluation will be offloaded to the GPU.
//   The CPU loop in buildNode() will be replaced by a call to a host wrapper
//   around the CUDA kernel in src/gpu/split_kernel.cu.  The node allocation
//   pattern (nodes_ vector + index bookkeeping) stays unchanged.
// Milestone 3: DecisionTree is used as-is inside a RandomForest ensemble.
//   Each tree trains on a bootstrap sample of the data.
//
// Node storage: flat array-of-structs — see src/tree/node.h.
//   Root is always nodes_[0].  Children are stored by index, not pointer,
//   making the tree trivially serialisable and GPU-copyable.
// ---------------------------------------------------------------------------
class DecisionTree {
public:
    // max_depth:        stop splitting when this depth is reached.
    // min_samples_leaf: each child must receive at least this many samples.
    explicit DecisionTree(int max_depth       = 10,
                          int min_samples_leaf = 1);

    // Train on feature matrix X and integer label vector y.
    // X: [n_samples][n_features], all float.
    // y: class labels, length n_samples.
    void train(const std::vector<std::vector<float>>& X,
               const std::vector<int>&                y);

    // Predict the class label for a single feature vector.
    int predict(const std::vector<float>& sample) const;

    // Read-only access to the node array.
    // Milestone 2: use this to inspect/copy the tree structure to the GPU.
    const std::vector<Node>& nodes() const { return nodes_; }

    // -----------------------------------------------------------------------
    // Static utility functions
    // -----------------------------------------------------------------------

    // Compute Gini impurity for a set of class labels.
    //
    //   Gini = 1 - Σ_k  p_k²
    //
    // where p_k = (count of class k) / (total samples).
    // Returns 0.0 for a pure node, approaches (1 - 1/K) for K equal classes.
    static float computeGini(const std::vector<int>& labels);

    // Return the most frequent label in the set (used for leaf prediction).
    // Ties broken by lowest label value (deterministic, map-ordered).
    static int majorityLabel(const std::vector<int>& labels);

private:
    int max_depth_;
    int min_samples_leaf_;

    // All tree nodes in BFS-like allocation order.  Root = nodes_[0].
    std::vector<Node> nodes_;

    // --- Presorted index tables (built once in train(), reused every node) ---
    //
    // sorted_indices_[f][i] = the i-th sample index when all samples are
    // sorted ascending by feature f.  Eliminates the per-node O(N log N) sort.
    int n_samples_ = 0;
    int n_classes_ = 0;
    std::vector<std::vector<int>> sorted_indices_;

    // Recursively build a node for the given subset of training samples.
    // Returns the index of the newly created node in nodes_.
    //
    // Note on reallocation: nodes_.push_back() may invalidate references.
    // Always use nodes_[idx] (index) rather than a stored reference after
    // any recursive call that may push more nodes.
    int buildNode(const std::vector<std::vector<float>>& X,
                  const std::vector<int>&                sample_indices,
                  const std::vector<int>&                y,
                  int                                    depth,
                  const std::vector<uint8_t>&            active_mask);
};
