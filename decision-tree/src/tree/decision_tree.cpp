#include "decision_tree.h"

#include <algorithm>    // std::sort, std::max_element
#include <cassert>
#include <cmath>        // std::fabs
#include <limits>       // std::numeric_limits
#include <map>          // std::map (label counting)
#include <numeric>      // std::iota
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

DecisionTree::DecisionTree(int max_depth, int min_samples_leaf)
    : max_depth_(max_depth), min_samples_leaf_(min_samples_leaf) {}

// ---------------------------------------------------------------------------
// computeGini
//
// Gini impurity measures the probability that a randomly chosen sample would
// be mis-classified if labelled according to the distribution at this node.
//
//   Gini = 1 - Σ_k  p_k²
//
// Implementation counts label frequencies in a map, then subtracts p²
// for each class from 1.  O(n log K) where K = number of distinct classes.
// ---------------------------------------------------------------------------

float DecisionTree::computeGini(const std::vector<int>& labels) {
    if (labels.empty())
        return 0.0f;

    // Count how many samples belong to each class.
    std::map<int, int> counts;
    for (int label : labels)
        counts[label]++;

    float gini = 1.0f;
    float n    = static_cast<float>(labels.size());

    for (const auto& [label, count] : counts) {
        float p  = static_cast<float>(count) / n;
        gini    -= p * p;
    }

    return gini;
}

// ---------------------------------------------------------------------------
// majorityLabel
// ---------------------------------------------------------------------------

int DecisionTree::majorityLabel(const std::vector<int>& labels) {
    std::map<int, int> counts;
    for (int label : labels)
        counts[label]++;

    // max_element with a comparator on the count (second) field.
    return std::max_element(
               counts.begin(), counts.end(),
               [](const auto& a, const auto& b) {
                   return a.second < b.second;
               })
        ->first;
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

void DecisionTree::train(const std::vector<std::vector<float>>& X,
                         const std::vector<int>&                y) {
    if (X.empty() || y.empty())
        throw std::runtime_error("DecisionTree::train: training data is empty");
    if (X.size() != y.size())
        throw std::runtime_error("DecisionTree::train: X and y size mismatch");

    nodes_.clear();

    // Build index set {0, 1, …, n-1} for the root — all samples participate.
    std::vector<int> all_indices(X.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);

    buildNode(X, all_indices, y, /*depth=*/0);
}

// ---------------------------------------------------------------------------
// buildNode — recursive CART split search (sequential, Milestone 1)
//
// Milestone 2 replacement plan:
//   The inner for-loop over features (marked below) will be replaced by:
//     1. A GPU call that builds histograms for all features in parallel.
//     2. A second GPU pass that evaluates Gini gain per bin and returns
//        (best_feature, best_threshold) to the CPU.
//   Everything outside that loop (node allocation, partitioning, recursion)
//   remains on the CPU and is unchanged.
// ---------------------------------------------------------------------------

int DecisionTree::buildNode(const std::vector<std::vector<float>>& X,
                             const std::vector<int>&                sample_indices,
                             const std::vector<int>&                y,
                             int                                    depth)
{
    // Collect the labels for samples at this node.
    std::vector<int> labels;
    labels.reserve(sample_indices.size());
    for (int idx : sample_indices)
        labels.push_back(y[idx]);

    // Allocate a slot for this node BEFORE recursing (children get higher indices).
    int  node_idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back();

    // Initialise fields that are valid for every node type.
    nodes_[node_idx].sample_count = static_cast<int>(sample_indices.size());
    nodes_[node_idx].gini         = computeGini(labels);
    nodes_[node_idx].label        = majorityLabel(labels);

    // --- Leaf conditions ---
    bool is_pure      = (nodes_[node_idx].gini < 1e-9f);
    bool too_few      = (nodes_[node_idx].sample_count < 2 * min_samples_leaf_);
    bool at_max_depth = (depth >= max_depth_);

    if (is_pure || too_few || at_max_depth) {
        nodes_[node_idx].is_leaf = true;
        return node_idx;
    }

    // --- Find the best (feature, threshold) split ---
    // *** Milestone 2: replace this block with a GPU histogram kernel call. ***

    int   n_features  = static_cast<int>(X[0].size());
    float best_gain   = -std::numeric_limits<float>::infinity();
    int   best_feat   = -1;
    float best_thresh = 0.0f;

    for (int f = 0; f < n_features; ++f) {
        // Collect and sort (value, label) pairs for this feature.
        std::vector<std::pair<float, int>> fvals;
        fvals.reserve(sample_indices.size());
        for (int idx : sample_indices)
            fvals.push_back({X[idx][f], y[idx]});
        std::sort(fvals.begin(), fvals.end());

        // Evaluate candidate split thresholds at midpoints between distinct values.
        // This is the exact (non-histogram) method; the GPU version will bin values.
        for (std::size_t i = 0; i + 1 < fvals.size(); ++i) {
            // Skip identical adjacent values — same threshold, different split.
            if (fvals[i].first == fvals[i + 1].first)
                continue;

            float thresh = 0.5f * (fvals[i].first + fvals[i + 1].first);

            // Partition labels into left (≤ thresh) and right (> thresh) subsets.
            std::vector<int> left_labels, right_labels;
            for (const auto& [val, lbl] : fvals) {
                if (val <= thresh) left_labels.push_back(lbl);
                else               right_labels.push_back(lbl);
            }

            // Enforce minimum leaf size on both sides.
            if (static_cast<int>(left_labels.size())  < min_samples_leaf_) continue;
            if (static_cast<int>(right_labels.size()) < min_samples_leaf_) continue;

            // Weighted Gini gain: parent impurity minus weighted child impurities.
            float n    = static_cast<float>(sample_indices.size());
            float gain = nodes_[node_idx].gini
                       - (static_cast<float>(left_labels.size())  / n) * computeGini(left_labels)
                       - (static_cast<float>(right_labels.size()) / n) * computeGini(right_labels);

            if (gain > best_gain) {
                best_gain   = gain;
                best_feat   = f;
                best_thresh = thresh;
            }
        }
    }

    // No split improved impurity — make a leaf.
    if (best_feat == -1) {
        nodes_[node_idx].is_leaf = true;
        return node_idx;
    }

    // --- Partition sample indices using the best split ---
    std::vector<int> left_indices, right_indices;
    for (int idx : sample_indices) {
        if (X[idx][best_feat] <= best_thresh) left_indices.push_back(idx);
        else                                   right_indices.push_back(idx);
    }

    // Store split parameters on the node before recursing.
    nodes_[node_idx].feature_index = best_feat;
    nodes_[node_idx].threshold     = best_thresh;

    // Recurse.  IMPORTANT: nodes_ may reallocate during recursion.
    // Use nodes_[node_idx] (index) not a stored Node& after this point.
    int left_child  = buildNode(X, left_indices,  y, depth + 1);
    int right_child = buildNode(X, right_indices, y, depth + 1);

    // Re-index after potential reallocation.
    nodes_[node_idx].left_child  = left_child;
    nodes_[node_idx].right_child = right_child;

    return node_idx;
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

int DecisionTree::predict(const std::vector<float>& sample) const {
    if (nodes_.empty())
        throw std::runtime_error("DecisionTree::predict: tree has not been trained");

    int idx = 0; // start at root (always nodes_[0])
    while (!nodes_[idx].is_leaf) {
        const Node& node = nodes_[idx];
        idx = (sample[node.feature_index] <= node.threshold)
                  ? node.left_child
                  : node.right_child;
    }
    return nodes_[idx].label;
}
