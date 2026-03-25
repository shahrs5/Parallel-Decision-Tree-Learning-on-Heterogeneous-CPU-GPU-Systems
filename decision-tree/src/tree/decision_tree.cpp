#include "decision_tree.h"

#include <algorithm>    // std::sort, std::max_element
#include <cassert>
#include <cmath>        // std::fabs
#include <cstdint>      // uint8_t
#include <limits>       // std::numeric_limits
#include <map>          // std::map (label counting — used only in computeGini/majorityLabel)
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

    n_samples_ = static_cast<int>(X.size());
    n_classes_ = *std::max_element(y.begin(), y.end()) + 1;

    // --- Presort each feature column once ---
    //
    // sorted_indices_[f] holds all sample indices sorted ascending by X[i][f].
    // Each buildNode call filters this list to active samples in O(N), avoiding
    // the O(N log N) re-sort that the naive approach paid at every tree node.
    int n_features = static_cast<int>(X[0].size());
    sorted_indices_.assign(n_features, std::vector<int>(n_samples_));
    for (int f = 0; f < n_features; ++f) {
        std::iota(sorted_indices_[f].begin(), sorted_indices_[f].end(), 0);
        std::sort(sorted_indices_[f].begin(), sorted_indices_[f].end(),
                  [&](int a, int b){ return X[a][f] < X[b][f]; });
    }

    // Build index set {0, 1, …, n-1} for the root — all samples participate.
    std::vector<int> all_indices(n_samples_);
    std::iota(all_indices.begin(), all_indices.end(), 0);

    // Active mask: active_mask[i] == 1 iff sample i is in the current node.
    // Root node: all samples are active.
    std::vector<uint8_t> active_mask(n_samples_, 1);

    buildNode(X, all_indices, y, /*depth=*/0, active_mask);
}

// ---------------------------------------------------------------------------
// buildNode — recursive CART split search (optimised sequential, Milestone 1)
//
// Two CPU optimisations over the naive approach:
//
//  1. Presorted feature columns (sorted_indices_ built once in train()):
//     Instead of sorting each feature at every node, we filter the globally
//     presorted list to the active samples in O(N).  Eliminates the dominant
//     O(N log N) sort cost from every internal node.
//
//  2. Incremental Gini scan:
//     Class-count arrays (left_cnt / right_cnt) are maintained as we sweep
//     through the sorted active samples.  Gini for each candidate threshold
//     is computed in O(K) (K = number of classes) rather than O(N), removing
//     the quadratic inner loop of the naive implementation.
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
                             int                                    depth,
                             const std::vector<uint8_t>&            active_mask)
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
    int   n           = static_cast<int>(sample_indices.size());
    float best_gain   = -std::numeric_limits<float>::infinity();
    int   best_feat   = -1;
    float best_thresh = 0.0f;

    // Total class counts at this node — used to initialise the right-side
    // counts before each feature scan.
    std::vector<int> total_cnt(n_classes_, 0);
    for (int idx : sample_indices)
        total_cnt[y[idx]]++;

    // Reusable buffers for the incremental scan (avoids per-feature allocation).
    std::vector<int> left_cnt(n_classes_);
    std::vector<int> right_cnt(n_classes_);
    std::vector<int> sorted_active;
    sorted_active.reserve(n);

    for (int f = 0; f < n_features; ++f) {
        // --- Optimisation 1: filter the presorted column to active samples ---
        // sorted_indices_[f] is globally sorted ascending by X[i][f].
        // We iterate it once (O(N_total)) and keep only active samples,
        // giving us the node's samples already in sorted order — no re-sort.
        sorted_active.clear();
        for (int idx : sorted_indices_[f]) {
            if (active_mask[idx])
                sorted_active.push_back(idx);
        }

        // --- Optimisation 2: incremental Gini scan ---
        // Initialise: all samples on the right, none on the left.
        std::fill(left_cnt.begin(),  left_cnt.end(),  0);
        right_cnt = total_cnt;
        int n_left = 0, n_right = n;

        for (int i = 0; i < n - 1; ++i) {
            int   idx = sorted_active[i];
            int   lbl = y[idx];

            // Move sample i from right to left.
            left_cnt[lbl]++;
            right_cnt[lbl]--;
            n_left++;
            n_right--;

            // Only evaluate a split at value boundaries (skip ties).
            float val_curr = X[idx][f];
            float val_next = X[sorted_active[i + 1]][f];
            if (val_curr == val_next)
                continue;

            // Enforce minimum leaf size.
            if (n_left  < min_samples_leaf_) continue;
            if (n_right < min_samples_leaf_) continue;

            // Gini for left child — O(K), no array rebuild.
            float g_left = 1.0f;
            for (int k = 0; k < n_classes_; ++k) {
                float p = static_cast<float>(left_cnt[k]) / n_left;
                g_left -= p * p;
            }
            // Gini for right child — O(K).
            float g_right = 1.0f;
            for (int k = 0; k < n_classes_; ++k) {
                float p = static_cast<float>(right_cnt[k]) / n_right;
                g_right -= p * p;
            }

            // Weighted Gini gain.
            float gain = nodes_[node_idx].gini
                       - (static_cast<float>(n_left)  / n) * g_left
                       - (static_cast<float>(n_right) / n) * g_right;

            if (gain > best_gain) {
                best_gain   = gain;
                best_feat   = f;
                best_thresh = 0.5f * (val_curr + val_next);
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

    // Build child active masks by setting/clearing entries for this split.
    // We reuse the parent mask pattern: copy it, then deactivate the samples
    // that go to the other side.
    std::vector<uint8_t> left_mask  = active_mask;
    std::vector<uint8_t> right_mask = active_mask;
    for (int idx : left_indices)  right_mask[idx] = 0;
    for (int idx : right_indices) left_mask[idx]  = 0;

    // Recurse.  IMPORTANT: nodes_ may reallocate during recursion.
    // Use nodes_[node_idx] (index) not a stored Node& after this point.
    int left_child  = buildNode(X, left_indices,  y, depth + 1, left_mask);
    int right_child = buildNode(X, right_indices, y, depth + 1, right_mask);

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
