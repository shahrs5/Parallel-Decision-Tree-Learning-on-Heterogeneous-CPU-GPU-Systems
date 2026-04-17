#include "decision_tree.h"

#include <algorithm> // std::sort, std::max_element
#include <cmath>     // std::fabs
#include <limits>    // std::numeric_limits
#include <map>       // std::map
#include <numeric>   // std::iota
#include <stdexcept>

// ---------------------------------------------------------------------------
// Helper: compute Gini directly from class-count map.
// More efficient than rebuilding label vectors for every threshold.
// ---------------------------------------------------------------------------
static float giniFromCounts(const std::map<int, int> &counts, int total)
{
    if (total == 0)
        return 0.0f;

    float gini = 1.0f;
    float n = static_cast<float>(total);

    for (const auto &[label, count] : counts)
    {
        float p = static_cast<float>(count) / n;
        gini -= p * p;
    }

    return gini;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

DecisionTree::DecisionTree(int max_depth, int min_samples_leaf)
    : max_depth_(max_depth), min_samples_leaf_(min_samples_leaf)
{
    if (max_depth_ < 0)
        throw std::runtime_error("DecisionTree::DecisionTree: max_depth must be >= 0");
    if (min_samples_leaf_ < 1)
        throw std::runtime_error("DecisionTree::DecisionTree: min_samples_leaf must be >= 1");
}
int DecisionTree::createEmptyNode()
{
    int node_idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    return node_idx;
}
// ---------------------------------------------------------------------------
// computeGini
// ---------------------------------------------------------------------------

float DecisionTree::computeGini(const std::vector<int> &labels)
{
    if (labels.empty())
        return 0.0f;

    std::map<int, int> counts;
    for (int label : labels)
        counts[label]++;

    return giniFromCounts(counts, static_cast<int>(labels.size()));
}

// ---------------------------------------------------------------------------
// majorityLabel
// ---------------------------------------------------------------------------

int DecisionTree::majorityLabel(const std::vector<int> &labels)
{
    if (labels.empty())
        throw std::runtime_error("DecisionTree::majorityLabel: labels are empty");

    std::map<int, int> counts;
    for (int label : labels)
        counts[label]++;

    return std::max_element(
               counts.begin(), counts.end(),
               [](const auto &a, const auto &b)
               {
                   return a.second < b.second;
               })
        ->first;
}

// ---------------------------------------------------------------------------
// train
// ---------------------------------------------------------------------------

void DecisionTree::train(const std::vector<std::vector<float>> &X,
                         const std::vector<int> &y)
{
    if (X.empty() || y.empty())
        throw std::runtime_error("DecisionTree::train: training data is empty");
    if (X.size() != y.size())
        throw std::runtime_error("DecisionTree::train: X and y size mismatch");
    if (X[0].empty())
        throw std::runtime_error("DecisionTree::train: feature rows are empty");

    const std::size_t feature_count = X[0].size();
    for (std::size_t i = 1; i < X.size(); ++i)
    {
        if (X[i].size() != feature_count)
        {
            throw std::runtime_error("DecisionTree::train: inconsistent feature count across rows");
        }
    }

    nodes_.clear();
    trainLevelWise(X, y);
}
void DecisionTree::trainLevelWise(const std::vector<std::vector<float>> &X,
                                  const std::vector<int> &y)
{
    std::vector<int> all_indices(X.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);

    int root_idx = createEmptyNode();

    std::vector<PendingNode> current_level;
    current_level.push_back({root_idx, all_indices, 0});

    while (!current_level.empty())
    {
        std::vector<PendingNode> next_level;

        for (const auto &work : current_level)
        {
            int node_idx = work.node_idx;
            const std::vector<int> &sample_indices = work.sample_indices;
            int depth = work.depth;

            std::vector<int> labels;
            labels.reserve(sample_indices.size());
            for (int idx : sample_indices)
            {
                labels.push_back(y[idx]);
            }

            nodes_[node_idx].sample_count = static_cast<int>(sample_indices.size());
            nodes_[node_idx].gini = computeGini(labels);
            nodes_[node_idx].label = majorityLabel(labels);

            bool is_pure = (nodes_[node_idx].gini < 1e-9f);
            bool too_few = (nodes_[node_idx].sample_count < 2 * min_samples_leaf_);
            bool at_max_depth = (depth >= max_depth_);

            if (is_pure || too_few || at_max_depth)
            {
                nodes_[node_idx].is_leaf = true;
                continue;
            }
            int best_feat = -1;
            float best_thresh = 0.0f;

            bool found_split = findBestSplitForNode(X, sample_indices, y,
                                                    best_feat, best_thresh);

            if (!found_split)
            {
                nodes_[node_idx].is_leaf = true;
                continue;
            }
            std::vector<int> left_indices, right_indices;
            left_indices.reserve(sample_indices.size());
            right_indices.reserve(sample_indices.size());

            for (int idx : sample_indices)
            {
                if (X[idx][best_feat] <= best_thresh)
                    left_indices.push_back(idx);
                else
                    right_indices.push_back(idx);
            }

            if (left_indices.empty() || right_indices.empty())
            {
                nodes_[node_idx].is_leaf = true;
                continue;
            }
            nodes_[node_idx].feature_index = best_feat;
            nodes_[node_idx].threshold = best_thresh;
            nodes_[node_idx].is_leaf = false;

            int left_child = createEmptyNode();
            int right_child = createEmptyNode();

            nodes_[node_idx].left_child = left_child;
            nodes_[node_idx].right_child = right_child;

            next_level.push_back({left_child, left_indices, depth + 1});
            next_level.push_back({right_child, right_indices, depth + 1});
        }

        current_level = std::move(next_level);
    }
}
bool DecisionTree::findBestSplitForNode(const std::vector<std::vector<float>> &X,
                                        const std::vector<int> &sample_indices,
                                        const std::vector<int> &y,
                                        int &best_feat,
                                        float &best_thresh) const
{
    int n_features = static_cast<int>(X[0].size());
    float best_gain = -std::numeric_limits<float>::infinity();

    best_feat = -1;
    best_thresh = 0.0f;

    std::vector<int> labels;
    labels.reserve(sample_indices.size());
    for (int idx : sample_indices)
    {
        labels.push_back(y[idx]);
    }

    float parent_gini = computeGini(labels);

    for (int f = 0; f < n_features; ++f)
    {
        std::vector<std::pair<float, int>> fvals;
        fvals.reserve(sample_indices.size());

        for (int idx : sample_indices)
        {
            fvals.push_back({X[idx][f], y[idx]});
        }

        std::sort(fvals.begin(), fvals.end());

        std::map<int, int> left_counts;
        std::map<int, int> right_counts;
        for (const auto &[val, lbl] : fvals)
            right_counts[lbl]++;

        int left_size = 0;
        int right_size = static_cast<int>(fvals.size());

        for (std::size_t i = 0; i + 1 < fvals.size(); ++i)
        {
            int lbl = fvals[i].second;

            left_counts[lbl]++;
            right_counts[lbl]--;
            if (right_counts[lbl] == 0)
                right_counts.erase(lbl);

            left_size++;
            right_size--;

            if (fvals[i].first == fvals[i + 1].first)
                continue;

            if (left_size < min_samples_leaf_ || right_size < min_samples_leaf_)
                continue;

            float thresh = 0.5f * (fvals[i].first + fvals[i + 1].first);

            float n = static_cast<float>(sample_indices.size());
            float left_gini = giniFromCounts(left_counts, left_size);
            float right_gini = giniFromCounts(right_counts, right_size);

            float gain = parent_gini - (static_cast<float>(left_size) / n) * left_gini - (static_cast<float>(right_size) / n) * right_gini;

            if (gain > best_gain)
            {
                best_gain = gain;
                best_feat = f;
                best_thresh = thresh;
            }
        }
    }

    return best_feat != -1;
}
// ---------------------------------------------------------------------------
// buildNode
// ---------------------------------------------------------------------------

int DecisionTree::buildNode(const std::vector<std::vector<float>> &X,
                            const std::vector<int> &sample_indices,
                            const std::vector<int> &y,
                            int depth)
{
    if (sample_indices.empty())
        throw std::runtime_error("DecisionTree::buildNode: empty sample_indices");

    // Collect labels for samples at this node
    std::vector<int> labels;
    labels.reserve(sample_indices.size());
    for (int idx : sample_indices)
    {
        if (idx < 0 || static_cast<std::size_t>(idx) >= y.size())
            throw std::runtime_error("DecisionTree::buildNode: sample index out of range");
        labels.push_back(y[idx]);
    }

    // Allocate this node
    int node_idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back();

    nodes_[node_idx].sample_count = static_cast<int>(sample_indices.size());
    nodes_[node_idx].gini = computeGini(labels);
    nodes_[node_idx].label = majorityLabel(labels);

    // Leaf conditions
    bool is_pure = (nodes_[node_idx].gini < 1e-9f);
    bool too_few = (nodes_[node_idx].sample_count < 2 * min_samples_leaf_);
    bool at_max_depth = (depth >= max_depth_);

    if (is_pure || too_few || at_max_depth)
    {
        nodes_[node_idx].is_leaf = true;
        return node_idx;
    }

    int n_features = static_cast<int>(X[0].size());
    float best_gain = -std::numeric_limits<float>::infinity();
    int best_feat = -1;
    float best_thresh = 0.0f;

    // -----------------------------------------------------------------------
    // Efficient exact split evaluation:
    // sort once per feature at this node, then sweep left-to-right while
    // maintaining incremental class counts.
    // -----------------------------------------------------------------------
    for (int f = 0; f < n_features; ++f)
    {
        std::vector<std::pair<float, int>> fvals;
        fvals.reserve(sample_indices.size());

        for (int idx : sample_indices)
        {
            fvals.push_back({X[idx][f], y[idx]});
        }

        std::sort(fvals.begin(), fvals.end());

        std::map<int, int> left_counts;
        std::map<int, int> right_counts;
        for (const auto &[val, lbl] : fvals)
            right_counts[lbl]++;

        int left_size = 0;
        int right_size = static_cast<int>(fvals.size());

        for (std::size_t i = 0; i + 1 < fvals.size(); ++i)
        {
            int lbl = fvals[i].second;

            // Move current sample from right partition to left partition
            left_counts[lbl]++;
            right_counts[lbl]--;
            if (right_counts[lbl] == 0)
                right_counts.erase(lbl);

            left_size++;
            right_size--;

            // Skip identical adjacent values: no valid threshold between them
            if (fvals[i].first == fvals[i + 1].first)
                continue;

            // Enforce minimum child size
            if (left_size < min_samples_leaf_ || right_size < min_samples_leaf_)
                continue;

            float thresh = 0.5f * (fvals[i].first + fvals[i + 1].first);

            float n = static_cast<float>(sample_indices.size());
            float left_gini = giniFromCounts(left_counts, left_size);
            float right_gini = giniFromCounts(right_counts, right_size);

            float gain = nodes_[node_idx].gini - (static_cast<float>(left_size) / n) * left_gini - (static_cast<float>(right_size) / n) * right_gini;

            if (gain > best_gain)
            {
                best_gain = gain;
                best_feat = f;
                best_thresh = thresh;
            }
        }
    }

    // No useful split found -> leaf
    if (best_feat == -1)
    {
        nodes_[node_idx].is_leaf = true;
        return node_idx;
    }

    // Partition sample indices using best split
    std::vector<int> left_indices, right_indices;
    left_indices.reserve(sample_indices.size());
    right_indices.reserve(sample_indices.size());

    for (int idx : sample_indices)
    {
        if (X[idx][best_feat] <= best_thresh)
            left_indices.push_back(idx);
        else
            right_indices.push_back(idx);
    }

    // Safety check
    if (left_indices.empty() || right_indices.empty())
    {
        nodes_[node_idx].is_leaf = true;
        return node_idx;
    }

    nodes_[node_idx].feature_index = best_feat;
    nodes_[node_idx].threshold = best_thresh;

    int left_child = buildNode(X, left_indices, y, depth + 1);
    int right_child = buildNode(X, right_indices, y, depth + 1);

    nodes_[node_idx].left_child = left_child;
    nodes_[node_idx].right_child = right_child;

    return node_idx;
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------

int DecisionTree::predict(const std::vector<float> &sample) const
{
    if (nodes_.empty())
        throw std::runtime_error("DecisionTree::predict: tree has not been trained");

    int idx = 0;
    while (!nodes_[idx].is_leaf)
    {
        const Node &node = nodes_[idx];

        if (node.feature_index < 0 ||
            static_cast<std::size_t>(node.feature_index) >= sample.size())
        {
            throw std::runtime_error("DecisionTree::predict: sample has fewer features than expected");
        }

        if (node.left_child < 0 || node.right_child < 0)
        {
            throw std::runtime_error("DecisionTree::predict: internal node has invalid child index");
        }

        idx = (sample[node.feature_index] <= node.threshold)
                  ? node.left_child
                  : node.right_child;
    }

    return nodes_[idx].label;
}