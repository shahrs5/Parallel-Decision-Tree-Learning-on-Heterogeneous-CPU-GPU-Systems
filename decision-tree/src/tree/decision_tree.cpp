#include "decision_tree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// OpenMP (Person 3): parallelise the level-wise node loop.
#ifdef USE_OPENMP
#include <omp.h>
#endif

// CUDA GPU split path (Person 3).
#ifdef USE_CUDA
#include "../gpu/split_kernel.cuh"
#endif

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------
static float giniFromCounts(const std::map<int, int> &counts, int total)
{
    if (total == 0) return 0.0f;
    float gini = 1.0f;
    float n    = static_cast<float>(total);
    for (const auto &[label, count] : counts) {
        float p = static_cast<float>(count) / n;
        gini -= p * p;
    }
    return gini;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
DecisionTree::DecisionTree(int max_depth,
                           int min_samples_leaf,
                           int feature_subsample,
                           unsigned seed)
    : max_depth_(max_depth),
      min_samples_leaf_(min_samples_leaf),
      feature_subsample_(feature_subsample),
      tree_seed_(seed)
{
    if (max_depth_        < 0) throw std::runtime_error("max_depth must be >= 0");
    if (min_samples_leaf_ < 1) throw std::runtime_error("min_samples_leaf must be >= 1");
}

DecisionTree::~DecisionTree()
{
#ifdef USE_CUDA
    if (d_X_) { freeGPUData(d_X_, d_y_); d_X_ = nullptr; d_y_ = nullptr; }
#endif
}

// ---------------------------------------------------------------------------
// Static utilities
// ---------------------------------------------------------------------------
float DecisionTree::computeGini(const std::vector<int> &labels)
{
    if (labels.empty()) return 0.0f;
    std::map<int, int> counts;
    for (int l : labels) counts[l]++;
    return giniFromCounts(counts, static_cast<int>(labels.size()));
}

int DecisionTree::majorityLabel(const std::vector<int> &labels)
{
    if (labels.empty()) throw std::runtime_error("majorityLabel: empty labels");
    std::map<int, int> counts;
    for (int l : labels) counts[l]++;
    return std::max_element(counts.begin(), counts.end(),
                            [](const auto &a, const auto &b) {
                                return a.second < b.second;
                            })->first;
}

// ---------------------------------------------------------------------------
// getPackedNodes — compact flat copy of nodes_ for GPU inference.
// ---------------------------------------------------------------------------
std::vector<FlatNode> DecisionTree::getPackedNodes() const
{
    std::vector<FlatNode> packed;
    packed.reserve(nodes_.size());
    for (const auto &n : nodes_) {
        FlatNode fn;
        fn.threshold     = n.threshold;
        fn.feature_index = n.is_leaf ? -1 : n.feature_index;
        fn.left_child    = n.left_child;
        fn.right_child   = n.right_child;
        fn.label         = n.label;
        packed.push_back(fn);
    }
    return packed;
}

// ---------------------------------------------------------------------------
// createEmptyNode
// ---------------------------------------------------------------------------
int DecisionTree::createEmptyNode()
{
    int idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    return idx;
}

// ---------------------------------------------------------------------------
// train — entry point
// Person 3: flatten X once, upload to GPU (CUDA path), then run level-wise.
// ---------------------------------------------------------------------------
void DecisionTree::train(const std::vector<std::vector<float>> &X,
                         const std::vector<int> &y)
{
    if (X.empty() || y.empty())          throw std::runtime_error("train: empty data");
    if (X.size() != y.size())            throw std::runtime_error("train: X/y size mismatch");
    if (X[0].empty())                    throw std::runtime_error("train: empty feature rows");

    n_samples_  = static_cast<int>(X.size());
    n_features_ = static_cast<int>(X[0].size());

    for (std::size_t i = 1; i < X.size(); ++i)
        if (static_cast<int>(X[i].size()) != n_features_)
            throw std::runtime_error("train: inconsistent feature count");

    // Flatten row-major for GPU upload (Person 3).
    X_flat_.resize((std::size_t)n_samples_ * n_features_);
    for (int i = 0; i < n_samples_; ++i)
        for (int f = 0; f < n_features_; ++f)
            X_flat_[(std::size_t)i * n_features_ + f] = X[i][f];

#ifdef USE_CUDA
    // Free any previous allocation, then upload once.
    // M3: skip the upload entirely if the CPU path is forced (use_gpu_=false).
    // RandomForest forces this for tree-level parallel training, so each tree
    // avoids paying cudaMalloc+cudaMemcpy for data it will never touch.
    if (d_X_) { freeGPUData(d_X_, d_y_); d_X_ = nullptr; d_y_ = nullptr; }
    if (use_gpu_) {
        uploadDataToGPU(X_flat_.data(), y.data(), n_samples_, n_features_, &d_X_, &d_y_);
    }
#endif

    nodes_.clear();
    trainLevelWise(X, y);
}

// ---------------------------------------------------------------------------
// trainLevelWise
// Person 3: level-wise BFS tree construction with optional OpenMP parallelism.
//
// Each iteration processes all nodes at the current depth simultaneously.
// With OpenMP:  nodes at the same level are split in parallel threads.
// With CUDA:    findBestSplitForNode → findBestSplitGPU per node.
// ---------------------------------------------------------------------------
void DecisionTree::trainLevelWise(const std::vector<std::vector<float>> &X,
                                  const std::vector<int> &y)
{
    std::vector<int> all_indices(X.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);

    int root_idx = createEmptyNode();
    std::vector<PendingNode> current_level;
    current_level.push_back({root_idx, all_indices, 0});

    // Per-tree RNG used to draw per-split feature subsets.
    // Drawn serially in the level loop below — keeps results deterministic and
    // avoids races with the OpenMP parallel-for over nodes.
    std::mt19937 tree_rng(tree_seed_);

    // Effective subsample size: <=0 or >= n_features means "use all features".
    int eff_sub = (feature_subsample_ > 0 && feature_subsample_ < n_features_)
                  ? feature_subsample_ : 0;

    while (!current_level.empty()) {
        int n_nodes = static_cast<int>(current_level.size());

        // Pre-allocate split results so OpenMP threads can write without races.
        std::vector<int>   best_feats(n_nodes, -1);
        std::vector<float> best_thresholds(n_nodes, 0.0f);
        std::vector<bool>  split_found(n_nodes, false);
        std::vector<bool>  make_leaf(n_nodes, false);

        // Per-node feature subsets. Empty vector => use all features.
        // Generated here in serial so the RNG sequence is deterministic
        // regardless of OpenMP scheduling in Phase 1 below.
        std::vector<std::vector<int>> feature_subsets(n_nodes);
        if (eff_sub > 0) {
            std::vector<int> pool(n_features_);
            std::iota(pool.begin(), pool.end(), 0);
            for (int ni = 0; ni < n_nodes; ++ni) {
                // Partial Fisher-Yates: shuffle just the first eff_sub slots.
                for (int i = 0; i < eff_sub; ++i) {
                    std::uniform_int_distribution<int> d(i, n_features_ - 1);
                    int j = d(tree_rng);
                    std::swap(pool[i], pool[j]);
                }
                feature_subsets[ni].assign(pool.begin(), pool.begin() + eff_sub);
                std::sort(feature_subsets[ni].begin(), feature_subsets[ni].end());
            }
        }

        // ----------------------------------------------------------------
        // Phase 1: compute splits.
        // Parallelism strategy (adaptive):
        //   - When n_nodes >= max_threads: node-level parallel (many nodes to share).
        //   - When n_nodes < max_threads:  run node loop serially; feature-level
        //     parallelism inside findBestSplitForNode() kicks in instead
        //     (controlled by omp_in_parallel() check there).
        // This avoids nested OpenMP and keeps full thread utilisation at both
        // shallow levels (few nodes, many features) and deep levels (many nodes).
        // ----------------------------------------------------------------
#ifdef USE_OPENMP
        // M3: also disable when already inside an OpenMP region (e.g. when
        // RandomForest is training trees in parallel) — nested OMP would
        // either no-op or oversubscribe cores.
        #pragma omp parallel for schedule(dynamic) \
            if(n_nodes >= omp_get_max_threads() && !omp_in_parallel())
#endif
        for (int ni = 0; ni < n_nodes; ++ni) {
            const auto &work   = current_level[ni];
            const auto &sidx   = work.sample_indices;
            int         nidx   = work.node_idx;
            int         depth  = work.depth;

            // Gather labels.
            std::vector<int> labels;
            labels.reserve(sidx.size());
            for (int idx : sidx) labels.push_back(y[idx]);

            float gini        = computeGini(labels);
            bool  is_pure     = (gini < 1e-9f);
            bool  too_few     = (static_cast<int>(sidx.size()) < 2 * min_samples_leaf_);
            bool  at_max_dep  = (depth >= max_depth_);

            // Write non-split fields (these nodes are pre-allocated).
            // Concurrent writes to different node indices are safe.
            nodes_[nidx].sample_count = static_cast<int>(sidx.size());
            nodes_[nidx].gini         = gini;
            nodes_[nidx].label        = majorityLabel(labels);

            if (is_pure || too_few || at_max_dep) {
                make_leaf[ni] = true;
                continue;
            }

            int   bf = -1;
            float bt = 0.0f;
            bool  found = false;

            const std::vector<int>& fsub = feature_subsets[ni];
#ifdef USE_CUDA
            if (use_gpu_) {
                // GPU histogram path (Person 2 kernels, Person 3 integration).
                std::map<int,int> cls_map;
                for (int l : labels) cls_map[l]++;
                int n_classes = static_cast<int>(cls_map.size());

                findBestSplitGPU(
                    d_X_, d_y_,
                    X_flat_.data(), y.data(),
                    sidx.data(), static_cast<int>(sidx.size()),
                    n_features_,
                    /*n_bins=*/32,
                    n_classes,
                    gini,
                    min_samples_leaf_,
                    fsub.empty() ? nullptr : fsub.data(),
                    static_cast<int>(fsub.size()),
                    bf, bt);
                found = (bf >= 0);
            } else {
                // CPU exact path (forced via setUseGPU(false) for comparison).
                found = findBestSplitForNode(X, sidx, y, fsub, bf, bt);
            }
#else
            // CPU path (no CUDA build).
            found = findBestSplitForNode(X, sidx, y, fsub, bf, bt);
#endif
            best_feats[ni]      = bf;
            best_thresholds[ni] = bt;
            split_found[ni]     = found;
        }

        // ----------------------------------------------------------------
        // Phase 2: apply splits, allocate children (serial — tree mutation)
        // ----------------------------------------------------------------
        std::vector<PendingNode> next_level;

        for (int ni = 0; ni < n_nodes; ++ni) {
            const auto &work  = current_level[ni];
            int         nidx  = work.node_idx;
            int         depth = work.depth;
            const auto &sidx  = work.sample_indices;

            if (make_leaf[ni] || !split_found[ni]) {
                nodes_[nidx].is_leaf = true;
                continue;
            }

            int   bf = best_feats[ni];
            float bt = best_thresholds[ni];

            std::vector<int> left_idx, right_idx;
            left_idx.reserve(sidx.size());
            right_idx.reserve(sidx.size());
            for (int idx : sidx) {
                if (X[idx][bf] <= bt) left_idx.push_back(idx);
                else                   right_idx.push_back(idx);
            }

            if (left_idx.empty() || right_idx.empty()) {
                nodes_[nidx].is_leaf = true;
                continue;
            }

            nodes_[nidx].feature_index = bf;
            nodes_[nidx].threshold     = bt;
            nodes_[nidx].is_leaf       = false;

            int lc = createEmptyNode();
            int rc = createEmptyNode();
            nodes_[nidx].left_child  = lc;
            nodes_[nidx].right_child = rc;

            next_level.push_back({lc, std::move(left_idx),  depth + 1});
            next_level.push_back({rc, std::move(right_idx), depth + 1});
        }

        current_level = std::move(next_level);
    }
}

// ---------------------------------------------------------------------------
// findBestSplitForNode — CPU exact split (Milestone 1 code, unchanged)
// ---------------------------------------------------------------------------
bool DecisionTree::findBestSplitForNode(
    const std::vector<std::vector<float>> &X,
    const std::vector<int> &sample_indices,
    const std::vector<int> &y,
    const std::vector<int> &feature_subset,
    int   &best_feat,
    float &best_thresh) const
{
    int n_features = static_cast<int>(X[0].size());
    best_feat   = -1;
    best_thresh = 0.0f;

    std::vector<int> labels;
    labels.reserve(sample_indices.size());
    for (int idx : sample_indices) labels.push_back(y[idx]);
    float parent_gini = computeGini(labels);

    // Decide which features to evaluate.
    //   - feature_subset empty   -> evaluate every feature (M2 single-tree path).
    //   - feature_subset present -> Random Forest mode; only those features.
    std::vector<int> all_features;
    const std::vector<int>* feats_to_use = &feature_subset;
    if (feature_subset.empty()) {
        all_features.resize(n_features);
        std::iota(all_features.begin(), all_features.end(), 0);
        feats_to_use = &all_features;
    }
    int n_evaluate = static_cast<int>(feats_to_use->size());

    // One result slot per feature — threads write to separate indices, no races.
    // Slots not in the subset stay at -inf and are ignored by the reduction.
    std::vector<float> f_gain(n_features, -std::numeric_limits<float>::infinity());
    std::vector<float> f_thresh(n_features, 0.0f);

    // Feature-level parallelism:
    //   Activates only when:
    //   (a) not already inside a parallel region (no nested OpenMP), AND
    //   (b) enough samples in this node to make thread overhead worthwhile.
    //   Threshold 256: below this the sort+sweep finishes in < thread-launch time.
    int n_active = static_cast<int>(sample_indices.size());
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) if(!omp_in_parallel() && n_active >= 256)
#endif
    for (int fi = 0; fi < n_evaluate; ++fi) {
        int f = (*feats_to_use)[fi];
        std::vector<std::pair<float, int>> fvals;
        fvals.reserve(sample_indices.size());
        for (int idx : sample_indices)
            fvals.push_back({X[idx][f], y[idx]});
        std::sort(fvals.begin(), fvals.end());

        std::map<int,int> left_counts, right_counts;
        for (const auto &[v, l] : fvals) right_counts[l]++;

        int left_size = 0, right_size = static_cast<int>(fvals.size());
        float local_gain   = -std::numeric_limits<float>::infinity();
        float local_thresh = 0.0f;

        for (std::size_t i = 0; i + 1 < fvals.size(); ++i) {
            int lbl = fvals[i].second;
            left_counts[lbl]++;
            right_counts[lbl]--;
            if (right_counts[lbl] == 0) right_counts.erase(lbl);
            ++left_size; --right_size;

            if (fvals[i].first == fvals[i + 1].first) continue;
            if (left_size  < min_samples_leaf_)        continue;
            if (right_size < min_samples_leaf_)        continue;

            float thresh = 0.5f * (fvals[i].first + fvals[i + 1].first);
            float n      = static_cast<float>(sample_indices.size());
            float gain   = parent_gini
                         - (static_cast<float>(left_size)  / n) * giniFromCounts(left_counts,  left_size)
                         - (static_cast<float>(right_size) / n) * giniFromCounts(right_counts, right_size);
            if (gain > local_gain) { local_gain = gain; local_thresh = thresh; }
        }
        f_gain[f]   = local_gain;
        f_thresh[f] = local_thresh;
    }

    // Serial reduction — pick the best feature across all f_gain slots.
    float best_gain = -std::numeric_limits<float>::infinity();
    for (int f = 0; f < n_features; ++f) {
        if (f_gain[f] > best_gain) {
            best_gain   = f_gain[f];
            best_feat   = f;
            best_thresh = f_thresh[f];
        }
    }
    return best_feat != -1;
}

// ---------------------------------------------------------------------------
// buildNode — recursive builder kept for compatibility, not used in M2 path
// ---------------------------------------------------------------------------
int DecisionTree::buildNode(const std::vector<std::vector<float>> &X,
                            const std::vector<int> &sample_indices,
                            const std::vector<int> &y,
                            int depth)
{
    if (sample_indices.empty())
        throw std::runtime_error("buildNode: empty sample_indices");

    std::vector<int> labels;
    labels.reserve(sample_indices.size());
    for (int idx : sample_indices) labels.push_back(y[idx]);

    int node_idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    nodes_[node_idx].sample_count = static_cast<int>(sample_indices.size());
    nodes_[node_idx].gini         = computeGini(labels);
    nodes_[node_idx].label        = majorityLabel(labels);

    if (nodes_[node_idx].gini < 1e-9f ||
        nodes_[node_idx].sample_count < 2 * min_samples_leaf_ ||
        depth >= max_depth_) {
        nodes_[node_idx].is_leaf = true;
        return node_idx;
    }

    int bf = -1; float bt = 0.0f;
    std::vector<int> empty_subset;   // legacy path: evaluate all features
    if (!findBestSplitForNode(X, sample_indices, y, empty_subset, bf, bt)) {
        nodes_[node_idx].is_leaf = true;
        return node_idx;
    }

    std::vector<int> left_idx, right_idx;
    for (int idx : sample_indices) {
        if (X[idx][bf] <= bt) left_idx.push_back(idx);
        else                   right_idx.push_back(idx);
    }
    if (left_idx.empty() || right_idx.empty()) {
        nodes_[node_idx].is_leaf = true;
        return node_idx;
    }

    nodes_[node_idx].feature_index = bf;
    nodes_[node_idx].threshold     = bt;
    int lc = buildNode(X, left_idx,  y, depth + 1);
    int rc = buildNode(X, right_idx, y, depth + 1);
    nodes_[node_idx].left_child  = lc;
    nodes_[node_idx].right_child = rc;
    return node_idx;
}

// ---------------------------------------------------------------------------
// predict
// ---------------------------------------------------------------------------
int DecisionTree::predict(const std::vector<float> &sample) const
{
    if (nodes_.empty()) throw std::runtime_error("predict: tree not trained");

    int idx = 0;
    while (!nodes_[idx].is_leaf) {
        const Node &node = nodes_[idx];
        if (node.feature_index < 0 ||
            static_cast<std::size_t>(node.feature_index) >= sample.size())
            throw std::runtime_error("predict: sample too narrow for tree");
        if (node.left_child < 0 || node.right_child < 0)
            throw std::runtime_error("predict: invalid child index");
        idx = (sample[node.feature_index] <= node.threshold)
              ? node.left_child : node.right_child;
    }
    return nodes_[idx].label;
}
