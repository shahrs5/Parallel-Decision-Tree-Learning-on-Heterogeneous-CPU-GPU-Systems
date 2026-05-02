#include "random_forest.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
RandomForest::RandomForest(int n_trees,
                           int max_depth,
                           int min_samples_leaf,
                           int feature_subsample,
                           unsigned seed)
    : n_trees_(n_trees),
      max_depth_(max_depth),
      min_samples_leaf_(min_samples_leaf),
      feature_subsample_(feature_subsample),
      seed_(seed)
{
    if (n_trees_ < 1)
        throw std::runtime_error("RandomForest: n_trees must be >= 1");
}

// ---------------------------------------------------------------------------
// bootstrapSample — draw N samples with replacement.
//
// Each tree gets its own RNG seed (derived deterministically from seed_)
// so that all trees see different bootstrap samples but the whole forest
// is reproducible across runs.
// ---------------------------------------------------------------------------
void RandomForest::bootstrapSample(
    const std::vector<std::vector<float>> &X,
    const std::vector<int>                &y,
    unsigned                               rng_seed,
    std::vector<std::vector<float>>       &X_boot,
    std::vector<int>                      &y_boot)
{
    int n = static_cast<int>(X.size());
    std::mt19937 rng(rng_seed);
    std::uniform_int_distribution<int> dist(0, n - 1);

    X_boot.clear();
    y_boot.clear();
    X_boot.reserve(n);
    y_boot.reserve(n);

    for (int i = 0; i < n; ++i) {
        int idx = dist(rng);
        X_boot.push_back(X[idx]);
        y_boot.push_back(y[idx]);
    }
}

// ---------------------------------------------------------------------------
// train — serial loop over trees (Step 1 of M3).
// Tree-level OpenMP comes in a later step.
// ---------------------------------------------------------------------------
void RandomForest::train(const std::vector<std::vector<float>> &X,
                         const std::vector<int>                &y)
{
    if (X.empty() || y.empty())
        throw std::runtime_error("RandomForest::train: empty data");
    if (X.size() != y.size())
        throw std::runtime_error("RandomForest::train: X/y size mismatch");

    // Resolve the effective feature-subsample size now that we know F.
    int n_features = static_cast<int>(X[0].size());
    int eff_sub;
    if (feature_subsample_ < 0)
        eff_sub = std::max(1, static_cast<int>(std::sqrt((double)n_features)));
    else if (feature_subsample_ == 0 || feature_subsample_ >= n_features)
        eff_sub = -1;   // -1 inside DecisionTree => use all features
    else
        eff_sub = feature_subsample_;

    // Generate one independent seed per tree from the master RNG.
    std::vector<unsigned> tree_seeds(n_trees_);
    {
        std::mt19937 master(seed_);
        for (int t = 0; t < n_trees_; ++t)
            tree_seeds[t] = master();
    }

    // Build the per-tree DecisionTree objects with their own seeds + subsample.
    trees_.clear();
    trees_.reserve(n_trees_);
    for (int t = 0; t < n_trees_; ++t) {
        auto tree = std::make_unique<DecisionTree>(
            max_depth_, min_samples_leaf_, eff_sub, tree_seeds[t]);
        // Force CPU split path: under tree-level OpenMP, kernel launches from
        // multiple threads would serialize on the GPU and the timing globals
        // in split_kernel.cu would race. The single-tree GPU benchmark in
        // main.cpp does not go through RandomForest, so this is fine.
        tree->setUseGPU(false);
        trees_.push_back(std::move(tree));
    }

    // Tree-level OpenMP: trees are independent, so the loop is embarrassingly
    // parallel. The DecisionTree internals self-disable via !omp_in_parallel()
    // once we are inside this region, which keeps thread counts sane.
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic) if(n_trees_ > 1)
#endif
    for (int t = 0; t < n_trees_; ++t) {
        std::vector<std::vector<float>> X_boot;
        std::vector<int>                y_boot;
        bootstrapSample(X, y, tree_seeds[t], X_boot, y_boot);
        trees_[t]->train(X_boot, y_boot);
    }
}

// ---------------------------------------------------------------------------
// predict — majority vote across trees.
// ---------------------------------------------------------------------------
int RandomForest::predict(const std::vector<float> &sample) const
{
    if (trees_.empty())
        throw std::runtime_error("RandomForest::predict: forest not trained");

    std::map<int, int> votes;
    for (const auto &tree : trees_)
        votes[tree->predict(sample)]++;

    return std::max_element(
        votes.begin(), votes.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; })
        ->first;
}

// ---------------------------------------------------------------------------
// predictBatch — sample-level parallel inference.
// Each iteration computes a per-sample majority vote over all trees; iterations
// are independent. The if() guard avoids OMP overhead on tiny batches.
// ---------------------------------------------------------------------------
std::vector<int> RandomForest::predictBatch(
    const std::vector<std::vector<float>> &X) const
{
    int n = static_cast<int>(X.size());
    std::vector<int> preds(n);
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static) if(n >= 64)
#endif
    for (int i = 0; i < n; ++i)
        preds[i] = predict(X[i]);
    return preds;
}
