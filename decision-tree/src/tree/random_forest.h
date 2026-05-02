#pragma once

#include <memory>
#include <vector>

#include "decision_tree.h"

// ---------------------------------------------------------------------------
// RandomForest — Milestone 3 ensemble classifier.
//
// Owns a collection of DecisionTree instances, each trained on an independent
// bootstrap sample of the training data. Prediction is by majority vote.
//
// Step 1 of M3 (current): serial training, no feature subsampling, serial
// batch prediction. Subsequent M3 steps will add:
//   - per-split feature subsampling (sqrt(F) by default)
//   - tree-level OpenMP parallelism in train()
//   - parallel-for batch inference
//
// Trees are stored as unique_ptr because DecisionTree owns raw CUDA pointers
// in its destructor; aliasing them via vector reallocation/copy would cause
// a double-free.
// ---------------------------------------------------------------------------
class RandomForest
{
public:
    // feature_subsample: per-split feature pool size. Sentinel values:
    //   -1 => sqrt(n_features)  (standard random-forest default)
    //    0 => use all features  (degenerates to bagging)
    //   >0 => use this many features per split
    RandomForest(int n_trees,
                 int max_depth,
                 int min_samples_leaf,
                 int feature_subsample = -1,
                 unsigned seed = 42);

    void train(const std::vector<std::vector<float>> &X,
               const std::vector<int> &y);

    int predict(const std::vector<float> &sample) const;

    std::vector<int> predictBatch(
        const std::vector<std::vector<float>> &X) const;

    int n_trees() const { return static_cast<int>(trees_.size()); }

private:
    int      n_trees_;
    int      max_depth_;
    int      min_samples_leaf_;
    int      feature_subsample_;   // -1 = sqrt(F), 0 = all, >0 = explicit
    unsigned seed_;

    std::vector<std::unique_ptr<DecisionTree>> trees_;

    // Draws n samples with replacement from (X, y) into (X_boot, y_boot).
    static void bootstrapSample(
        const std::vector<std::vector<float>> &X,
        const std::vector<int>                &y,
        unsigned                               rng_seed,
        std::vector<std::vector<float>>       &X_boot,
        std::vector<int>                      &y_boot);
};
