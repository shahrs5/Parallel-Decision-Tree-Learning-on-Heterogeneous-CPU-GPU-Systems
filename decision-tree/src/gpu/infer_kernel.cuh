#pragma once

// ---------------------------------------------------------------------------
// infer_kernel.cuh  —  GPU batch inference interface (Milestone 3)
//
// Evaluates a trained random forest on a batch of samples in parallel.
// Each CUDA thread handles one sample; it walks all trees and majority-votes.
//
// Call flow:
//   1. RandomForest::predictBatchGPU() calls getPackedNodes() on each tree to
//      build a contiguous FlatNode array for the whole forest.
//   2. forestInferGPU() uploads that array + the sample matrix, launches the
//      kernel, and returns predictions.
// ---------------------------------------------------------------------------

#ifdef USE_CUDA

#include "tree/node.h"   // FlatNode

#ifdef __cplusplus
extern "C" {
#endif

// Batch inference across an entire random forest on the GPU.
//
// h_trees_flat : concatenated FlatNode arrays for all trees [total_nodes]
// total_nodes  : total length of h_trees_flat
// h_tree_offsets: start index of each tree in h_trees_flat [n_trees]
// n_trees      : number of trees
// h_X          : host sample matrix [n_samples * n_features], row-major float
// n_samples    : number of samples to classify
// n_features   : features per sample
// out_preds    : output buffer [n_samples], filled with majority-vote labels
void forestInferGPU(
    const FlatNode* h_trees_flat,
    int             total_nodes,
    const int*      h_tree_offsets,
    int             n_trees,
    const float*    h_X,
    int             n_samples,
    int             n_features,
    int*            out_preds);

#ifdef __cplusplus
}
#endif

#endif // USE_CUDA
