#pragma once

// GPU split-finding interface (Milestone 2 — Person 2).
// Called by Person 3's integration layer in decision_tree.cpp when USE_CUDA=1.
//
// Memory contract:
//   d_X, d_y are allocated once at train() start and remain GPU-resident.
//   Only d_indices is re-uploaded per node; only (feature, threshold) come back.

#ifdef USE_CUDA

#ifdef __cplusplus
extern "C" {
#endif

// One-time upload: flatten X to row-major float* and upload X, y to GPU.
// Returns pointers to GPU memory that the caller must free with freeGPUData().
void uploadDataToGPU(
    const float* h_X,      // [n_samples * n_features] row-major
    const int*   h_y,      // [n_samples]
    int          n_samples,
    int          n_features,
    float**      d_X_out,
    int**        d_y_out);

void freeGPUData(float* d_X, int* d_y);

// Find the best (feature, threshold) split for a node's sample subset.
// d_X / d_y remain on GPU; only indices need to be sent each call.
void findBestSplitGPU(
    const float* d_X,
    const int*   d_y,
    const float* h_X, 
    const int*   h_indices,   // CPU array — copied to GPU inside this call
    int          n_active,
    int          n_features,
    int          n_bins,
    int          n_classes,
    float        parent_gini,
    int          min_samples_leaf,
    int&         out_feature,
    float&       out_threshold);

#ifdef __cplusplus
}
#endif

#endif // USE_CUDA
