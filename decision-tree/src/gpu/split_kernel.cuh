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

// Initialise the CUDA context so subsequent timed calls are not penalised
// by the one-time driver load (~100-150 ms on Windows).  Call once before
// any benchmark timer starts.
void warmupCUDA();

// Force batch mode regardless of available GPU memory (for testing).
void setForceBatchMode(bool force);

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

// Per-training-run GPU timing stats.
struct GPUCallStats {
    float kernel_ms;   // time inside both CUDA kernels
    float total_ms;    // total findBestSplitGPU wall time (incl. transfers)
    int   n_calls;     // number of nodes processed on GPU
};
void resetGPUCallStats();
void getGPUCallStats(GPUCallStats* out);

// Find the best (feature, threshold) split for a node's sample subset.
// d_X / d_y remain on GPU; only indices need to be sent each call.
//
// feature_subset / n_subset (Milestone 3): when n_subset > 0 the host-side
// argmax over per-feature gains only considers the listed features. The
// kernel still computes histograms for ALL features (TODO: optimise by
// restricting kernel work to subset features).
//   feature_subset = nullptr OR n_subset == 0  ->  consider all features.
void findBestSplitGPU(
    const float* d_X,         // GPU feature matrix (null if batch mode)
    const int*   d_y,         // GPU labels (always valid)
    const float* h_X,         // host flat feature matrix
    const int*   h_y,         // host labels (used in batch mode when d_X==null)
    const int*   h_indices,   // CPU array — copied to GPU inside this call
    int          n_active,
    int          n_features,
    int          n_bins,
    int          n_classes,
    float        parent_gini,
    int          min_samples_leaf,
    const int*   feature_subset,  // M3: feature index list (sorted) or nullptr
    int          n_subset,        // M3: subset size (0 = all features)
    int&         out_feature,
    float&       out_threshold);

#ifdef __cplusplus
}
#endif

#endif // USE_CUDA
