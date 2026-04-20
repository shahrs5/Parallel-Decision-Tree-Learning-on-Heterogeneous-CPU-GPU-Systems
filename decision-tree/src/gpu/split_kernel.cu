// ---------------------------------------------------------------------------
// split_kernel.cu  —  GPU-accelerated histogram-based split finding
// Milestone 2, Person 2: GPU Split Computation
//
// Compile with:  cmake -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ..
//   (89 = RTX 40xx Ada Lovelace, e.g. RTX 4060)
//   (86 = RTX 30xx Ampere)
//   (75 = RTX 20xx Turing)
//
// Design:
//   Phase 1 — buildHistogramsKernel
//     Grid : (n_features, 1, 1)   one block-column per feature
//     Block: min(n_active, 256) threads
//     Each thread processes one or more samples and atomically accumulates
//     class counts into per-bin histograms stored in shared + global memory.
//
//   Phase 2 — findBestSplitKernel
//     Grid : (n_features, 1, 1)
//     Block: (n_bins, 1, 1)
//     Each thread sweeps one bin boundary: prefix-sums the histogram columns
//     left-to-right, computes weighted Gini gain, and does a warp-level
//     reduction to find the best (feature, bin) pair across all features.
//
//   Memory transfer strategy (minimise CPU↔GPU traffic):
//     - d_X and d_y uploaded ONCE per tree at train() time, stay on GPU.
//     - Only the index array for the current node (small) is uploaded.
//     - Only (best_feature, best_threshold) — two scalars — come back.
// ---------------------------------------------------------------------------

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>
#include <algorithm>

#include "split_kernel.cuh"

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int MAX_CLASSES = 16;   // max number of distinct class labels
static const int BLOCK_SIZE  = 256;  // threads per block for histogram kernel

// ---------------------------------------------------------------------------
// Kernel 1: buildHistogramsKernel
//
// For each (feature, bin) pair accumulates class counts from the node's
// active sample set using atomic operations on global memory.
//
// d_hist layout: [n_features][n_bins][n_classes]  (row-major)
// d_bin_edges layout: [n_features][n_bins]  — upper edge of each bin
// ---------------------------------------------------------------------------
__global__ void buildHistogramsKernel(
    const float* __restrict__ d_X,         // [n_samples * n_features] row-major
    const int*   __restrict__ d_y,         // [n_samples]
    const int*   __restrict__ d_indices,   // [n_active] indices into d_X/d_y
    int          n_active,
    int          n_samples,
    int          n_features,
    int          n_bins,
    int          n_classes,
    const float* __restrict__ d_bin_edges, // [n_features * n_bins]
    int*         d_hist)                   // [n_features * n_bins * n_classes]
{
    int feat = blockIdx.x;   // one block-column per feature
    if (feat >= n_features) return;

    // Stride through samples assigned to this block.
    for (int tid = threadIdx.x; tid < n_active; tid += blockDim.x) {
        int sample = d_indices[tid];
        float val  = d_X[sample * n_features + feat];
        int   cls  = d_y[sample];
        if (cls < 0 || cls >= n_classes) continue;

        // Binary search for the correct bin.
        const float* edges = d_bin_edges + feat * n_bins;
        int lo = 0, hi = n_bins - 1, bin = n_bins - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (val <= edges[mid]) { bin = mid; hi = mid - 1; }
            else                   { lo  = mid + 1; }
        }

        int hist_idx = (feat * n_bins + bin) * n_classes + cls;
        atomicAdd(&d_hist[hist_idx], 1);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: findBestSplitKernel
//
// Each block handles one feature. Each thread handles one bin boundary.
// A prefix sweep left-to-right computes weighted Gini gain at each split.
// An atomicMin (encoded as int) across all blocks finds the global best.
//
// d_hist layout: [n_features][n_bins][n_classes]
// d_bin_edges layout: [n_features][n_bins]
// ---------------------------------------------------------------------------
__device__ float computeGiniDevice(const int* counts, int total, int n_classes)
{
    if (total == 0) return 0.0f;
    float g = 1.0f;
    float n = (float)total;
    for (int c = 0; c < n_classes; ++c) {
        float p = (float)counts[c] / n;
        g -= p * p;
    }
    return g;
}

__global__ void findBestSplitKernel(
    const int*   __restrict__ d_hist,       // [n_features * n_bins * n_classes]
    const float* __restrict__ d_bin_edges,  // [n_features * n_bins]
    int          n_features,
    int          n_bins,
    int          n_classes,
    int          n_total,
    float        parent_gini,
    int          min_samples_leaf,
    // Outputs — one slot per feature, reduced on CPU.
    float*       d_best_gains,              // [n_features]
    int*         d_best_bins,              // [n_features]
    float*       d_best_thresholds)        // [n_features]
{
    int feat = blockIdx.x;
    if (feat >= n_features) return;

    // Prefix sum buffers in shared memory.
    // left_counts[c] / right_counts[c] accumulated across threads.
    // Each thread needs its own slice → use dynamic shared memory indexed by threadIdx.x.
    // But n_classes is runtime, so use global-memory prefix arrays instead.

    // This kernel is launched with blockDim.x == n_bins.
    int bin = threadIdx.x;
    if (bin >= n_bins) return;

    // Build prefix sums [0..bin] into left_counts.

    int left_counts[MAX_CLASSES]  = {0};
    int right_counts[MAX_CLASSES] = {0};

    const int* feat_hist = d_hist + feat * n_bins * n_classes;

    // total counts for feature
    for (int b = 0; b < n_bins; ++b)
        for (int c = 0; c < n_classes; ++c)
            right_counts[c] += feat_hist[b * n_classes + c];

    // compute left side
    int left_total = 0; 
    for (int b = 0; b <= bin; ++b)
        for (int c = 0; c < n_classes; ++c) {
            int val = feat_hist[b * n_classes + c];
            left_counts[c] += val;
            left_total += val;
        }

    // compute right side from totals
    int right_total = 0;
    for (int c = 0; c < n_classes; ++c) {
        right_counts[c] -= left_counts[c];
        right_total += right_counts[c];
    }

    float gain = -FLT_MAX;
    if (left_total >= min_samples_leaf && right_total >= min_samples_leaf) {
        float left_gini  = computeGiniDevice(left_counts,  left_total,  n_classes);
        float right_gini = computeGiniDevice(right_counts, right_total, n_classes);
        float n = (float)n_total;
        gain = parent_gini
             - ((float)left_total  / n) * left_gini
             - ((float)right_total / n) * right_gini;
    }

    // Warp-level reduction over bins to find the best bin for this feature.
    // We use shared memory to collect per-thread gains.
    extern __shared__ float s_gains[];
    float* s_bins = s_gains + blockDim.x;   // reinterpret second half as int via float bits

    s_gains[bin] = gain;
    __syncthreads();

    // Simple parallel reduction (blockDim.x == n_bins, power-of-2 not required).
    if (bin == 0) {
        float best_gain = -FLT_MAX;
        int   best_bin  = -1;
        for (int b = 0; b < n_bins; ++b) {
            if (s_gains[b] > best_gain) {
                best_gain = s_gains[b];
                best_bin  = b;
            }
        }
        d_best_gains[feat]      = best_gain;
        d_best_bins[feat]       = best_bin;
        d_best_thresholds[feat] = (best_bin >= 0)
                                  ? d_bin_edges[feat * n_bins + best_bin]
                                  : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Helper: compute per-feature bin edges from the active sample set.
// Called on host, result uploaded to GPU.
// ---------------------------------------------------------------------------
static void computeBinEdges(
    const float* h_X,
    const int*   h_indices,
    int          n_active,
    int          n_features,
    int          n_bins,
    float*       h_bin_edges)   // [n_features * n_bins]
{
    for (int f = 0; f < n_features; ++f) {
        // Collect feature values.
        float* vals = new float[n_active];
        for (int i = 0; i < n_active; ++i)
            vals[i] = h_X[h_indices[i] * n_features + f];

        // Partial sort — just need quantile boundaries.
        // Full sort for simplicity (n_active is manageable).
        for (int i = 0; i < n_active - 1; ++i) {
            for (int j = i + 1; j < n_active; ++j) {
                if (vals[j] < vals[i]) { float tmp = vals[i]; vals[i] = vals[j]; vals[j] = tmp; }
            }
        }

        float* edges = h_bin_edges + f * n_bins;
        for (int b = 0; b < n_bins; ++b) {
            // Quantile index (exclusive upper boundary for bin b).
            int q = (int)(((float)(b + 1) / (float)n_bins) * n_active);
            if (q >= n_active) q = n_active - 1;
            edges[b] = vals[q];
        }
        delete[] vals;
    }
}

// ---------------------------------------------------------------------------
// Public host wrapper — called by Person 3's integration in decision_tree.cpp
// ---------------------------------------------------------------------------

extern "C" void uploadDataToGPU(
    const float* h_X,
    const int*   h_y,
    int          n_samples,
    int          n_features,
    float**      d_X_out,
    int**        d_y_out)
{
    size_t X_bytes = (size_t)n_samples * n_features * sizeof(float);
    size_t y_bytes = (size_t)n_samples * sizeof(int);

    CUDA_CHECK(cudaMalloc(d_X_out, X_bytes));
    CUDA_CHECK(cudaMalloc(d_y_out, y_bytes));
    CUDA_CHECK(cudaMemcpy(*d_X_out, h_X, X_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_y_out, h_y, y_bytes, cudaMemcpyHostToDevice));
}

extern "C" void freeGPUData(float* d_X, int* d_y)
{
    if (d_X) cudaFree(d_X);
    if (d_y) cudaFree(d_y);
}

extern "C" void findBestSplitGPU(
    const float* d_X,
    const int*   d_y,
    const float* h_X,   
    const int*   h_indices,
    int          n_active,
    int          n_features,
    int          n_bins,
    int          n_classes,
    float        parent_gini,
    int          min_samples_leaf,
    int&         out_feature,
    float&       out_threshold)
{
    out_feature   = -1;
    out_threshold = 0.0f;

    if (n_active < 2 || n_classes > MAX_CLASSES) return;

    // --- Upload indices ---
    int* d_indices = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices, n_active * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, n_active * sizeof(int), cudaMemcpyHostToDevice));

    // --- Compute bin edges on CPU, upload ---
    // We need the flattened host X for this; reconstruct from GPU is expensive,
    // so the caller should also keep a host copy (see decision_tree.cpp).
    // Here we approximate: bin edges will be computed by the caller and passed
    // through a helper path. For the standalone kernel we compute them here
    // by downloading the needed feature columns (acceptable for correctness).
    //
    // In the full integration (Person 3), h_X_flat is kept as a class member
    // and passed alongside d_X to avoid this download.
    //
    // For now, use a uniform binning approach as a fallback that avoids download.
    // We download only min/max per feature (2 values × n_features << full data).

    float* h_bin_edges = new float[n_features * n_bins];

    // Download the active feature values for bin-edge estimation.
    // Only download n_active × n_features values — much smaller than full X.
    float* h_X_active = new float[(size_t)n_active * n_features];

    for (int i = 0; i < n_active; ++i) {
       int sample = h_indices[i];
       memcpy(
           h_X_active + (size_t)i * n_features,
           h_X + (size_t)sample * n_features,
           n_features * sizeof(float)
       );
    }

   // Compute quantile bin edges.
   for (int f = 0; f < n_features; ++f) {
        float* vals = new float[n_active];
        for (int i = 0; i < n_active; ++i)
           vals[i] = h_X_active[(size_t)i * n_features + f];

        float* edges = h_bin_edges + f * n_bins;

        // copy once so nth_element doesn't corrupt future selections
        float* temp = new float[n_active];
        memcpy(temp, vals, n_active * sizeof(float));

        for (int b = 0; b < n_bins; ++b) {
          int q = (int)(((float)(b + 1) / (float)n_bins) * n_active);
          if (q >= n_active) q = n_active - 1;

          std::nth_element(temp, temp + q, temp + n_active);
          edges[b] = temp[q];
        }

        delete[] temp;
        delete[] vals;
    }
    delete[] h_X_active;

    float* d_bin_edges = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bin_edges, n_features * n_bins * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_bin_edges, h_bin_edges, n_features * n_bins * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_bin_edges;

    // --- Allocate histogram ---
    int* d_hist = nullptr;
    size_t hist_bytes = (size_t)n_features * n_bins * n_classes * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_hist, hist_bytes));
    CUDA_CHECK(cudaMemset(d_hist, 0, hist_bytes));

    // --- Launch Phase 1: buildHistogramsKernel ---
    {
        dim3 grid(n_features, 1, 1);
        dim3 block(min(n_active, BLOCK_SIZE), 1, 1);
        buildHistogramsKernel<<<grid, block>>>(
            d_X, d_y, d_indices,
            n_active, 0 /*n_samples unused*/, n_features,
            n_bins, n_classes,
            d_bin_edges, d_hist);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- Allocate output arrays ---
    float* d_best_gains      = nullptr;
    int*   d_best_bins_arr   = nullptr;
    float* d_best_thresholds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_best_gains,      n_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_bins_arr,   n_features * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_best_thresholds, n_features * sizeof(float)));

    // --- Launch Phase 2: findBestSplitKernel ---
    {
        int bins_block = min(n_bins, 1024);
        size_t shared_bytes = (size_t)bins_block * 2 * sizeof(float);
        dim3 grid(n_features, 1, 1);
        dim3 block(bins_block, 1, 1);
        findBestSplitKernel<<<grid, block, shared_bytes>>>(
            d_hist, d_bin_edges,
            n_features, n_bins, n_classes,
            n_active, parent_gini, min_samples_leaf,
            d_best_gains, d_best_bins_arr, d_best_thresholds);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- Copy results back (only n_features scalars) ---
    float* h_gains      = new float[n_features];
    int*   h_bins       = new int[n_features];
    float* h_thresholds = new float[n_features];
    CUDA_CHECK(cudaMemcpy(h_gains,      d_best_gains,      n_features * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bins,       d_best_bins_arr,   n_features * sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_thresholds, d_best_thresholds, n_features * sizeof(float), cudaMemcpyDeviceToHost));

    // --- CPU-side reduction: pick best feature ---
    float best_gain = -1e30f;
    for (int f = 0; f < n_features; ++f) {
        if (h_gains[f] > best_gain && h_bins[f] >= 0) {
            best_gain     = h_gains[f];
            out_feature   = f;
            out_threshold = h_thresholds[f];
        }
    }

    delete[] h_gains;
    delete[] h_bins;
    delete[] h_thresholds;

    cudaFree(d_indices);
    cudaFree(d_bin_edges);
    cudaFree(d_hist);
    cudaFree(d_best_gains);
    cudaFree(d_best_bins_arr);
    cudaFree(d_best_thresholds);
}

#endif // USE_CUDA
