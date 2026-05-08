// ---------------------------------------------------------------------------
// infer_kernel.cu  —  GPU batch forest inference (Milestone 3)
//
// Grid/block layout:
//   Block : INFER_BLOCK threads (one thread = one sample)
//   Grid  : ceil(n_samples / INFER_BLOCK) blocks
//
// Each thread loops over all trees, walks the tree for its sample, and
// accumulates a vote array in registers.  Majority vote is resolved locally.
//
// Parallelism: over samples (embarrassingly parallel — no inter-sample deps).
// Divergence: threads in the same warp follow different tree paths; this is
// inherent to tree inference and unavoidable.  At n_trees=10, max_depth=7,
// divergence cost is modest compared to the sample-level parallelism gained.
// ---------------------------------------------------------------------------

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "infer_kernel.cuh"   // nvcc looks in the .cu file's own directory first

#define CUDA_CHK(call)                                                      \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess)                                              \
            fprintf(stderr, "CUDA infer %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
    } while (0)

// Max supported classes in register-based vote array.
// 16 covers all UCI datasets (iris=3, wine=3, BC=2, letter=26 → use predictBatch).
// Increase if needed; each +1 costs one extra register per thread.
static const int INFER_CLASSES = 16;
static const int INFER_BLOCK   = 128;

// ---------------------------------------------------------------------------
// forestInferKernel
//
// One thread per sample.  Each thread walks every tree from root to leaf,
// votes for the leaf label, then picks the majority-vote label.
// ---------------------------------------------------------------------------
__global__ void forestInferKernel(
    const FlatNode* __restrict__ d_nodes,    // all trees concatenated
    const int*      __restrict__ d_offsets,  // d_offsets[t] = first node of tree t
    int             n_trees,
    const float*    __restrict__ d_X,        // [n_samples * n_features] row-major
    int             n_samples,
    int             n_features,
    int*            d_preds)                 // [n_samples] output
{
    int sid = (int)(blockIdx.x * blockDim.x) + (int)threadIdx.x;
    if (sid >= n_samples) return;

    const float* x = d_X + (size_t)sid * n_features;

    int votes[INFER_CLASSES];
    for (int c = 0; c < INFER_CLASSES; ++c) votes[c] = 0;

    for (int t = 0; t < n_trees; ++t) {
        const FlatNode* tree = d_nodes + d_offsets[t];
        int idx = 0;

        // Walk the tree.  Depth guard prevents infinite loops on corrupt data.
        for (int depth = 0; depth < 64; ++depth) {
            if (tree[idx].feature_index < 0) {
                // Leaf: record vote.
                int lbl = tree[idx].label;
                if (lbl >= 0 && lbl < INFER_CLASSES) votes[lbl]++;
                break;
            }
            int feat = tree[idx].feature_index;
            int next = (x[feat] <= tree[idx].threshold)
                       ? tree[idx].left_child
                       : tree[idx].right_child;
            if (next < 0) break;  // safety: malformed tree
            idx = next;
        }
    }

    // Majority vote (serial over INFER_CLASSES; small and register-resident).
    int best_lbl = 0, best_cnt = -1;
    for (int c = 0; c < INFER_CLASSES; ++c) {
        if (votes[c] > best_cnt) { best_cnt = votes[c]; best_lbl = c; }
    }
    d_preds[sid] = best_lbl;
}

// ---------------------------------------------------------------------------
// forestInferGPU  —  host wrapper (extern "C" to match .cuh declaration)
// ---------------------------------------------------------------------------
extern "C" void forestInferGPU(
    const FlatNode* h_trees_flat,
    int             total_nodes,
    const int*      h_tree_offsets,
    int             n_trees,
    const float*    h_X,
    int             n_samples,
    int             n_features,
    int*            out_preds)
{
    if (n_samples == 0 || n_trees == 0 || total_nodes == 0) return;

    FlatNode* d_nodes   = nullptr;
    int*      d_offsets = nullptr;
    float*    d_X       = nullptr;
    int*      d_preds   = nullptr;

    size_t nodes_bytes   = (size_t)total_nodes * sizeof(FlatNode);
    size_t offsets_bytes = (size_t)n_trees      * sizeof(int);
    size_t X_bytes       = (size_t)n_samples * n_features * sizeof(float);
    size_t preds_bytes   = (size_t)n_samples * sizeof(int);

    CUDA_CHK(cudaMalloc(&d_nodes,   nodes_bytes));
    CUDA_CHK(cudaMalloc(&d_offsets, offsets_bytes));
    CUDA_CHK(cudaMalloc(&d_X,       X_bytes));
    CUDA_CHK(cudaMalloc(&d_preds,   preds_bytes));

    CUDA_CHK(cudaMemcpy(d_nodes,   h_trees_flat,   nodes_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_offsets, h_tree_offsets, offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_X,       h_X,            X_bytes,       cudaMemcpyHostToDevice));

    dim3 block(INFER_BLOCK);
    dim3 grid((n_samples + INFER_BLOCK - 1) / INFER_BLOCK);
    forestInferKernel<<<grid, block>>>(
        d_nodes, d_offsets, n_trees,
        d_X, n_samples, n_features,
        d_preds);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemcpy(out_preds, d_preds, preds_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_nodes);
    cudaFree(d_offsets);
    cudaFree(d_X);
    cudaFree(d_preds);
}

#endif // USE_CUDA
