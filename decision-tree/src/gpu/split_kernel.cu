// ---------------------------------------------------------------------------
// split_kernel.cu — GPU-accelerated split evaluation (Milestone 2 stub)
//
// This file is intentionally empty for Milestone 1.
// No CUDA toolchain is needed to build the project yet.
// Enable it with:  cmake -DENABLE_CUDA=ON ..
//
// ---------------------------------------------------------------------------
// Milestone 2 implementation plan (Student 2)
// ---------------------------------------------------------------------------
//
// The CPU split-search loop in decision_tree.cpp (marked M2) will call a
// host-side wrapper here.  The overall GPU strategy:
//
//  1. buildHistogramsKernel
//       For each active node and each feature, atomically accumulate
//       (sum, count) per bin into a histogram array.
//       Grid: one block per (node, feature) pair.
//       Block: one thread per sample in the node's subset.
//
//  __global__ void buildHistogramsKernel(
//      const float* __restrict__ d_X,        // [n_samples * n_features] row-major
//      const int*   __restrict__ d_y,         // [n_samples] labels
//      const int*   __restrict__ d_indices,   // [n_active] sample indices for this node
//      int          n_active,                 // samples in this node
//      int          n_features,
//      int          n_bins,                   // number of histogram bins
//      int          n_classes,
//      float*       d_bin_edges,              // [n_features * n_bins] precomputed
//      float*       d_hist_out                // [n_features * n_bins * n_classes]
//  );
//
//  2. findBestSplitKernel
//       Sweep the histogram left-to-right, computing the weighted Gini gain
//       at each bin boundary, and write the best (feature, bin) pair.
//
//  __global__ void findBestSplitKernel(
//      const float* __restrict__ d_hist,      // output of buildHistogramsKernel
//      int          n_features,
//      int          n_bins,
//      int          n_classes,
//      int          n_total,                  // total samples at this node
//      float        parent_gini,
//      int*         d_best_feature,           // output: best feature index
//      float*       d_best_threshold          // output: best threshold value
//  );
//
//  3. Host wrapper (called by DecisionTree::buildNode)
//
//  void findBestSplitGPU(
//      const float* d_X,           // data already resident on GPU
//      const int*   d_y,
//      const int*   d_indices,
//      int          n_active,
//      int          n_features,
//      int          n_bins,
//      int          n_classes,
//      float        parent_gini,
//      int&         out_feature,
//      float&       out_threshold
//  );
//
// Memory strategy (to minimise transfers, per the M2 spec):
//   - d_X and d_y are allocated once in DecisionTree::train() and stay on GPU.
//   - Only d_indices (the active sample list per node) is updated each call.
//   - Only (best_feature, best_threshold) integers are transferred back to CPU.
// ---------------------------------------------------------------------------
