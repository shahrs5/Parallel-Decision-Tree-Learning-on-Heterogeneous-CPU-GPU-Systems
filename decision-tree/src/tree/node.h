#pragma once

// ---------------------------------------------------------------------------
// Node — represents a single node in the decision tree.
//
// Storage layout: array-of-structs (AoS).
// All nodes are stored in a flat std::vector<Node> inside DecisionTree.
// Children are referenced by their index into that vector; -1 means "none".
//
// Why AoS here?
//   - Simple for teammates to read/extend.
//   - Easy to copy to GPU memory as a contiguous blob in Milestone 2.
//   - Can be converted to SoA (struct-of-arrays) for GPU if needed later.
//
// Teammates extending this:
//   - M2 (GPU split finding): you may add a histogram pointer or bin count.
//   - M3 (ensemble):          Node is reused as-is; RandomForest owns a
//                             vector<DecisionTree>, each with its own nodes_.
// ---------------------------------------------------------------------------

struct Node {
    // --- Split information (internal nodes only) ---

    // Index of the feature column to split on.
    // -1 indicates this field is unused (leaf node).
    int   feature_index = -1;

    // Decision threshold: route left if X[sample][feature_index] <= threshold,
    // right otherwise.
    float threshold = 0.0f;

    // --- Impurity ---

    // Gini impurity computed at this node over its training samples.
    // Useful for debugging, visualization, and future pruning (Milestone 3).
    float gini = 0.0f;

    // --- Tree structure ---

    // Indices into DecisionTree::nodes_[].  -1 = no child (this is a leaf).
    int left_child  = -1;
    int right_child = -1;

    // --- Leaf prediction ---

    // Majority class label among training samples at this node.
    // Only used at prediction time when is_leaf == true.
    int  label   = -1;

    // True when this node makes a prediction rather than a split.
    bool is_leaf = false;

    // --- Diagnostics ---

    // Number of training samples routed to this node during training.
    // Useful for analysis and minimum-samples stopping criteria.
    int sample_count = 0;
};

// ---------------------------------------------------------------------------
// FlatNode — compact node layout for fast inference and GPU transfer.
//
// Drops debug fields (gini, sample_count) present in Node, saving ~38% space
// (20 bytes vs 32 bytes per node).  Used by DecisionTree::getPackedNodes()
// and the GPU batch inference kernel in src/gpu/infer_kernel.cu.
//
// feature_index < 0 identifies a leaf (mirrors is_leaf in Node).
// All fields are plain 4-byte types so the struct is directly uploadable to
// GPU memory with no padding or endianness issues.
// ---------------------------------------------------------------------------
struct FlatNode {
    float threshold;      // split threshold (0 if leaf)
    int   feature_index;  // split feature index, -1 if leaf
    int   left_child;     // index into tree's FlatNode array, -1 if leaf
    int   right_child;    // index into tree's FlatNode array, -1 if leaf
    int   label;          // majority-vote class (valid at leaves)
};
