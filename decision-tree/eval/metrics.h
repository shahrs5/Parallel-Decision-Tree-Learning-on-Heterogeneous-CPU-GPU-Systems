#pragma once

#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Classification accuracy
//
// Returns the fraction of predictions that match the ground-truth labels.
// Range: [0.0, 1.0].
//
// Milestone 3: extend this file with additional metrics as needed:
//   - confusionMatrix()  — useful for multi-class evaluation
//   - f1Score()          — better than accuracy on imbalanced datasets
//   - oobError()         — out-of-bag error for random forest
// ---------------------------------------------------------------------------
inline float accuracy(const std::vector<int>& y_true,
                      const std::vector<int>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::runtime_error("accuracy: y_true and y_pred size mismatch");
    if (y_true.empty())
        return 0.0f;

    int correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i)
        correct += (y_true[i] == y_pred[i]) ? 1 : 0;

    return static_cast<float>(correct) / static_cast<float>(y_true.size());
}
