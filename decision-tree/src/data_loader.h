#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// loadCSV — loads a numerical CSV file into feature matrix X and labels y.
//
// Expected format
//   - Optional single header row.  Detected automatically: if the first
//     token of the first row cannot be parsed as a float, the row is skipped.
//   - Every subsequent row: N numeric columns followed by one integer label.
//   - Features → X  (vector of rows, each row is a vector<float>)
//   - Last column → y  (vector<int>)
//
// Returns: number of samples loaded.
// Throws:  std::runtime_error on file-open failure or malformed data.
//
// Teammates: X is row-major [n_samples][n_features].
//   In Milestone 2, flatten to a 1-D float* with row-major order before
//   copying to the GPU:  cudaMemcpy(d_X, X[0].data(), ...)  — but only
//   after ensuring the data is in a contiguous 2-D array.  Consider
//   switching to std::vector<float> with manual indexing at that point.
// ---------------------------------------------------------------------------
inline int loadCSV(const std::string &filepath,
                   std::vector<std::vector<float>> &X,
                   std::vector<int> &y)
{
    std::ifstream file(filepath);
    if (!file.is_open())
        throw std::runtime_error("loadCSV: cannot open file: " + filepath);

    X.clear();
    y.clear();

    std::string line;
    bool first_row = true;
    std::size_t expected_cols = 0;
    bool expected_cols_set = false;

    while (std::getline(file, line))
    {
        // Strip Windows-style carriage return (\r\n → \n).
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line.empty())
            continue;

        // Tokenise by comma.
        std::vector<std::string> tokens;
        {
            std::stringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, ','))
                tokens.push_back(tok);
        }

        if (tokens.empty())
            continue;

        // Auto-detect and skip a header row: if the first token cannot be
        // parsed as a floating-point number, treat it as a header.
        if (first_row)
        {
            first_row = false;
            try
            {
                std::stof(tokens[0]);
            }
            catch (...)
            {
                continue; // header row — skip it
            }
        }

        if (tokens.size() < 2)
            throw std::runtime_error("loadCSV: row has fewer than 2 columns");
        if (!expected_cols_set)
        {
            expected_cols = tokens.size();
            expected_cols_set = true;
        }
        else if (tokens.size() != expected_cols)
        {
            throw std::runtime_error("loadCSV: inconsistent column count across rows");
        }
        // All tokens except the last → features.
        std::vector<float> row;
        row.reserve(tokens.size() - 1);
        for (std::size_t i = 0; i + 1 < tokens.size(); ++i)
            try
            {
                row.push_back(std::stof(tokens[i]));
            }
            catch (...)
            {
                throw std::runtime_error("loadCSV: invalid numeric feature value");
            }

        // Last token → integer label.
        // stof handles both "0" and "0.0" gracefully.
        int label = static_cast<int>(std::stof(tokens.back()));

        X.push_back(std::move(row));
        y.push_back(label);
    }

    return static_cast<int>(X.size());
}
