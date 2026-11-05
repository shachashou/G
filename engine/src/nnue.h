#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "board.h"

namespace gomoku {

// Ticket 09: scalar (non-SIMD) NNUE inference.
//
// The numeric contract is trainer/weights_format.py; the feature-index contract
// is trainer/features.py. Both are replayed bit-for-bit here:
//
//   cell = y * 15 + x, black -> [0, 225), white -> [225, 450)
//   acc[j] = clamp_i16(b0[j] + sum(W0[i, j] for i in sparse_indices))
//   a[j]   = clamp(acc[j], 0, QA)
//   z[k]   = sum_j a[j] * W1[j, k]
//   zq[k]  = round_div(z[k], QA) + b1[k]
//   a2[k]  = clamp(zq[k], 0, QB)
//   y      = sum_k a2[k] * W2[h, k]       h = 0 (black to move), 1 (white)
//   eval   = round_div(y, QB) + b2[h]
//
// The accumulator is kept as int32 (exact sum, no intermediate saturation) so
// incremental place/undo updates match a full recompute: clamp_i16 is applied
// once, at inference time, exactly like the Python reference.
constexpr int kNnueFeatures = 2 * kCells;  // 450
constexpr int kNnueAcc = 1024;
constexpr int kNnueL1 = 15;
constexpr int kNnueHeads = 2;
constexpr int kNnueHeaderSize = 64;
constexpr int kNnueMagicSize = 8;
constexpr int kNnueHeaderFields = 11;  // uint32 fields after the magic
constexpr int kNnueHeaderPrefix = kNnueMagicSize + kNnueHeaderFields * 4;  // 52

constexpr size_t kNnueW0Len = static_cast<size_t>(kNnueFeatures) * kNnueAcc;
constexpr size_t kNnueB0Len = kNnueAcc;
constexpr size_t kNnueW1Len = static_cast<size_t>(kNnueAcc) * kNnueL1;
constexpr size_t kNnueB1Len = kNnueL1;
constexpr size_t kNnueW2Len = kNnueHeads * kNnueL1;
constexpr size_t kNnueB2Len = kNnueHeads;
constexpr size_t kNnueBodySize =
    kNnueW0Len + 2 * kNnueB0Len + kNnueW1Len + 2 * kNnueB1Len +
    kNnueW2Len + 2 * kNnueB2Len;
constexpr size_t kNnueFileSize = kNnueHeaderSize + kNnueBodySize;  // 478336

using NnueAccumulator = std::array<int32_t, kNnueAcc>;

class NnueError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Absolute-colour feature index; `stone` must be Black or White.
constexpr int nnueFeatureIndex(int cell, Cell stone) {
  return stone == Cell::Black ? cell : kCells + cell;
}

// Sorted sparse index set for a position (<= 225 entries).
std::vector<int> nnueSparseIndices(const Board& board);

// Round n/d half away from zero (d > 0). Plain C++ division truncates toward
// zero and MUST NOT be used to replace this.
int64_t roundDiv(int64_t n, int d);

class NnueNetwork {
 public:
  bool loaded() const { return loaded_; }
  const std::string& path() const { return path_; }
  int qa() const { return qa_; }
  int qb() const { return qb_; }
  int scoreScale() const { return scoreScale_; }

  // Loads and fully validates a `GMNNUE01` v1 file. Throws NnueError with a
  // descriptive message; never writes to stdout (Gomocup protocol stream).
  void load(const std::string& path);

  // Full sparse scalar forward: the reference path, bit-identical to
  // weights_format.infer_int() for the same weights / indices / stm.
  int infer(const std::vector<int>& sparse, int stm) const;

  // Incremental (still scalar) path used by Evaluator during search.
  void initAccumulator(const Board& board, NnueAccumulator& acc) const;
  void updateAccumulator(const Board& board, int move, Cell to,
                         NnueAccumulator& acc) const;
  int inferAccumulator(const NnueAccumulator& acc, int stm) const;

 private:
  void addFeature(int index, int sign, NnueAccumulator& acc) const;
  int inferSum(const int32_t* acc, int stm) const;

  std::vector<int8_t> w0_, w1_, w2_;
  std::vector<int16_t> b0_, b1_, b2_;
  int qa_ = 127;
  int qb_ = 64;
  int scoreScale_ = 400;
  int weightClamp_ = 64;
  std::string path_;
  bool loaded_ = false;
};

}  // namespace gomoku
