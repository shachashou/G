#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "board.h"

namespace gomoku {

// Ticket 07: deployment of the new value network.
//
// The numeric contract is trainer/value_format.py (`GMVALUE1`); the feature
// contract is trainer/features.py (450 absolute-colour features). This module
// is deliberately *only* the reference path: a full sparse scalar forward and
// a full first-layer rebuild. Ticket 11 adds the incremental accumulator and
// SIMD, and compares against these two functions.
//
//   features  = 450 = 2 * 225 cells  (black plane 0..224, white plane 225..449)
//   acc[j]    = b0[j] + sum(W0[i][j] for every occupied feature i)   (int64)
//   a0[j]     = clamp(acc[j], 0, act_max_int)
//   z1[k]     = round_div(sum_j a0[j]*W1[j][k], S) + b1[k];  a1 = clamp(...,0,AM)
//   z2[m]     = round_div(sum_k a1[k]*W2[k][m], S) + b2[m];  a2 = clamp(...,0,AM)
//   z         = round_div(sum_m a2[m]*W3[m], S) + b3
//   value     = tanh(z / S)                     (BLACK-absolute, [-1, 1])
//
// The side to move is not an input: negamax converts the sign. The search
// consumes a monotonic integer rescaling of the pre-tanh black-absolute z
// (see zToSearchScore), never the raw tanh value, so the inference reference
// stays bit-exact and integer-only.
constexpr int kValueFeatures = 450;
constexpr int kValueCells = 225;
constexpr int kValueAcc = 1024;
constexpr int kValueH1 = 64;
constexpr int kValueH2 = 32;
constexpr int kValueOut = 1;
constexpr int kValueHeaderSize = 64;
constexpr int kValueBodyBytes =
    kValueFeatures * kValueAcc * 2 + kValueAcc * 4 +
    kValueAcc * kValueH1 * 2 + kValueH1 * 4 +
    kValueH1 * kValueH2 * 2 + kValueH2 * 4 +
    kValueH2 * kValueOut * 2 + kValueOut * 4;

// Search score scale: 1000 units per unit of pre-tanh z. tanh is monotonic,
// so move selection is unchanged; the clamp keeps every network score far
// below the mate band so terminal / proven-VCF scores always dominate.
constexpr int kValueScoreScale = 1000;
constexpr int kValueScoreLimit = 100000;

// Fixed-point first layer of the reference path. int64 because the Python
// reference sums in int64: bias values are int32 and a position can add up to
// 225 int16 rows, so a narrower accumulator could not be reverted exactly.
using ValueAccumulator = std::array<int64_t, kValueAcc>;

class ValueNetError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Human/tool-facing identity of the loaded file; also echoed by ABOUT/think so
// the GUI can show which model is actually playing.
struct ValueModelInfo {
  std::string path;
  std::string model_id;
  std::string rule;
  std::string feature_spec;
  std::string value_contract;
  std::string created_at;
  std::vector<std::string> label_versions;
  int64_t training_steps = -1;
  int64_t file_bytes = 0;
  int ea = 9;
  int act_max_int = 0;
  uint32_t fnv1a = 0;

  std::string arch() const;  // "450-1024-64-32-1"
};

class ValueNetwork {
 public:
  // Strict load: any malformed / incompatible file throws ValueNetError.
  // Never prints to stdout (the protocol streams use it).
  void load(const std::string& path);

  bool loaded() const { return loaded_; }
  const ValueModelInfo& info() const { return info_; }
  const std::string& path() const { return info_.path; }
  int ea() const { return info_.ea; }
  int scale() const { return 1 << info_.ea; }
  int actMaxInt() const { return info_.act_max_int; }

  // Reference scalar forward from a sparse feature-index set (<= 225
  // ascending indices). Throws ValueNetError on out-of-range indices.
  int64_t inferSparse(const std::vector<int>& features) const;

  // Full first-layer rebuild from the board (the long-term numeric reference
  // for ticket 11's incremental update), then the scalar forward.
  void rebuildAccumulator(const Board& board, ValueAccumulator& acc) const;
  int64_t inferAccumulator(const ValueAccumulator& acc) const;

  // Full rebuild + forward. This is the only path used by the evaluator until
  // ticket 11 lands; it is intentionally not incremental.
  int64_t evaluateBoard(const Board& board) const;

  // Fixed-point black-absolute z -> side-to-move search score (negamax sign).
  int sideToMoveScore(int64_t z, Color side) const;

  // Integer score from z: round(z * kValueScoreScale / 2^ea), clamped.
  static int zToSearchScore(int64_t z, int ea);

  // tanh(z / 2^ea) in black-absolute space; tools only, never the search.
  static double valueFromZ(int64_t z, int ea);

 private:
  std::vector<int16_t> w0_, w1_, w2_, w3_;
  std::vector<int32_t> b0_, b1_, b2_, b3_;
  ValueModelInfo info_;
  bool loaded_ = false;
};

}  // namespace gomoku
