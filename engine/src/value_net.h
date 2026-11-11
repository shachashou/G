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
// Ticket 11 keeps this width for the incremental path too: every add has an
// exact inverse, and no saturation is ever applied to the accumulator itself.
// The bound is |int32| + 225 * 32768 < 2^32, far inside int64.
using ValueAccumulator = std::array<int64_t, kValueAcc>;

// Ticket 11: how the search evaluator turns a board into z.
//
//   Incremental + Avx2    production: first layer updated by +/- one feature
//                         row, 1024->64 layer shortened with AVX2.
//   Incremental + Scalar  same arithmetic without SIMD (A/B and fallback).
//   FullRebuild + Scalar  the ticket-07 reference: rebuild all 450 rows and
//                         run the scalar forward on every evaluation.
//
// All three paths are required to be bit-identical. The reference mode is
// selectable at runtime so a numeric or performance regression can always be
// compared against the path that produced the frozen contract.
enum class ValueBackend { Scalar, Avx2 };
enum class ValueAccumulatorMode { Incremental, FullRebuild };

const char* valueBackendName(ValueBackend backend);
const char* valueAccumulatorModeName(ValueAccumulatorMode mode);
// "avx2" | "scalar" | "reference" (full rebuild + scalar forward).
bool valueEvalModeFromName(const std::string& name, ValueBackend& backend,
                           ValueAccumulatorMode& accumulator);
// Stable identity string, e.g. "incremental-avx2".
std::string valueEvalModeName(ValueBackend backend,
                              ValueAccumulatorMode accumulator);

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

  // Ticket 11 runtime configuration. Defaults match production (incremental,
  // AVX2); `reference` restores the ticket-07 full-rebuild scalar path.
  void setBackend(ValueBackend backend) { backend_ = backend; }
  ValueBackend backend() const { return backend_; }
  void setAccumulatorMode(ValueAccumulatorMode mode) {
    accumulatorMode_ = mode;
  }
  ValueAccumulatorMode accumulatorMode() const { return accumulatorMode_; }
  std::string evalModeName() const {
    return valueEvalModeName(backend_, accumulatorMode_);
  }

  // Reference scalar forward from a sparse feature-index set (<= 225
  // ascending indices). Throws ValueNetError on out-of-range indices.
  // Always scalar: this is the frozen numeric reference.
  int64_t inferSparse(const std::vector<int>& features) const;

  // Full first-layer rebuild from the board (the long-term numeric reference
  // for the incremental update), then the scalar forward.
  void rebuildAccumulator(const Board& board, ValueAccumulator& acc) const;

  // First-layer incremental update: adds the feature row for `to`, subtracts
  // the row for the stone currently on `move` (board is in its pre-change
  // state, like Evaluator::update). Invalid requests throw ValueNetError
  // BEFORE either row is touched, so the accumulator is unchanged.
  void updateAccumulator(const Board& board, int move, Cell to,
                         ValueAccumulator& acc) const;

  // Forward from an accumulator with the selected backend.
  int64_t inferAccumulator(const ValueAccumulator& acc) const;
  // Explicit scalar reference forward (used by evaluateBoard and the probe).
  int64_t inferAccumulatorScalar(const ValueAccumulator& acc) const;
  int64_t inferAccumulatorAvx2(const ValueAccumulator& acc) const;

  // Full rebuild + scalar forward. Independent of the selected backend; this
  // is the long-term numeric reference the optimized paths are compared to.
  int64_t evaluateBoard(const Board& board) const;

  // Fixed-point black-absolute z -> side-to-move search score (negamax sign).
  int sideToMoveScore(int64_t z, Color side) const;

  // Integer score from z: round(z * kValueScoreScale / 2^ea), clamped.
  static int zToSearchScore(int64_t z, int ea);

  // tanh(z / 2^ea) in black-absolute space; tools only, never the search.
  static double valueFromZ(int64_t z, int ea);

 private:
  void addFeature(int feature, int sign, ValueAccumulator& acc) const;
  void addFeatureAvx2(int feature, int sign, ValueAccumulator& acc) const;
  // Shared 64->32->1 half once z1 is known, so scalar and AVX2 cannot drift.
  int64_t finishForward(const std::array<int64_t, kValueH1>& z1) const;

  std::vector<int16_t> w0_, w1_, w2_, w3_;
  std::vector<int32_t> b0_, b1_, b2_, b3_;
  ValueModelInfo info_;
  ValueBackend backend_ = ValueBackend::Avx2;
  ValueAccumulatorMode accumulatorMode_ = ValueAccumulatorMode::Incremental;
  bool loaded_ = false;
};

}  // namespace gomoku
