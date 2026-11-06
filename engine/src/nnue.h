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

// v3 body: W* float16 (2 bytes), b* float32 (4 bytes).
constexpr size_t kNnueW0BytesV3 = static_cast<size_t>(kNnueFeatures) * kNnueAcc * 2;
constexpr size_t kNnueB0BytesV3 = kNnueAcc * 4;
constexpr size_t kNnueW1BytesV3 = static_cast<size_t>(kNnueAcc) * kNnueL1 * 2;
constexpr size_t kNnueB1BytesV3 = kNnueL1 * 4;
constexpr size_t kNnueW2BytesV3 = kNnueHeads * kNnueL1 * 2;
constexpr size_t kNnueB2BytesV3 = kNnueHeads * 4;
constexpr size_t kNnueBodySizeV3 =
    kNnueW0BytesV3 + kNnueB0BytesV3 + kNnueW1BytesV3 + kNnueB1BytesV3 +
    kNnueW2BytesV3 + kNnueB2BytesV3;  // 956544

using NnueAccumulator = std::array<int32_t, kNnueAcc>;

// Ticket 21: v3 (float16 weights) uses a float32 accumulator instead of the
// exact int32 accumulator. The two accumulator types are never mixed: the
// network's format version selects which one the Evaluator maintains.
using NnueFloatAccumulator = std::array<float, kNnueAcc>;

// Ticket 10: three evaluation backends are reachable from the CLI:
//   handcrafted (no --nnue), scalar NNUE, AVX2 NNUE.
// Both NNUE backends share the exact same int32 accumulator representation,
// so switching backends never changes the output by even one unit.
enum class NnueBackend { Scalar, Avx2 };

const char* nnueBackendName(NnueBackend backend);
bool nnueBackendFromName(const std::string& name, NnueBackend& out);

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
  int formatVersion() const { return formatVersion_; }
  int outputGain() const { return outputGain_; }

  // Defaults to AVX2 (the production backend); scalar stays available for
  // bit-exact A/B comparison and as a fallback.
  void setBackend(NnueBackend backend) { backend_ = backend; }
  NnueBackend backend() const { return backend_; }

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

  // Float32 accumulator path (v3 float16 weights). Scalar only for now.
  void initAccumulator(const Board& board, NnueFloatAccumulator& acc) const;
  void updateAccumulator(const Board& board, int move, Cell to,
                         NnueFloatAccumulator& acc) const;
  int inferAccumulator(const NnueFloatAccumulator& acc, int stm) const;

 private:
  void addFeature(int index, int sign, NnueAccumulator& acc) const;
  void addFeatureAvx2(int index, int sign, NnueAccumulator& acc) const;
  void computeL1(const int32_t* acc,
                 std::array<int32_t, kNnueL1>& z) const;
  int inferSum(const int32_t* acc, int stm) const;

  void addFeatureFloat(int index, int sign, NnueFloatAccumulator& acc) const;
  void computeL1Float(const float* acc, std::array<float, kNnueL1>& z) const;
  int inferSumFloat(const float* acc, int stm) const;

  std::vector<int8_t> w0_, w1_, w2_;
  // W1 with each 15-wide row padded to 16 bytes so the AVX2 L1 loop can load
  // two aligned 8-byte groups without ever touching the next row / the end.
  std::vector<int8_t> w1Padded_;
  std::vector<int16_t> b0_, b1_, b2_;
  // v3 float weights (converted from half to float at load).
  std::vector<float> w0f_, w1f_, w2f_, b0f_, b1f_, b2f_;
  int qa_ = 127;
  int qb_ = 64;
  int scoreScale_ = 400;
  int formatVersion_ = 1;
  int outputGain_ = 1;
  int weightClamp_ = 64;
  NnueBackend backend_ = NnueBackend::Avx2;
  std::string path_;
  bool loaded_ = false;
};

}  // namespace gomoku
