#include "nnue.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace gomoku {

namespace {

constexpr char kMagic[kNnueMagicSize] = {'G', 'M', 'N', 'N', 'U', 'E', '0', '1'};
constexpr uint32_t kFnvOffset = 0x811C9DC5u;
constexpr uint32_t kFnvPrime = 0x01000193u;
constexpr int32_t kInt16Min = -32768;
constexpr int32_t kInt16Max = 32767;

uint32_t readU32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

int16_t readI16(const uint8_t* p) {
  const uint32_t raw = static_cast<uint32_t>(p[0]) |
                       (static_cast<uint32_t>(p[1]) << 8);
  const int32_t value = raw >= 0x8000u ? static_cast<int32_t>(raw) - 0x10000
                                       : static_cast<int32_t>(raw);
  return static_cast<int16_t>(value);
}

uint32_t fnv1a(const uint8_t* data, size_t size) {
  uint32_t hash = kFnvOffset;
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ static_cast<uint32_t>(data[i])) * kFnvPrime;
  }
  return hash;
}

int32_t clampInt16(int32_t value) {
  return std::clamp(value, kInt16Min, kInt16Max);
}

}  // namespace

std::vector<int> nnueSparseIndices(const Board& board) {
  std::vector<int> indices;
  indices.reserve(kCells);
  const auto& cells = board.rawCells();
  for (int m = 0; m < kCells; ++m) {
    if (cells[m] != Cell::Empty) {
      indices.push_back(nnueFeatureIndex(m, cells[m]));
    }
  }
  // Black indices are < 225, white indices are >= 225, so a cell-order scan
  // already yields an ascending set. Keep the sort explicit for the contract.
  std::sort(indices.begin(), indices.end());
  return indices;
}

int64_t roundDiv(int64_t n, int d) {
  if (d <= 0) throw NnueError("roundDiv: divisor must be positive");
  const int64_t sign = n < 0 ? -1 : 1;
  const int64_t magnitude = n < 0 ? -n : n;
  return sign * ((2 * magnitude + d) / (2 * static_cast<int64_t>(d)));
}

void NnueNetwork::load(const std::string& path) {
  const std::string prefix = "NNUE weights '" + path + "': ";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw NnueError(prefix + "cannot open file");

  in.seekg(0, std::ios::end);
  const std::streamoff length = in.tellg();
  if (length < 0) throw NnueError(prefix + "cannot determine file size");
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> blob(static_cast<size_t>(length));
  if (!blob.empty()) {
    in.read(reinterpret_cast<char*>(blob.data()),
            static_cast<std::streamsize>(blob.size()));
  }
  if (static_cast<size_t>(in.gcount()) != blob.size()) {
    throw NnueError(prefix + "short read");
  }
  in.close();

  if (blob.size() < static_cast<size_t>(kNnueHeaderSize)) {
    throw NnueError(prefix + "truncated header (need " +
                    std::to_string(kNnueHeaderSize) + " bytes, got " +
                    std::to_string(blob.size()) + ")");
  }
  if (std::memcmp(blob.data(), kMagic, kNnueMagicSize) != 0) {
    throw NnueError(prefix + "bad magic (expected GMNNUE01)");
  }

  const uint32_t version = readU32(&blob[8]);
  const uint32_t headerSize = readU32(&blob[12]);
  const uint32_t nFeatures = readU32(&blob[16]);
  const uint32_t nAcc = readU32(&blob[20]);
  const uint32_t nL1 = readU32(&blob[24]);
  const uint32_t nHeads = readU32(&blob[28]);
  const uint32_t qa = readU32(&blob[32]);
  const uint32_t qb = readU32(&blob[36]);
  const uint32_t scoreScale = readU32(&blob[40]);
  const uint32_t weightClamp = readU32(&blob[44]);
  const uint32_t checksum = readU32(&blob[48]);

  if (version != 1) {
    throw NnueError(prefix + "unsupported version " + std::to_string(version));
  }
  if (headerSize != kNnueHeaderSize) {
    throw NnueError(prefix + "bad header size " + std::to_string(headerSize));
  }
  if (nFeatures != kNnueFeatures || nAcc != kNnueAcc || nL1 != kNnueL1 ||
      nHeads != kNnueHeads) {
    throw NnueError(prefix + "unsupported network dimensions " +
                    std::to_string(nFeatures) + "/" + std::to_string(nAcc) +
                    "/" + std::to_string(nL1) + "/" + std::to_string(nHeads));
  }
  if (qa == 0 || qb == 0 || scoreScale == 0 || weightClamp == 0 ||
      qa > 0x7FFFFFFFu || qb > 0x7FFFFFFFu || weightClamp > 127) {
    throw NnueError(prefix + "invalid quantisation constants in header");
  }
  for (int i = kNnueHeaderPrefix; i < kNnueHeaderSize; ++i) {
    if (blob[i] != 0) {
      throw NnueError(prefix + "non-zero reserved header bytes");
    }
  }

  const uint8_t* body = blob.data() + kNnueHeaderSize;
  const size_t bodySize = blob.size() - kNnueHeaderSize;
  if (bodySize != kNnueBodySize) {
    throw NnueError(prefix + "body length " + std::to_string(bodySize) +
                    " != " + std::to_string(kNnueBodySize));
  }
  if (fnv1a(body, bodySize) != checksum) {
    throw NnueError(prefix + "checksum mismatch");
  }

  std::vector<int8_t> w0(kNnueW0Len), w1(kNnueW1Len), w2(kNnueW2Len);
  std::vector<int16_t> b0(kNnueB0Len), b1(kNnueB1Len), b2(kNnueB2Len);
  size_t cursor = 0;
  for (size_t i = 0; i < w0.size(); ++i) {
    w0[i] = static_cast<int8_t>(body[cursor + i]);
  }
  cursor += kNnueW0Len;
  for (size_t i = 0; i < b0.size(); ++i) {
    b0[i] = readI16(&body[cursor + 2 * i]);
  }
  cursor += 2 * kNnueB0Len;
  for (size_t i = 0; i < w1.size(); ++i) {
    w1[i] = static_cast<int8_t>(body[cursor + i]);
  }
  cursor += kNnueW1Len;
  for (size_t i = 0; i < b1.size(); ++i) {
    b1[i] = readI16(&body[cursor + 2 * i]);
  }
  cursor += 2 * kNnueB1Len;
  for (size_t i = 0; i < w2.size(); ++i) {
    w2[i] = static_cast<int8_t>(body[cursor + i]);
  }
  cursor += kNnueW2Len;
  for (size_t i = 0; i < b2.size(); ++i) {
    b2[i] = readI16(&body[cursor + 2 * i]);
  }
  cursor += 2 * kNnueB2Len;
  if (cursor != kNnueBodySize) throw NnueError(prefix + "internal body cursor error");

  const int clamp = static_cast<int>(weightClamp);
  const auto checkWeights = [&](const std::vector<int8_t>& v, const char* name) {
    for (int8_t value : v) {
      const int magnitude = value < 0 ? -static_cast<int>(value)
                                      : static_cast<int>(value);
      if (magnitude > clamp) {
        throw NnueError(prefix + std::string(name) + " weight " +
                        std::to_string(static_cast<int>(value)) +
                        " exceeds weight_clamp " + std::to_string(clamp));
      }
    }
  };
  checkWeights(w0, "W0");
  checkWeights(w1, "W1");
  checkWeights(w2, "W2");

  w0_ = std::move(w0);
  b0_ = std::move(b0);
  w1_ = std::move(w1);
  b1_ = std::move(b1);
  w2_ = std::move(w2);
  b2_ = std::move(b2);
  qa_ = static_cast<int>(qa);
  qb_ = static_cast<int>(qb);
  scoreScale_ = static_cast<int>(scoreScale);
  weightClamp_ = clamp;
  path_ = path;
  loaded_ = true;
}

void NnueNetwork::addFeature(int index, int sign, NnueAccumulator& acc) const {
  const int8_t* row = &w0_[static_cast<size_t>(index) * kNnueAcc];
  for (int j = 0; j < kNnueAcc; ++j) {
    acc[j] += sign * static_cast<int32_t>(row[j]);
  }
}

void NnueNetwork::initAccumulator(const Board& board,
                                  NnueAccumulator& acc) const {
  if (!loaded_) throw NnueError("NNUE accumulator: weights not loaded");
  for (int j = 0; j < kNnueAcc; ++j) acc[j] = b0_[j];
  const auto& cells = board.rawCells();
  for (int m = 0; m < kCells; ++m) {
    if (cells[m] != Cell::Empty) {
      addFeature(nnueFeatureIndex(m, cells[m]), 1, acc);
    }
  }
}

void NnueNetwork::updateAccumulator(const Board& board, int move, Cell to,
                                    NnueAccumulator& acc) const {
  if (!loaded_) throw NnueError("NNUE accumulator: weights not loaded");
  // `board` is the state BEFORE the change, exactly like Evaluator::update().
  const Cell before = board.cell(move);
  if (before != Cell::Empty) {
    addFeature(nnueFeatureIndex(move, before), -1, acc);
  }
  if (to != Cell::Empty) {
    addFeature(nnueFeatureIndex(move, to), 1, acc);
  }
}

int NnueNetwork::inferSum(const int32_t* acc, int stm) const {
  std::array<int32_t, kNnueL1> z{};
  for (int j = 0; j < kNnueAcc; ++j) {
    const int32_t a = std::clamp(clampInt16(acc[j]), 0, qa_);
    if (a == 0) continue;
    const int8_t* row = &w1_[static_cast<size_t>(j) * kNnueL1];
    for (int k = 0; k < kNnueL1; ++k) {
      z[k] += a * static_cast<int32_t>(row[k]);
    }
  }

  std::array<int32_t, kNnueL1> a2{};
  for (int k = 0; k < kNnueL1; ++k) {
    const int64_t zq = roundDiv(z[k], qa_) + b1_[k];
    a2[k] = static_cast<int32_t>(std::clamp<int64_t>(zq, 0, qb_));
  }

  int64_t y = 0;
  const int8_t* head = &w2_[static_cast<size_t>(stm) * kNnueL1];
  for (int k = 0; k < kNnueL1; ++k) {
    y += static_cast<int64_t>(a2[k]) * head[k];
  }
  return static_cast<int>(roundDiv(y, qb_) + b2_[stm]);
}

int NnueNetwork::infer(const std::vector<int>& sparse, int stm) const {
  if (!loaded_) throw NnueError("NNUE inference: weights not loaded");
  if (stm != 0 && stm != 1) {
    throw NnueError("NNUE inference: stm must be 0 (black) or 1 (white)");
  }
  NnueAccumulator acc{};
  for (int j = 0; j < kNnueAcc; ++j) acc[j] = b0_[j];
  for (int index : sparse) {
    if (index < 0 || index >= kNnueFeatures) {
      throw NnueError("NNUE inference: feature index out of range " +
                      std::to_string(index));
    }
    addFeature(index, 1, acc);
  }
  return inferSum(acc.data(), stm);
}

int NnueNetwork::inferAccumulator(const NnueAccumulator& acc, int stm) const {
  if (!loaded_) throw NnueError("NNUE inference: weights not loaded");
  if (stm != 0 && stm != 1) {
    throw NnueError("NNUE inference: stm must be 0 (black) or 1 (white)");
  }
  return inferSum(acc.data(), stm);
}

}  // namespace gomoku
