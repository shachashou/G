#include "value_net.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <utility>

#include "json.h"

namespace gomoku {

const char* valueBackendName(ValueBackend backend) {
  return backend == ValueBackend::Avx2 ? "avx2" : "scalar";
}

const char* valueAccumulatorModeName(ValueAccumulatorMode mode) {
  return mode == ValueAccumulatorMode::Incremental ? "incremental"
                                                   : "full-rebuild";
}

bool valueEvalModeFromName(const std::string& name, ValueBackend& backend,
                           ValueAccumulatorMode& accumulator) {
  if (name == "avx2") {
    backend = ValueBackend::Avx2;
    accumulator = ValueAccumulatorMode::Incremental;
    return true;
  }
  if (name == "scalar") {
    backend = ValueBackend::Scalar;
    accumulator = ValueAccumulatorMode::Incremental;
    return true;
  }
  if (name == "reference") {
    backend = ValueBackend::Scalar;
    accumulator = ValueAccumulatorMode::FullRebuild;
    return true;
  }
  return false;
}

std::string valueEvalModeName(ValueBackend backend,
                              ValueAccumulatorMode accumulator) {
  return std::string(valueAccumulatorModeName(accumulator)) + "-" +
         valueBackendName(backend);
}

namespace {

constexpr char kMagic[8] = {'G', 'M', 'V', 'A', 'L', 'U', 'E', '1'};
constexpr uint32_t kVersion = 1;
// Header enums from trainer/value_format.py.
constexpr uint32_t kRuleRenju = 2;
constexpr uint32_t kFeatureSpecAbsoluteColor = 1;
constexpr uint32_t kValueContractBlackAbsolute = 1;
constexpr size_t kMaxMetaBytes = 1u << 20;
constexpr size_t kMaxLabelVersions = 64;
constexpr size_t kMaxStringBytes = 1024;

uint32_t readU32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) |
         (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

int16_t readI16(const uint8_t* p) {
  const uint32_t raw = static_cast<uint32_t>(p[0]) |
                       static_cast<uint32_t>(p[1]) << 8;
  return raw >= 0x8000u ? static_cast<int16_t>(static_cast<int32_t>(raw) - 0x10000)
                        : static_cast<int16_t>(raw);
}

int32_t readI32(const uint8_t* p) {
  return static_cast<int32_t>(readU32(p));
}

uint32_t fnv1a(const uint8_t* data, size_t size) {
  uint32_t hash = 0x811C9DC5u;
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ static_cast<uint32_t>(data[i])) * 0x01000193u;
  }
  return hash;
}

// Round n/d half away from zero (d > 0). Mirrors value_format.round_div; the
// integer reference is the parity contract, plain truncation is not.
int64_t valueRoundDiv(int64_t n, int64_t d) {
  const int64_t sign = n < 0 ? -1 : 1;
  const int64_t magnitude = n < 0 ? -n : n;
  return sign * ((2 * magnitude + d) / (2 * d));
}

const json::Value* findField(const json::Value& object, const char* key,
                             const std::string& prefix) {
  const json::Value* value = object.find(key);
  if (!value) {
    throw ValueNetError(prefix + "metadata is missing '" + key + "'");
  }
  return value;
}

std::string requireString(const json::Value& object, const char* key,
                          const std::string& prefix) {
  const json::Value* value = findField(object, key, prefix);
  if (value->type != json::Value::Type::String || value->string.empty()) {
    throw ValueNetError(prefix + "metadata '" + key + "' must be a non-empty string");
  }
  if (value->string.size() > kMaxStringBytes) {
    throw ValueNetError(prefix + "metadata '" + key + "' is unreasonably long");
  }
  return value->string;
}

int requireInt(const json::Value& object, const char* key,
               const std::string& prefix) {
  const json::Value* value = findField(object, key, prefix);
  if (value->type != json::Value::Type::Number ||
      value->number != static_cast<double>(static_cast<int64_t>(value->number))) {
    throw ValueNetError(prefix + "metadata '" + key + "' must be an integer");
  }
  const int64_t as_int = static_cast<int64_t>(value->number);
  if (as_int < std::numeric_limits<int>::min() ||
      as_int > std::numeric_limits<int>::max()) {
    throw ValueNetError(prefix + "metadata '" + key + "' is out of range");
  }
  return static_cast<int>(as_int);
}

}  // namespace

std::string ValueModelInfo::arch() const {
  return std::to_string(kValueFeatures) + "-" + std::to_string(kValueAcc) +
         "-" + std::to_string(kValueH1) + "-" + std::to_string(kValueH2) +
         "-" + std::to_string(kValueOut);
}

void ValueNetwork::load(const std::string& path) {
  const std::string prefix = "value model '" + path + "': ";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw ValueNetError(prefix + "cannot open file");
  in.seekg(0, std::ios::end);
  const std::streamoff length = in.tellg();
  if (length < 0) throw ValueNetError(prefix + "cannot determine file size");
  in.seekg(0, std::ios::beg);
  std::vector<uint8_t> blob(static_cast<size_t>(length));
  if (!blob.empty()) {
    in.read(reinterpret_cast<char*>(blob.data()),
            static_cast<std::streamsize>(blob.size()));
  }
  if (static_cast<size_t>(in.gcount()) != blob.size()) {
    throw ValueNetError(prefix + "short read");
  }
  in.close();

  if (blob.size() < static_cast<size_t>(kValueHeaderSize)) {
    throw ValueNetError(prefix + "truncated header (need 64 bytes, got " +
                        std::to_string(blob.size()) + ")");
  }
  if (std::memcmp(blob.data(), kMagic, sizeof(kMagic)) != 0) {
    std::string magic(blob.begin(), blob.begin() + 8);
    throw ValueNetError(prefix + "bad magic '" + magic +
                        "' (expected GMVALUE1)");
  }
  const uint32_t version = readU32(&blob[8]);
  const uint32_t header_size = readU32(&blob[12]);
  const uint32_t n_features = readU32(&blob[16]);
  const uint32_t n_acc = readU32(&blob[20]);
  const uint32_t n_h1 = readU32(&blob[24]);
  const uint32_t n_h2 = readU32(&blob[28]);
  const uint32_t n_out = readU32(&blob[32]);
  const uint32_t rule = readU32(&blob[36]);
  const uint32_t feature_spec = readU32(&blob[40]);
  const uint32_t value_contract = readU32(&blob[44]);
  const uint32_t ea = readU32(&blob[48]);
  const uint32_t act_max_int = readU32(&blob[52]);
  const uint32_t checksum = readU32(&blob[56]);
  const uint32_t meta_len = readU32(&blob[60]);

  if (version != kVersion) {
    throw ValueNetError(prefix + "unsupported value-model version " +
                        std::to_string(version));
  }
  if (header_size != kValueHeaderSize) {
    throw ValueNetError(prefix + "bad header size " +
                        std::to_string(header_size));
  }
  if (n_features != kValueFeatures || n_acc != kValueAcc ||
      n_h1 != kValueH1 || n_h2 != kValueH2 || n_out != kValueOut) {
    throw ValueNetError(prefix + "unsupported network dimensions " +
                        std::to_string(n_features) + "/" +
                        std::to_string(n_acc) + "/" + std::to_string(n_h1) +
                        "/" + std::to_string(n_h2) + "/" +
                        std::to_string(n_out) + " (expected 450/1024/64/32/1)");
  }
  if (rule != kRuleRenju) {
    throw ValueNetError(prefix + "unsupported rule id " +
                        std::to_string(rule) + " (expected Renju)");
  }
  if (feature_spec != kFeatureSpecAbsoluteColor) {
    throw ValueNetError(prefix + "unsupported feature spec id " +
                        std::to_string(feature_spec));
  }
  if (value_contract != kValueContractBlackAbsolute) {
    throw ValueNetError(prefix + "unsupported value contract id " +
                        std::to_string(value_contract));
  }
  if (ea < 1 || ea > 16) {
    throw ValueNetError(prefix + "invalid fixed-point exponent " +
                        std::to_string(ea));
  }
  if (act_max_int < 1 || act_max_int > (1u << 30)) {
    throw ValueNetError(prefix + "invalid activation ceiling " +
                        std::to_string(act_max_int));
  }
  if (meta_len == 0 || meta_len > kMaxMetaBytes) {
    throw ValueNetError(prefix + "bad metadata length " +
                        std::to_string(meta_len));
  }
  const size_t meta_start = kValueHeaderSize;
  const size_t meta_end = meta_start + static_cast<size_t>(meta_len);
  const size_t expected_size = meta_end + static_cast<size_t>(kValueBodyBytes);
  if (expected_size != blob.size()) {
    throw ValueNetError(prefix + "file length " + std::to_string(blob.size()) +
                        " does not match header (expected " +
                        std::to_string(expected_size) + ")");
  }
  const uint8_t* meta_ptr = blob.data() + meta_start;
  const uint8_t* body = blob.data() + meta_end;
  if (fnv1a(meta_ptr, static_cast<size_t>(meta_len) +
                       static_cast<size_t>(kValueBodyBytes)) != checksum) {
    throw ValueNetError(prefix + "checksum mismatch");
  }

  // Metadata must agree with the header; a file whose JSON describes another
  // architecture / rule / feature / value contract is rejected instead of
  // being reinterpreted.
  json::Value meta;
  try {
    meta = json::parse(std::string(reinterpret_cast<const char*>(meta_ptr),
                                   static_cast<size_t>(meta_len)));
  } catch (const json::ParseError& error) {
    throw ValueNetError(prefix + "metadata is not valid JSON: " +
                        std::string(error.what()));
  }
  if (!meta.isObject()) {
    throw ValueNetError(prefix + "metadata must be a JSON object");
  }
  const json::Value* arch_value = findField(meta, "arch", prefix);
  if (!arch_value->isArray() || arch_value->array.size() != 5) {
    throw ValueNetError(prefix + "metadata 'arch' must be a 5-element array");
  }
  const int expected_arch[5] = {kValueFeatures, kValueAcc, kValueH1, kValueH2,
                                kValueOut};
  for (size_t i = 0; i < 5; ++i) {
    const json::Value& item = arch_value->array[i];
    if (item.type != json::Value::Type::Number ||
        item.number != static_cast<double>(expected_arch[i])) {
      throw ValueNetError(prefix + "metadata architecture does not match the "
                          "450-1024-64-32-1 value net");
    }
  }
  if (requireString(meta, "rule", prefix) != "renju") {
    throw ValueNetError(prefix + "metadata rule is not renju");
  }
  if (requireString(meta, "feature_spec", prefix) != "absolute-color-450") {
    throw ValueNetError(prefix + "metadata feature spec is not "
                        "absolute-color-450");
  }
  if (requireString(meta, "value_contract", prefix) !=
      "black-absolute-tanh-v1") {
    throw ValueNetError(prefix + "metadata value contract is not "
                        "black-absolute-tanh-v1");
  }
  const json::Value* labels = findField(meta, "label_versions", prefix);
  if (!labels->isArray() || labels->array.size() > kMaxLabelVersions) {
    throw ValueNetError(prefix + "metadata 'label_versions' must be a list");
  }
  std::vector<std::string> label_versions;
  for (const json::Value& item : labels->array) {
    if (item.type != json::Value::Type::String ||
        item.string.size() > kMaxStringBytes) {
      throw ValueNetError(prefix + "metadata 'label_versions' entries must be "
                          "strings");
    }
    label_versions.push_back(item.string);
  }
  const json::Value* training = findField(meta, "training", prefix);
  if (!training->isObject()) {
    throw ValueNetError(prefix + "metadata 'training' must be an object");
  }
  const int meta_ea = requireInt(meta, "ea", prefix);
  if (static_cast<uint32_t>(meta_ea) != ea) {
    throw ValueNetError(prefix + "metadata/header fixed-point exponent mismatch");
  }
  const int meta_act_max = requireInt(meta, "act_max_int", prefix);
  if (static_cast<uint32_t>(meta_act_max) != act_max_int) {
    throw ValueNetError(prefix +
                        "metadata/header activation ceiling mismatch");
  }
  ValueModelInfo info;
  info.path = path;
  info.model_id = requireString(meta, "model_id", prefix);
  info.rule = "renju";
  info.feature_spec = "absolute-color-450";
  info.value_contract = "black-absolute-tanh-v1";
  info.label_versions = std::move(label_versions);
  info.file_bytes = static_cast<int64_t>(blob.size());
  info.ea = static_cast<int>(ea);
  info.act_max_int = static_cast<int>(act_max_int);
  info.fnv1a = checksum;
  if (const json::Value* created = training->find("created_at")) {
    if (created->type == json::Value::Type::String &&
        created->string.size() <= kMaxStringBytes) {
      info.created_at = created->string;
    }
  }
  if (const json::Value* steps = training->find("steps")) {
    if (steps->type == json::Value::Type::Number) {
      info.training_steps = static_cast<int64_t>(steps->number);
    }
  }

  // Decode the body exactly as BODY_SPEC lays it out.
  std::vector<int16_t> w0(static_cast<size_t>(kValueFeatures) * kValueAcc);
  std::vector<int32_t> b0(kValueAcc);
  std::vector<int16_t> w1(static_cast<size_t>(kValueAcc) * kValueH1);
  std::vector<int32_t> b1(kValueH1);
  std::vector<int16_t> w2(static_cast<size_t>(kValueH1) * kValueH2);
  std::vector<int32_t> b2(kValueH2);
  std::vector<int16_t> w3(kValueH2 * kValueOut);
  std::vector<int32_t> b3(kValueOut);
  size_t cursor = 0;
  for (size_t i = 0; i < w0.size(); ++i) {
    w0[i] = readI16(body + cursor + 2 * i);
  }
  cursor += w0.size() * 2;
  for (size_t i = 0; i < b0.size(); ++i) {
    b0[i] = readI32(body + cursor + 4 * i);
  }
  cursor += b0.size() * 4;
  for (size_t i = 0; i < w1.size(); ++i) {
    w1[i] = readI16(body + cursor + 2 * i);
  }
  cursor += w1.size() * 2;
  for (size_t i = 0; i < b1.size(); ++i) {
    b1[i] = readI32(body + cursor + 4 * i);
  }
  cursor += b1.size() * 4;
  for (size_t i = 0; i < w2.size(); ++i) {
    w2[i] = readI16(body + cursor + 2 * i);
  }
  cursor += w2.size() * 2;
  for (size_t i = 0; i < b2.size(); ++i) {
    b2[i] = readI32(body + cursor + 4 * i);
  }
  cursor += b2.size() * 4;
  for (size_t i = 0; i < w3.size(); ++i) {
    w3[i] = readI16(body + cursor + 2 * i);
  }
  cursor += w3.size() * 2;
  for (size_t i = 0; i < b3.size(); ++i) {
    b3[i] = readI32(body + cursor + 4 * i);
  }
  cursor += b3.size() * 4;
  if (cursor != static_cast<size_t>(kValueBodyBytes)) {
    throw ValueNetError(prefix + "internal body cursor error");
  }

  w0_ = std::move(w0);
  b0_ = std::move(b0);
  w1_ = std::move(w1);
  b1_ = std::move(b1);
  w2_ = std::move(w2);
  b2_ = std::move(b2);
  w3_ = std::move(w3);
  b3_ = std::move(b3);
  info_ = std::move(info);
  loaded_ = true;
}

void ValueNetwork::rebuildAccumulator(const Board& board,
                                      ValueAccumulator& acc) const {
  if (!loaded_) throw ValueNetError("value network: model not loaded");
  for (int j = 0; j < kValueAcc; ++j) acc[j] = b0_[j];
  const auto& cells = board.rawCells();
  for (int cell = 0; cell < kCells; ++cell) {
    const Cell stone = cells[cell];
    if (stone == Cell::Empty) continue;
    // Sparse reference set: black -> cell, white -> 225 + cell.
    const int feature =
        stone == Cell::Black ? cell : kValueCells + cell;
    const int16_t* row = &w0_[static_cast<size_t>(feature) * kValueAcc];
    for (int j = 0; j < kValueAcc; ++j) acc[j] += row[j];
  }
}

void ValueNetwork::addFeature(int feature, int sign,
                              ValueAccumulator& acc) const {
  if (backend_ == ValueBackend::Avx2) {
    addFeatureAvx2(feature, sign, acc);
    return;
  }
  const int16_t* row = &w0_[static_cast<size_t>(feature) * kValueAcc];
  for (int j = 0; j < kValueAcc; ++j) {
    acc[j] += static_cast<int64_t>(sign) * row[j];
  }
}

void ValueNetwork::addFeatureAvx2(int feature, int sign,
                                  ValueAccumulator& acc) const {
  // 8 int16 weights -> 8 exact int64 lanes (two 4-lane VPMOVSXDQ halves).
  // Add/sub on int64 lanes has no saturation, so any sequence of updates is
  // exactly reversible and bit-identical to the scalar path.
  const int16_t* row = &w0_[static_cast<size_t>(feature) * kValueAcc];
  int64_t* out = acc.data();
  for (int j = 0; j < kValueAcc; j += 8) {
    const __m256i weights = _mm256_cvtepi16_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + j)));
    const __m256i lo_w =
        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(weights));
    const __m256i hi_w =
        _mm256_cvtepi32_epi64(_mm256_extracti128_si256(weights, 1));
    __m256i lo =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(out + j));
    __m256i hi =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(out + j + 4));
    if (sign > 0) {
      lo = _mm256_add_epi64(lo, lo_w);
      hi = _mm256_add_epi64(hi, hi_w);
    } else {
      lo = _mm256_sub_epi64(lo, lo_w);
      hi = _mm256_sub_epi64(hi, hi_w);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j), lo);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j + 4), hi);
  }
}

void ValueNetwork::updateAccumulator(const Board& board, int move, Cell to,
                                     ValueAccumulator& acc) const {
  if (!loaded_) throw ValueNetError("value network: model not loaded");
  if (!Board::inBounds(move)) {
    throw ValueNetError("value network: accumulator move index out of range " +
                        std::to_string(move));
  }
  if (to != Cell::Empty && to != Cell::Black && to != Cell::White) {
    throw ValueNetError("value network: accumulator target is not a stone");
  }
  const Cell before = board.cell(move);
  if (before != Cell::Empty && to != Cell::Empty) {
    throw ValueNetError(
        "value network: accumulator place onto an occupied point");
  }
  if (before == Cell::Empty && to == Cell::Empty) {
    throw ValueNetError(
        "value network: accumulator undo of an empty point");
  }
  // Validation finished: the two operations below cannot fail, so an invalid
  // request never leaves the accumulator half-updated.
  if (before != Cell::Empty) {
    addFeature(before == Cell::Black ? move : kValueCells + move, -1, acc);
  }
  if (to != Cell::Empty) {
    addFeature(to == Cell::Black ? move : kValueCells + move, 1, acc);
  }
}

int64_t ValueNetwork::finishForward(
    const std::array<int64_t, kValueH1>& z1) const {
  const int act_max = info_.act_max_int;
  const int64_t scale = static_cast<int64_t>(1) << info_.ea;

  std::array<int32_t, kValueH1> a1{};
  for (int k = 0; k < kValueH1; ++k) {
    const int64_t value = valueRoundDiv(z1[k], scale) + b1_[k];
    a1[k] = static_cast<int32_t>(std::clamp<int64_t>(value, 0, act_max));
  }

  std::array<int64_t, kValueH2> z2{};
  for (int k = 0; k < kValueH1; ++k) {
    const int32_t value = a1[k];
    if (value == 0) continue;
    const int16_t* row = &w2_[static_cast<size_t>(k) * kValueH2];
    for (int m = 0; m < kValueH2; ++m) {
      z2[m] += static_cast<int64_t>(value) * row[m];
    }
  }
  std::array<int32_t, kValueH2> a2{};
  for (int m = 0; m < kValueH2; ++m) {
    const int64_t value = valueRoundDiv(z2[m], scale) + b2_[m];
    a2[m] = static_cast<int32_t>(std::clamp<int64_t>(value, 0, act_max));
  }

  int64_t z = 0;
  for (int m = 0; m < kValueH2; ++m) {
    if (a2[m] == 0) continue;
    z += static_cast<int64_t>(a2[m]) * w3_[static_cast<size_t>(m) * kValueOut];
  }
  return valueRoundDiv(z, scale) + b3_[0];
}

int64_t ValueNetwork::inferAccumulatorScalar(
    const ValueAccumulator& acc) const {
  if (!loaded_) throw ValueNetError("value network: model not loaded");
  const int act_max = info_.act_max_int;

  std::array<int32_t, kValueAcc> a0{};
  for (int j = 0; j < kValueAcc; ++j) {
    a0[j] = static_cast<int32_t>(std::clamp<int64_t>(acc[j], 0, act_max));
  }

  std::array<int64_t, kValueH1> z1{};
  for (int j = 0; j < kValueAcc; ++j) {
    const int32_t value = a0[j];
    if (value == 0) continue;
    const int16_t* row = &w1_[static_cast<size_t>(j) * kValueH1];
    for (int k = 0; k < kValueH1; ++k) {
      z1[k] += static_cast<int64_t>(value) * row[k];
    }
  }
  return finishForward(z1);
}

int64_t ValueNetwork::inferAccumulatorAvx2(
    const ValueAccumulator& acc) const {
  if (!loaded_) throw ValueNetError("value network: model not loaded");
  const int act_max = info_.act_max_int;

  std::array<int32_t, kValueAcc> a0{};
  for (int j = 0; j < kValueAcc; ++j) {
    a0[j] = static_cast<int32_t>(std::clamp<int64_t>(acc[j], 0, act_max));
  }

  // 1024 x 64 products with a 64-bit running sum. AVX2 has no 64-bit
  // multiply, so each 8-wide int32 row is split into even/odd 32-bit lanes and
  // multiplied with VPMULDQ (signed 32x32 -> 64). Exact, not approximate.
  // Four 8-output blocks are accumulated per pass (32 of 64 outputs), which
  // keeps the eight accumulators in registers and halves W1 traffic versus
  // one block at a time.
  std::array<int64_t, kValueH1> z1{};
  for (int k0 = 0; k0 < kValueH1; k0 += 32) {
    __m256i even0 = _mm256_setzero_si256();
    __m256i odd0 = _mm256_setzero_si256();
    __m256i even1 = _mm256_setzero_si256();
    __m256i odd1 = _mm256_setzero_si256();
    __m256i even2 = _mm256_setzero_si256();
    __m256i odd2 = _mm256_setzero_si256();
    __m256i even3 = _mm256_setzero_si256();
    __m256i odd3 = _mm256_setzero_si256();
    for (int j = 0; j < kValueAcc; ++j) {
      const int32_t value = a0[j];
      if (value == 0) continue;
      const int16_t* row = &w1_[static_cast<size_t>(j) * kValueH1 + k0];
      const __m256i factor = _mm256_set1_epi32(value);
      const __m256i factor_odd = _mm256_srli_epi64(factor, 32);

      const __m256i w0 = _mm256_cvtepi16_epi32(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(row)));
      even0 = _mm256_add_epi64(even0, _mm256_mul_epi32(w0, factor));
      odd0 = _mm256_add_epi64(
          odd0, _mm256_mul_epi32(_mm256_srli_epi64(w0, 32), factor_odd));

      const __m256i w1 = _mm256_cvtepi16_epi32(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 8)));
      even1 = _mm256_add_epi64(even1, _mm256_mul_epi32(w1, factor));
      odd1 = _mm256_add_epi64(
          odd1, _mm256_mul_epi32(_mm256_srli_epi64(w1, 32), factor_odd));

      const __m256i w2 = _mm256_cvtepi16_epi32(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 16)));
      even2 = _mm256_add_epi64(even2, _mm256_mul_epi32(w2, factor));
      odd2 = _mm256_add_epi64(
          odd2, _mm256_mul_epi32(_mm256_srli_epi64(w2, 32), factor_odd));

      const __m256i w3 = _mm256_cvtepi16_epi32(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 24)));
      even3 = _mm256_add_epi64(even3, _mm256_mul_epi32(w3, factor));
      odd3 = _mm256_add_epi64(
          odd3, _mm256_mul_epi32(_mm256_srli_epi64(w3, 32), factor_odd));
    }
    alignas(32) int64_t even_lanes[4];
    alignas(32) int64_t odd_lanes[4];
    const auto storeBlock = [&](int offset, __m256i even, __m256i odd) {
      _mm256_store_si256(reinterpret_cast<__m256i*>(even_lanes), even);
      _mm256_store_si256(reinterpret_cast<__m256i*>(odd_lanes), odd);
      for (int n = 0; n < 4; ++n) {
        z1[offset + 2 * n] = even_lanes[n];
        z1[offset + 2 * n + 1] = odd_lanes[n];
      }
    };
    storeBlock(k0, even0, odd0);
    storeBlock(k0 + 8, even1, odd1);
    storeBlock(k0 + 16, even2, odd2);
    storeBlock(k0 + 24, even3, odd3);
  }
  return finishForward(z1);
}

int64_t ValueNetwork::inferAccumulator(const ValueAccumulator& acc) const {
  return backend_ == ValueBackend::Avx2 ? inferAccumulatorAvx2(acc)
                                        : inferAccumulatorScalar(acc);
}

int64_t ValueNetwork::inferSparse(const std::vector<int>& features) const {
  if (!loaded_) throw ValueNetError("value network: model not loaded");
  ValueAccumulator acc{};
  for (int j = 0; j < kValueAcc; ++j) acc[j] = b0_[j];
  for (int feature : features) {
    if (feature < 0 || feature >= kValueFeatures) {
      throw ValueNetError("value network: feature index out of range " +
                          std::to_string(feature));
    }
    const int16_t* row = &w0_[static_cast<size_t>(feature) * kValueAcc];
    for (int j = 0; j < kValueAcc; ++j) acc[j] += row[j];
  }
  return inferAccumulatorScalar(acc);
}

int64_t ValueNetwork::evaluateBoard(const Board& board) const {
  ValueAccumulator acc{};
  rebuildAccumulator(board, acc);
  return inferAccumulatorScalar(acc);
}

int ValueNetwork::zToSearchScore(int64_t z, int ea) {
  const int64_t scale = static_cast<int64_t>(1) << ea;
  // |z| stays below ~1.2e15 for any header-valid model (act_max <= 2^30,
  // int16 weights, <= 64 activation lanes), so z*1000 fits int64.
  const int64_t rounded = valueRoundDiv(z * kValueScoreScale, scale);
  return static_cast<int>(
      std::clamp<int64_t>(rounded, -kValueScoreLimit, kValueScoreLimit));
}

int ValueNetwork::sideToMoveScore(int64_t z, Color side) const {
  const int black_score = zToSearchScore(z, info_.ea);
  return side == Color::Black ? black_score : -black_score;
}

double ValueNetwork::valueFromZ(int64_t z, int ea) {
  return std::tanh(static_cast<double>(z) /
                   static_cast<double>(static_cast<int64_t>(1) << ea));
}

}  // namespace gomoku
