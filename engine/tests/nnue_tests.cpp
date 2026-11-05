// Ticket 09: scalar NNUE loader, integer forward and incremental accumulator.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "board.h"
#include "eval.h"
#include "nnue.h"

static int g_failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::cerr << "FAIL line " << __LINE__ << ": " << #cond << "\n"; \
      ++g_failures;                                                    \
    }                                                                  \
  } while (0)

using namespace gomoku;

static_assert(kNnueFeatures == 450);
static_assert(kNnueFileSize == 478336);

namespace {

void putU32(std::vector<uint8_t>& blob, size_t at, uint32_t value) {
  for (int i = 0; i < 4; ++i) blob[at + i] = static_cast<uint8_t>(value >> (8 * i));
}

void putI16(uint8_t* data, size_t at, int value) {
  const uint16_t raw = static_cast<uint16_t>(static_cast<int16_t>(value));
  data[at] = static_cast<uint8_t>(raw & 0xFF);
  data[at + 1] = static_cast<uint8_t>(raw >> 8);
}

uint32_t fnv1a(const uint8_t* data, size_t size) {
  uint32_t hash = 0x811C9DC5u;
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ static_cast<uint32_t>(data[i])) * 0x01000193u;
  }
  return hash;
}

constexpr size_t kW0 = 0;
constexpr size_t kB0 = kW0 + kNnueW0Len;
constexpr size_t kW1 = kB0 + 2 * kNnueB0Len;
constexpr size_t kB1 = kW1 + kNnueW1Len;
constexpr size_t kW2 = kB1 + 2 * kNnueB1Len;
constexpr size_t kB2 = kW2 + kNnueW2Len;

// Same hand-computable weights as engine/tests/training_tests.py:
// W0[112,0]=5, b0[0]=3, W1[0,0]=7, b1[0]=1,
// W2[head0,0]=11, W2[head1,0]=-9, b2=[2,5].
std::vector<uint8_t> makeBlob(int b0First = 3, int w0First = 5,
                              int w0Extra = 0) {
  std::vector<uint8_t> blob(kNnueFileSize, 0);
  std::memcpy(blob.data(), "GMNNUE01", 8);
  putU32(blob, 8, 1);           // version
  putU32(blob, 12, 64);         // header_size
  putU32(blob, 16, 450);        // n_features
  putU32(blob, 20, 1024);       // n_acc
  putU32(blob, 24, 15);         // n_l1
  putU32(blob, 28, 2);          // n_heads
  putU32(blob, 32, 127);        // QA
  putU32(blob, 36, 64);         // QB
  putU32(blob, 40, 400);        // score_scale
  putU32(blob, 44, 64);         // weight_clamp
  // 48 = FNV-1a checksum, 52..63 reserved zeros.

  uint8_t* body = blob.data() + kNnueHeaderSize;
  body[kW0 + 112 * kNnueAcc] = static_cast<uint8_t>(static_cast<int8_t>(w0First));
  if (w0Extra != 0) {
    body[kW0 + 113 * kNnueAcc] = static_cast<uint8_t>(static_cast<int8_t>(w0Extra));
  }
  putI16(body, kB0, b0First);
  body[kW1] = 7;
  putI16(body, kB1, 1);
  body[kW2] = 11;                                         // head 0
  body[kW2 + kNnueL1] = static_cast<uint8_t>(static_cast<int8_t>(-9));  // head 1
  putI16(body, kB2, 2);
  putI16(body, kB2 + 2, 5);
  putU32(blob, 48, fnv1a(body, kNnueBodySize));
  return blob;
}

bool writeBlob(const std::string& path, const std::vector<uint8_t>& blob) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(blob.data()),
            static_cast<std::streamsize>(blob.size()));
  return static_cast<bool>(out);
}

std::string loadError(const std::string& path) {
  NnueNetwork network;
  try {
    network.load(path);
  } catch (const NnueError& error) {
    return error.what();
  }
  return "<accepted>";
}

}  // namespace

int main() {
  CHECK(nnueFeatureIndex(0, Cell::Black) == 0);
  CHECK(nnueFeatureIndex(112, Cell::Black) == 112);
  CHECK(nnueFeatureIndex(112, Cell::White) == 337);
  CHECK(nnueFeatureIndex(224, Cell::White) == 449);

  CHECK(roundDiv(-1, 64) == 0);
  CHECK(roundDiv(-33, 64) == -1);
  CHECK(roundDiv(33, 64) == 1);
  CHECK(roundDiv(-64, 127) == -1);
  CHECK(roundDiv(64, 127) == 1);
  CHECK(roundDiv(-191, 127) == -2);

  const std::string path = "nnue_tests_weights.nnue";
  const std::vector<uint8_t> blob = makeBlob();
  CHECK(writeBlob(path, blob));

  NnueNetwork network;
  CHECK(!network.loaded());
  bool threw = false;
  try {
    network.infer({}, 0);
  } catch (const NnueError&) {
    threw = true;
  }
  CHECK(threw);

  network.load(path);
  CHECK(network.loaded());
  CHECK(network.qa() == 127 && network.qb() == 64);
  CHECK(network.scoreScale() == 400);

  // Hand-computed integer forward (see training_tests.py):
  // acc0 = 3+5 = 8; z0 = 8*7 = 56; zq0 = round_div(56,127)+1 = 1;
  // y = 1*11 = 11; eval = round_div(11,64)+2 = 2.
  CHECK(network.infer({112}, 0) == 2);
  CHECK(network.infer({112}, 1) == 5);   // head 1: -9 / 64 -> 0, +5
  CHECK(network.infer({}, 0) == 2);      // empty board still has b0[0]=3
  CHECK(network.infer({112}, 0) != network.infer({112}, 1));

  threw = false;
  try {
    network.infer({112}, 2);
  } catch (const NnueError&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    network.infer({450}, 0);
  } catch (const NnueError&) {
    threw = true;
  }
  CHECK(threw);

  // Feature indices: cell order plus the black/white plane split.
  {
    Board board(RuleMode::Freestyle);
    board.setCell(Board::indexOf(7, 7), Cell::Black);   // 112
    board.setCell(Board::indexOf(0, 1), Cell::Black);   // 15
    board.setCell(Board::indexOf(8, 8), Cell::White);   // 8*15+8 = 128 -> 353
    board.finalizeSync();
    const std::vector<int> indices = nnueSparseIndices(board);
    CHECK(indices == std::vector<int>({15, 112, 353}));
    CHECK(std::is_sorted(indices.begin(), indices.end()));
  }

  // Incremental accumulator == full sparse forward, at every place/undo step.
  {
    Board board(RuleMode::Freestyle);
    Evaluator eval;
    eval.setNetwork(&network);
    eval.init(board);
    CHECK(eval.network() == &network);
    const int moves[] = {112, 15, 128, 113, 97, 96, 129, 111, 114, 130};
    const int count = static_cast<int>(sizeof(moves) / sizeof(moves[0]));
    for (int i = 0; i < count; ++i) {
      const Color c = board.sideToMove();
      const int stm = c == Color::Black ? 0 : 1;
      CHECK(eval.evaluate(board) ==
            network.infer(nnueSparseIndices(board), stm));
      Evaluator fresh;
      fresh.setNetwork(&network);
      fresh.init(board);
      CHECK(eval.evaluate(board) == fresh.evaluate(board));

      eval.update(board, moves[i], cellOf(c));
      board.place(moves[i]);
      const int nextStm = board.sideToMove() == Color::Black ? 0 : 1;
      CHECK(eval.evaluate(board) ==
            network.infer(nnueSparseIndices(board), nextStm));
    }
    for (int i = count - 1; i >= 0; --i) {
      eval.update(board, moves[i], Cell::Empty);
      board.undo();
      const int stm = board.sideToMove() == Color::Black ? 0 : 1;
      CHECK(eval.evaluate(board) ==
            network.infer(nnueSparseIndices(board), stm));
    }
  }

  // int16 saturation happens once, after the full sum.
  {
    const std::string saturatedPath = "nnue_tests_saturated.nnue";
    CHECK(writeBlob(saturatedPath, makeBlob(32760, 64)));
    NnueNetwork saturated;
    saturated.load(saturatedPath);
    // acc0 = clamp_i16(32760+64) = 32767 -> a0 = 127;
    // z0 = 889; zq0 = 7+1 = 8; y = 88; eval = round_div(88,64)+2 = 3.
    CHECK(saturated.infer({112}, 0) == 3);
    std::remove(saturatedPath.c_str());
  }

  // Format validation: every corruption is rejected with a clear message.
  {
    const std::string badPath = "nnue_tests_bad.nnue";
    auto expect = [&](const std::vector<uint8_t>& data, const char* keyword) {
      CHECK(writeBlob(badPath, data));
      const std::string message = loadError(badPath);
      if (message == "<accepted>" ||
          message.find(keyword) == std::string::npos) {
        std::cerr << "FAIL expected rejection containing '" << keyword
                  << "', got: " << message << "\n";
        ++g_failures;
      }
    };

    std::vector<uint8_t> bad = blob;
    bad[0] = 'X';
    expect(bad, "magic");

    bad = blob;
    putU32(bad, 8, 9);
    expect(bad, "version");

    bad = blob;
    putU32(bad, 12, 32);
    expect(bad, "header size");

    bad = blob;
    putU32(bad, 16, 8);
    expect(bad, "dimensions");

    bad = blob;
    bad[kNnueHeaderPrefix] = 1;
    expect(bad, "reserved");

    bad = blob;
    bad[kNnueHeaderSize + 10] ^= 1;
    expect(bad, "checksum");

    bad = blob;
    bad[kNnueHeaderSize + 113 * kNnueAcc] = 100;  // |100| > weight_clamp 64
    putU32(bad, 48, fnv1a(bad.data() + kNnueHeaderSize, kNnueBodySize));
    expect(bad, "weight_clamp");

    bad.resize(63);
    expect(bad, "truncated");

    bad = blob;
    bad.resize(kNnueFileSize - 1);
    expect(bad, "body length");

    CHECK(loadError("nnue_tests_missing.nnue").find("cannot open") !=
          std::string::npos);
    std::remove(badPath.c_str());
  }

  std::remove(path.c_str());
  if (g_failures > 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "nnue_tests: all checks passed\n";
  return 0;
}
