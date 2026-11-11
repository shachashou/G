// Ticket 07: GMVALUE1 loader, scalar reference forward, full first-layer
// rebuild, value/side-to-move conversion, rejection paths and the Alpha-Beta
// integration (terminal and proven-VCF results are never replaced by a
// network score).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "board.h"
#include "eval.h"
#include "nnue.h"
#include "search.h"
#include "value_net.h"

static int g_failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::cerr << "FAIL line " << __LINE__ << ": " << #cond << "\n"; \
      ++g_failures;                                                    \
    }                                                                  \
  } while (0)

using namespace gomoku;

namespace {

uint32_t fnv1a(const std::vector<uint8_t>& data) {
  uint32_t hash = 0x811C9DC5u;
  for (uint8_t byte : data) {
    hash = (hash ^ static_cast<uint32_t>(byte)) * 0x01000193u;
  }
  return hash;
}

void appendU32(std::vector<uint8_t>& out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
  out.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}

void appendI16(std::vector<uint8_t>& out, int16_t value) {
  const uint16_t raw = static_cast<uint16_t>(value);
  out.push_back(static_cast<uint8_t>(raw & 0xFF));
  out.push_back(static_cast<uint8_t>(raw >> 8));
}

void appendI32(std::vector<uint8_t>& out, int32_t value) {
  appendU32(out, static_cast<uint32_t>(value));
}

struct Parameters {
  std::vector<int16_t> w0 = std::vector<int16_t>(
      static_cast<size_t>(kValueFeatures) * kValueAcc, 0);
  std::vector<int32_t> b0 = std::vector<int32_t>(kValueAcc, 0);
  std::vector<int16_t> w1 = std::vector<int16_t>(
      static_cast<size_t>(kValueAcc) * kValueH1, 0);
  std::vector<int32_t> b1 = std::vector<int32_t>(kValueH1, 0);
  std::vector<int16_t> w2 = std::vector<int16_t>(
      static_cast<size_t>(kValueH1) * kValueH2, 0);
  std::vector<int32_t> b2 = std::vector<int32_t>(kValueH2, 0);
  std::vector<int16_t> w3 = std::vector<int16_t>(kValueH2 * kValueOut, 0);
  std::vector<int32_t> b3 = std::vector<int32_t>(kValueOut, 0);
  int ea = 1;
  int act_max_int = 16;
};

std::vector<uint8_t> buildBlob(const Parameters& p, std::string meta,
                               uint32_t rule = 2,
                               uint32_t feature_spec = 1,
                               uint32_t value_contract = 1) {
  std::vector<uint8_t> body;
  for (int16_t v : p.w0) appendI16(body, v);
  for (int32_t v : p.b0) appendI32(body, v);
  for (int16_t v : p.w1) appendI16(body, v);
  for (int32_t v : p.b1) appendI32(body, v);
  for (int16_t v : p.w2) appendI16(body, v);
  for (int32_t v : p.b2) appendI32(body, v);
  for (int16_t v : p.w3) appendI16(body, v);
  for (int32_t v : p.b3) appendI32(body, v);

  std::vector<uint8_t> signed_content(meta.begin(), meta.end());
  signed_content.insert(signed_content.end(), body.begin(), body.end());
  const uint32_t checksum = fnv1a(signed_content);

  std::vector<uint8_t> out;
  const char kMagic[8] = {'G', 'M', 'V', 'A', 'L', 'U', 'E', '1'};
  out.insert(out.end(), kMagic, kMagic + 8);
  appendU32(out, 1);                              // version
  appendU32(out, kValueHeaderSize);               // header size
  appendU32(out, kValueFeatures);
  appendU32(out, kValueAcc);
  appendU32(out, kValueH1);
  appendU32(out, kValueH2);
  appendU32(out, kValueOut);
  appendU32(out, rule);
  appendU32(out, feature_spec);
  appendU32(out, value_contract);
  appendU32(out, static_cast<uint32_t>(p.ea));
  appendU32(out, static_cast<uint32_t>(p.act_max_int));
  appendU32(out, checksum);
  appendU32(out, static_cast<uint32_t>(meta.size()));
  out.insert(out.end(), meta.begin(), meta.end());
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

std::string metadata(const Parameters& p, const std::string& arch =
                                             "[450,1024,64,32,1]",
                     const std::string& contract =
                         "black-absolute-tanh-v1",
                     int meta_ea = -1, int meta_act = -1) {
  if (meta_ea < 0) meta_ea = p.ea;
  if (meta_act < 0) meta_act = p.act_max_int;
  return "{\"arch\":" + arch + ",\"ea\":" + std::to_string(meta_ea) +
         ",\"act_max_int\":" + std::to_string(meta_act) +
         ",\"feature_spec\":\"absolute-color-450\","
         "\"label_versions\":[\"unit-test-v1\"],"
         "\"model_id\":\"unit-test-model\","
         "\"rule\":\"renju\","
         "\"training\":{\"steps\":7,\"created_at\":\"2026-11-10T00:00:00Z\"},"
         "\"value_contract\":\"" + contract + "\"}";
}

class TempFile {
 public:
  explicit TempFile(const std::string& name)
      : path_(std::filesystem::temp_directory_path() / name),
        path_string_(path_.string()) {}
  ~TempFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  const std::string& path() const { return path_string_; }
  void write(const std::vector<uint8_t>& bytes) const {
    std::ofstream out(path(), std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }

 private:
  std::filesystem::path path_;
  std::string path_string_;
};

bool loadRejects(const std::vector<uint8_t>& blob, const std::string& keyword,
                 const std::string& label) {
  TempFile file("gomoku_value_net_reject.gmvalue");
  file.write(blob);
  ValueNetwork network;
  try {
    network.load(file.path());
  } catch (const ValueNetError& error) {
    const std::string message = error.what();
    if (message.find(keyword) == std::string::npos) {
      std::cerr << "  " << label << ": message lacks '" << keyword
                << "' -> " << message << "\n";
      return false;
    }
    return true;
  }
  std::cerr << "  " << label << ": load unexpectedly succeeded\n";
  return false;
}

// Hand-computable network (ea=1, act=16):
//   black cell 112 -> W0 row 0 = 4, white cell 113 -> W0 row 1 = 3, b0[0] = 3
//   W1 = diag(2), W2 = diag(2), W3 = [2,2], biases 0 (except b0[0]).
Parameters handParameters() {
  Parameters p;
  p.b0[0] = 3;
  p.w0[static_cast<size_t>(112) * kValueAcc + 0] = 4;
  p.w0[static_cast<size_t>(225 + 113) * kValueAcc + 1] = 3;
  p.w1[0 * kValueH1 + 0] = 2;
  p.w1[1 * kValueH1 + 1] = 2;
  p.w2[0 * kValueH2 + 0] = 2;
  p.w2[1 * kValueH2 + 1] = 2;
  p.w3[0] = 2;
  p.w3[1] = 2;
  return p;
}

// Constant black-absolute z = b3, independent of the board.
Parameters constantParameters(int32_t b3) {
  Parameters p;
  p.b3[0] = b3;
  return p;
}

bool completesFive(const Board& board, int move) {
  if (!board.isPlayable(move)) return false;
  Board copy = board;
  const Color side = copy.sideToMove();
  copy.place(move);
  return copy.result() == (side == Color::Black ? GameResult::BlackWin
                                                : GameResult::WhiteWin);
}

Board boardWith(const std::vector<std::pair<int, Cell>>& stones, Color side) {
  Board board(RuleMode::Renju);
  for (const auto& [cell, stone] : stones) board.setCell(cell, stone);
  board.finalizeSync();
  board.setSideToMove(side);
  return board;
}

int index(int x, int y) { return Board::indexOf(x, y); }

std::vector<int> legalMoves(const Board& board) {
  std::vector<int> moves;
  if (board.result() != GameResult::Ongoing) return moves;
  for (int m = 0; m < kCells; ++m) {
    if (board.isPlayable(m)) moves.push_back(m);
  }
  return moves;
}

// Random legal play with per-ply first-layer comparison, then full undo.
void checkRandomIncremental(ValueNetwork& network, uint32_t seed, int plies,
                            const Board& start = Board(RuleMode::Renju)) {
  Board board = start;
  ValueAccumulator acc{};
  network.rebuildAccumulator(board, acc);
  std::mt19937 rng(seed);
  std::vector<int> played;
  for (int ply = 0; ply < plies; ++ply) {
    const std::vector<int> moves = legalMoves(board);
    if (moves.empty()) break;
    const int move = moves[rng() % moves.size()];
    network.updateAccumulator(board, move, cellOf(board.sideToMove()), acc);
    board.place(move);
    played.push_back(move);
    ValueAccumulator full{};
    network.rebuildAccumulator(board, full);
    CHECK(acc == full);
    CHECK(network.inferAccumulator(acc) == network.evaluateBoard(board));
    CHECK(network.inferAccumulatorAvx2(acc) ==
          network.inferAccumulatorScalar(acc));
  }
  while (!played.empty()) {
    const int move = board.lastMove();
    network.updateAccumulator(board, move, Cell::Empty, acc);
    board.undo();
    played.pop_back();
    ValueAccumulator full{};
    network.rebuildAccumulator(board, full);
    CHECK(acc == full);
    CHECK(network.inferAccumulator(acc) == network.evaluateBoard(board));
  }
}

}  // namespace

int main() {
  const Parameters hand = handParameters();
  TempFile hand_file("gomoku_value_net_hand.gmvalue");
  hand_file.write(buildBlob(hand, metadata(hand)));

  {
    std::cout << "[load and identity]\n";
    ValueNetwork network;
    network.load(hand_file.path());
    CHECK(network.loaded());
    CHECK(network.info().arch() == "450-1024-64-32-1");
    CHECK(network.info().model_id == "unit-test-model");
    CHECK(network.info().rule == "renju");
    CHECK(network.info().feature_spec == "absolute-color-450");
    CHECK(network.info().value_contract == "black-absolute-tanh-v1");
    CHECK(network.info().ea == 1);
    CHECK(network.info().act_max_int == 16);
    CHECK(network.info().training_steps == 7);
    CHECK(network.info().label_versions.size() == 1);
    CHECK(network.info().file_bytes ==
          static_cast<int64_t>(kValueHeaderSize + kValueBodyBytes +
                               metadata(hand).size()));
  }

  {
    std::cout << "[reference scalar forward]\n";
    ValueNetwork network;
    network.load(hand_file.path());
    CHECK(network.inferSparse({}) == 3);
    CHECK(network.inferSparse({112}) == 7);
    CHECK(network.inferSparse({225 + 113}) == 6);
    CHECK(network.inferSparse({112, 225 + 113}) == 10);

    Board black_only = boardWith({{112, Cell::Black}}, Color::Black);
    CHECK(network.evaluateBoard(black_only) == 7);
    CHECK(network.inferSparse(nnueSparseIndices(black_only)) == 7);

    Board mixed = boardWith({{112, Cell::Black}, {113, Cell::White}},
                            Color::Black);
    CHECK(network.evaluateBoard(mixed) == 10);
    CHECK(network.inferSparse(nnueSparseIndices(mixed)) == 10);

    ValueAccumulator acc{};
    network.rebuildAccumulator(mixed, acc);
    CHECK(acc[0] == 7 && acc[1] == 3);
    CHECK(network.inferAccumulator(acc) == 10);
  }

  {
    std::cout << "[side-to-move value conversion]\n";
    ValueNetwork network;
    network.load(hand_file.path());
    const int64_t z = 10;  // black-absolute fixed point, value = tanh(5)
    CHECK(network.sideToMoveScore(z, Color::Black) == 5000);
    CHECK(network.sideToMoveScore(z, Color::White) == -5000);
    CHECK(network.sideToMoveScore(3, Color::Black) == 1500);
    CHECK(ValueNetwork::zToSearchScore(-3, 1) == -1500);
    CHECK(ValueNetwork::zToSearchScore(1000000000LL, 1) == kValueScoreLimit);
    CHECK(ValueNetwork::zToSearchScore(-1000000000LL, 1) == -kValueScoreLimit);
    const double value = ValueNetwork::valueFromZ(10, 1);
    CHECK(value > 0.99 && value <= 1.0);
  }

  {
    std::cout << "[full rebuild matches after place/undo sequences]\n";
    ValueNetwork network;
    network.load(hand_file.path());
    Board board(RuleMode::Renju);
    const int moves[] = {index(7, 7), index(8, 8), index(6, 6), index(9, 9),
                         index(5, 5), index(10, 10)};
    for (int move : moves) {
      const int64_t before = network.evaluateBoard(board);
      CHECK(before == network.inferSparse(nnueSparseIndices(board)));
      board.place(move);
      CHECK(network.evaluateBoard(board) ==
            network.inferSparse(nnueSparseIndices(board)));
    }
    for (int i = static_cast<int>(sizeof(moves) / sizeof(moves[0])) - 1;
         i >= 0; --i) {
      board.undo();
      CHECK(network.evaluateBoard(board) ==
            network.inferSparse(nnueSparseIndices(board)));
    }
    CHECK(network.evaluateBoard(board) == 3);
  }

  {
    std::cout << "[incremental first layer matches full rebuild]\n";
    ValueNetwork network;
    network.load(hand_file.path());
    for (uint32_t seed = 1; seed <= 4; ++seed) {
      checkRandomIncremental(network, seed, 225);
    }
    // The scalar update path must match the same full rebuild.
    network.setBackend(ValueBackend::Scalar);
    checkRandomIncremental(network, 7, 150);
    network.setBackend(ValueBackend::Avx2);
    // A BOARD-synced midgame position (no move history) must also rebuild and
    // then support place/undo exactly like a game started from empty.
    checkRandomIncremental(
        network, 99, 120,
        boardWith({{index(7, 7), Cell::Black},
                   {index(8, 8), Cell::White},
                   {index(6, 6), Cell::Black},
                   {index(9, 9), Cell::White},
                   {index(5, 5), Cell::Black},
                   {index(10, 10), Cell::White}},
                  Color::Black));
  }

  {
    std::cout << "[illegal first-layer updates leave the accumulator intact]\n";
    ValueNetwork network;
    network.load(hand_file.path());
    Board board = boardWith({{112, Cell::Black}}, Color::Black);
    ValueAccumulator acc{};
    network.rebuildAccumulator(board, acc);
    const ValueAccumulator before = acc;
    const auto expectThrow = [&](int move, Cell to, const char* label) {
      bool threw = false;
      try {
        network.updateAccumulator(board, move, to, acc);
      } catch (const ValueNetError&) {
        threw = true;
      }
      CHECK(threw);
      CHECK(acc == before);
      if (!threw) std::cerr << "  " << label << " did not throw\n";
    };
    expectThrow(112, Cell::Black, "place onto occupied");
    expectThrow(113, Cell::Empty, "undo of an empty point");
    expectThrow(-1, Cell::Black, "negative cell");
    expectThrow(kCells, Cell::Black, "cell past the board");
    expectThrow(114, static_cast<Cell>(7), "non-stone target");
  }

  {
    std::cout << "[wide accumulator is exact and reversible at the limits]\n";
    Parameters wide = constantParameters(0);
    wide.b0[0] = 2147483647;
    wide.b0[1] = -2147483647;
    for (size_t i = 0; i < wide.w0.size(); ++i) {
      wide.w0[i] = (i % 2 == 0) ? -32768 : 32767;
    }
    TempFile wide_file("gomoku_value_net_wide.gmvalue");
    wide_file.write(buildBlob(wide, metadata(wide)));
    ValueNetwork network;
    network.load(wide_file.path());

    std::vector<std::pair<int, Cell>> stones;
    for (int i = 0; i < 100; ++i) {
      stones.push_back({(i * 7 + 3) % kCells,
                        i % 2 == 0 ? Cell::Black : Cell::White});
    }
    Board board = boardWith(stones, Color::Black);
    // Use the board's final cell colours (duplicates, if any, would otherwise
    // be subtracted with the wrong feature plane).
    std::vector<std::pair<int, Cell>> unique;
    for (int m = 0; m < kCells; ++m) {
      if (board.cell(m) != Cell::Empty) unique.push_back({m, board.cell(m)});
    }
    ValueAccumulator acc{};
    network.rebuildAccumulator(board, acc);
    const ValueAccumulator full = acc;

    Board empty(RuleMode::Renju);
    ValueAccumulator base{};
    network.rebuildAccumulator(empty, base);

    // Subtract every row in a different order than the rebuild added them.
    for (const auto& [cell, stone] : unique) {
      (void)stone;
      network.updateAccumulator(board, cell, Cell::Empty, acc);
    }
    CHECK(acc == base);
    // Adding them back (to a board that reports those cells empty) restores
    // the exact full accumulator.
    for (const auto& [cell, stone] : unique) {
      network.updateAccumulator(empty, cell, stone, acc);
    }
    CHECK(acc == full);
    CHECK(network.inferAccumulator(acc) ==
          network.inferAccumulatorScalar(acc));
  }

  {
    std::cout << "[Evaluator incremental state matches the reference path]\n";
    ValueNetwork network;
    network.load(hand_file.path());
    Board board(RuleMode::Renju);
    Evaluator eval;
    eval.setValueNetwork(&network);
    eval.init(board);
    const auto reference = [&]() {
      return network.sideToMoveScore(network.evaluateBoard(board),
                                     board.sideToMove());
    };
    CHECK(eval.evaluate(board) == reference());
    const int moves[] = {index(7, 7), index(8, 8), index(6, 6), index(9, 9),
                         index(5, 5), index(10, 10)};
    for (int move : moves) {
      eval.update(board, move, cellOf(board.sideToMove()));
      board.place(move);
      CHECK(eval.evaluate(board) == reference());
    }
    for (int i = static_cast<int>(sizeof(moves) / sizeof(moves[0])) - 1;
         i >= 0; --i) {
      const int move = board.lastMove();
      eval.update(board, move, Cell::Empty);
      board.undo();
      CHECK(eval.evaluate(board) == reference());
    }
    CHECK(eval.evaluate(board) == reference());
  }

  {
    std::cout << "[search is identical across avx2 / scalar / reference]\n";
    ValueNetwork network;
    network.load(hand_file.path());
    Board position = boardWith({{index(7, 7), Cell::Black},
                                {index(8, 8), Cell::White},
                                {index(6, 6), Cell::Black},
                                {index(9, 9), Cell::White},
                                {index(5, 5), Cell::Black},
                                {index(10, 10), Cell::White}},
                               Color::Black);
    SearchLimits limits;
    limits.vcfPolicy = VcfPolicy::Off;
    limits.maxDepth = 2;
    limits.moveTimeMs = 0;
    Search search;
    search.setValueNetwork(&network);
    const auto run = [&](ValueBackend backend,
                         ValueAccumulatorMode accumulator) {
      network.setBackend(backend);
      network.setAccumulatorMode(accumulator);
      search.reset();
      Board copy = position;
      return search.findBestMove(copy, limits);
    };
    const SearchResult avx2 = run(ValueBackend::Avx2,
                                  ValueAccumulatorMode::Incremental);
    const SearchResult scalar = run(ValueBackend::Scalar,
                                    ValueAccumulatorMode::Incremental);
    const SearchResult reference = run(ValueBackend::Scalar,
                                       ValueAccumulatorMode::FullRebuild);
    CHECK(avx2.move == scalar.move && avx2.move == reference.move);
    CHECK(avx2.score == scalar.score && avx2.score == reference.score);
    CHECK(avx2.nodes == scalar.nodes && avx2.nodes == reference.nodes);
    CHECK(avx2.reachedDepth == reference.reachedDepth);
  }

  {
    std::cout << "[backend identity names]\n";
    ValueBackend backend = ValueBackend::Scalar;
    ValueAccumulatorMode accumulator = ValueAccumulatorMode::FullRebuild;
    CHECK(valueEvalModeFromName("avx2", backend, accumulator));
    CHECK(backend == ValueBackend::Avx2);
    CHECK(accumulator == ValueAccumulatorMode::Incremental);
    CHECK(valueEvalModeName(backend, accumulator) == "incremental-avx2");
    CHECK(valueEvalModeFromName("scalar", backend, accumulator));
    CHECK(valueEvalModeName(backend, accumulator) == "incremental-scalar");
    CHECK(valueEvalModeFromName("reference", backend, accumulator));
    CHECK(valueEvalModeName(backend, accumulator) == "full-rebuild-scalar");
    CHECK(!valueEvalModeFromName("sse", backend, accumulator));
  }

  {
    std::cout << "[format rejection]\n";
    std::vector<uint8_t> bad_magic = buildBlob(hand, metadata(hand));
    bad_magic[0] = 'X';
    CHECK(loadRejects(bad_magic, "bad magic", "bad magic"));
    CHECK(loadRejects(buildBlob(hand, metadata(hand), /*rule=*/1),
                      "unsupported rule", "non-renju rule"));
    CHECK(loadRejects(buildBlob(hand, metadata(hand), 2,
                                /*feature_spec=*/9),
                      "feature spec", "unknown feature spec"));
    CHECK(loadRejects(buildBlob(hand, metadata(hand), 2, 1,
                                /*value_contract=*/7),
                      "value contract", "unknown value contract"));
    CHECK(loadRejects(
        buildBlob(hand, metadata(hand, "[450,1024,15,2,1]")),
        "architecture", "old architecture in metadata"));
    CHECK(loadRejects(
        buildBlob(hand, metadata(hand, "[450,1024,64,32,1]", "teacher-v1")),
        "value contract", "wrong value contract in metadata"));
    CHECK(loadRejects(buildBlob(hand, metadata(hand, "[450,1024,64,32,1]",
                                               "black-absolute-tanh-v1", 2)),
                      "exponent mismatch", "metadata/header ea mismatch"));
    CHECK(loadRejects(buildBlob(hand, metadata(hand, "[450,1024,64,32,1]",
                                               "black-absolute-tanh-v1", 1,
                                               17)),
                      "activation ceiling mismatch",
                      "metadata/header activation mismatch"));

    std::vector<uint8_t> blob = buildBlob(hand, metadata(hand));
    const size_t original = blob.size();

    std::vector<uint8_t> truncated(blob.begin(), blob.end() - 1);
    CHECK(loadRejects(truncated, "file length", "truncated body"));

    std::vector<uint8_t> corrupted = blob;
    corrupted.back() ^= 0xFF;
    CHECK(loadRejects(corrupted, "checksum", "corrupted body checksum"));

    std::vector<uint8_t> short_header(blob.begin(), blob.begin() + 32);
    CHECK(loadRejects(short_header, "truncated header", "short header"));

    // Recomputing the checksum does not rescue a semantically bad header:
    std::vector<uint8_t> bad_header = buildBlob(hand, metadata(hand));
    bad_header[20] = 5;  // n_acc = 5 (header FNV covers body only)
    CHECK(loadRejects(bad_header, "dimensions", "bad header dimensions"));

    // An old-format file must not be reinterpreted by the new loader.
    std::vector<uint8_t> legacy(original, 0);
    const char kLegacy[8] = {'G', 'M', 'N', 'N', 'U', 'E', '0', '1'};
    std::memcpy(legacy.data(), kLegacy, 8);
    CHECK(loadRejects(legacy, "bad magic", "legacy GMNNUE01 file"));
  }

  {
    std::cout << "[Alpha-Beta integration: mate dominates the network]\n";
    TempFile negative_file("gomoku_value_net_negative.gmvalue");
    const Parameters negative = constantParameters(-500);
    negative_file.write(buildBlob(negative, metadata(negative)));
    ValueNetwork network;
    network.load(negative_file.path());
    // Black has an open four; the network says black is very bad, but the
    // immediate five is a terminal result that must dominate.
    Board board = boardWith({{index(7, 7), Cell::Black},
                             {index(8, 7), Cell::Black},
                             {index(9, 7), Cell::Black},
                             {index(10, 7), Cell::Black},
                             {index(6, 7), Cell::White},
                             {index(6, 8), Cell::White},
                             {index(5, 9), Cell::White}},
                            Color::Black);
    Search search;
    search.setValueNetwork(&network);
    SearchLimits limits;
    limits.vcfPolicy = VcfPolicy::Off;
    limits.maxDepth = 1;
    limits.moveTimeMs = 0;
    const SearchResult result = search.findBestMove(board, limits);
    CHECK(result.move >= 0);
    CHECK(completesFive(board, result.move));
    CHECK(result.score > 900000);
    CHECK(result.fromVcf == false);
  }

  {
    std::cout << "[Alpha-Beta integration: network score is bounded]\n";
    TempFile positive_file("gomoku_value_net_positive.gmvalue");
    const Parameters positive = constantParameters(500);
    positive_file.write(buildBlob(positive, metadata(positive)));
    ValueNetwork network;
    network.load(positive_file.path());
    Board board = boardWith({{index(7, 7), Cell::Black},
                             {index(8, 8), Cell::White}},
                            Color::Black);
    Search search;
    search.setValueNetwork(&network);
    SearchLimits limits;
    limits.vcfPolicy = VcfPolicy::Off;
    limits.maxDepth = 1;
    limits.moveTimeMs = 0;
    const SearchResult result = search.findBestMove(board, limits);
    CHECK(result.move >= 0);
    // b3=500 with ea=1 -> z=500 -> black score 250000 clamped to 100000.
    CHECK(result.score == kValueScoreLimit);
    CHECK(result.score < 900000);
  }

  {
    std::cout << "[Alpha-Beta integration: proven VCF is independent]\n";
    TempFile negative_file("gomoku_value_net_vcf.gmvalue");
    const Parameters negative = constantParameters(-500);
    negative_file.write(buildBlob(negative, metadata(negative)));
    ValueNetwork network;
    network.load(negative_file.path());
    Board board = boardWith(
        {{index(7, 7), Cell::Black},  {index(9, 7), Cell::Black},
         {index(10, 7), Cell::Black}, {index(8, 6), Cell::Black},
         {index(11, 6), Cell::Black}, {index(8, 5), Cell::Black},
         {index(7, 8), Cell::White},  {index(11, 7), Cell::White},
         {index(9, 6), Cell::White},  {index(10, 5), Cell::White},
         {index(8, 4), Cell::White},  {index(11, 4), Cell::White}},
        Color::Black);
    Search search;
    search.setValueNetwork(&network);
    SearchLimits limits;
    limits.vcfPolicy = VcfPolicy::Root;
    limits.maxDepth = 1;
    limits.moveTimeMs = 0;
    const SearchResult result = search.findBestMove(board, limits);
    CHECK(result.fromVcf);
    CHECK(result.move == index(8, 7));
    CHECK(result.score > 900000);
    CHECK(result.score < 1000000);
  }

  if (g_failures > 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "value_net_tests: all checks passed\n";
  return 0;
}
