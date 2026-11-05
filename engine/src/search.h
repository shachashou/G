#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

#include "board.h"
#include "eval.h"

namespace gomoku {

struct SearchLimits {
  int maxDepth = 8;
  int moveTimeMs = 1000;  // 0 => no clock, use maxDepth only
  bool useVcf = true;
  uint64_t vcfNodeLimit = 5000;
};

struct SearchResult {
  int move = -1;  // -1 when there is no legal move
  int score = 0;
  int reachedDepth = 0;
  uint64_t nodes = 0;
  bool fromVcf = false;
};

// Negamax Alpha-Beta with Principal Variation Search, a 64MB transposition
// table (Zobrist), killer moves, history heuristic and iterative deepening.
// Search mutates the board internally but restores it before returning.
class Search {
 public:
  explicit Search(uint64_t ttBytes = 64ull * 1024 * 1024);

  // Clears the transposition table and heuristics. Datagen calls this before
  // every move so a record's score depends only on (position, depth, VCF, TT
  // size) and can be reproduced offline by replaying that single position.
  void reset();

  // Selects the static evaluation backend for all subsequent searches.
  // Null keeps the handcrafted evaluator; the network is borrowed.
  void setNetwork(const NnueNetwork* network) { network_ = network; }

  SearchResult findBestMove(Board& board, const SearchLimits& limits);

 private:
  struct TTEntry {
    uint64_t key = 0;
    int32_t score = 0;
    int16_t move = -1;
    int8_t depth = 0;
    uint8_t flag = 0;
    uint8_t generation = 0;
  };

  int alphaBeta(Board& board, Evaluator& eval, int depth, int alpha, int beta,
                int ply, uint64_t hash, int ttMove);
  bool timeUp() const;

  std::vector<TTEntry> tt_;
  const NnueNetwork* network_ = nullptr;
  size_t ttMask_ = 0;
  uint8_t generation_ = 1;
  std::array<std::array<int, 2>, 64> killers_{};
  std::array<int, kCells> history_{};
  uint64_t nodes_ = 0;
  bool stop_ = false;
  bool useClock_ = false;
  int rootBestMove_ = -1;
  std::chrono::steady_clock::time_point deadline_{};
};

}  // namespace gomoku
