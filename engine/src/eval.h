#pragma once

#include <array>

#include "board.h"
#include "nnue.h"

namespace gomoku {

// Handcrafted threat evaluator (ticket 04).
// Sums, over every 5-cell window in all four directions, a pattern score for
// each color. A window containing both colors is dead and scores zero.
// Scores are maintained incrementally: only the <=20 windows through a move
// change when a stone is placed or removed.
class Evaluator {
 public:
  // Selects the static evaluation backend. A null pointer (default) keeps the
  // handcrafted threat evaluator, which also keeps datagen deterministic.
  // The network is borrowed and must outlive the Evaluator.
  void setNetwork(const NnueNetwork* network) { network_ = network; }
  const NnueNetwork* network() const { return network_; }

  // Full recompute (search root / BOARD sync).
  void init(const Board& board);

  // `board` must be in the state BEFORE the change; `to` is the new content
  // of `move` (a stone when placing, Empty when undoing).
  void update(const Board& board, int move, Cell to);

  // Static exchange estimate for move ordering: my gain minus opponent gain.
  int moveHeuristic(const Board& board, int move, Color c) const;

  // Score from the side-to-move's perspective (negamax convention).
  int evaluate(const Board& board) const;

  int blackScore() const { return blackScore_; }
  int whiteScore() const { return whiteScore_; }

 private:
  int blackScore_ = 0;
  int whiteScore_ = 0;
  const NnueNetwork* network_ = nullptr;
  // Exact (unsaturated) int32 accumulator, incrementally maintained by
  // update() so the NNUE backend stays scalar but cheap during search.
  NnueAccumulator nnueAcc_{};
  // v3 (float16 weights) maintains a float32 accumulator instead.
  NnueFloatAccumulator nnueAccF_{};
};

}  // namespace gomoku
