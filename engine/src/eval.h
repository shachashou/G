#pragma once

#include <array>

#include "board.h"

namespace gomoku {

// Handcrafted threat evaluator (ticket 04).
// Sums, over every 5-cell window in all four directions, a pattern score for
// each color. A window containing both colors is dead and scores zero.
// Scores are maintained incrementally: only the <=20 windows through a move
// change when a stone is placed or removed.
class Evaluator {
 public:
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
};

}  // namespace gomoku
