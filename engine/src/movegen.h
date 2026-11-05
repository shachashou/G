#pragma once

#include <vector>

#include "board.h"
#include "eval.h"

namespace gomoku {

struct ScoredMove {
  int move;
  int score;
};

// Legal candidate moves: empty cells within Chebyshev `radius` of a stone,
// with Renju fouls excluded. Sorted by descending threat heuristic + a small
// central bias. On an empty board the center is the single candidate.
void generateCandidates(const Board& board, const Evaluator& eval,
                        std::vector<ScoredMove>& out, int radius = 1);

}  // namespace gomoku
