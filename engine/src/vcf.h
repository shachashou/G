#pragma once

#include <cstdint>

#include "board.h"

namespace gomoku {

struct VcfResult {
  bool found = false;
  int firstMove = -1;  // first move of the winning sequence
  uint64_t nodes = 0;
};

struct VcfLimits {
  uint64_t nodeLimit = 5000;
  int maxAttackPlies = 10;
};

// Depth-first search for a forced win by CONTINUOUS FOURS (VCF) for the side
// to move. Each four has a single winning point, forcing the defender to
// block; the attacker then plays another four, until a five or an open four
// (two winning points, unblockable) appears. Renju fouls are respected:
// a four-making move must be playable, and a forbidden forced defense is a
// win for the attacker (禁手杀). The board is restored before returning.
VcfResult solveVcf(Board& board, const VcfLimits& limits = {});

}  // namespace gomoku
