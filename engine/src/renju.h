#pragma once

#include <array>

#include "board.h"

namespace gomoku {
namespace renju {

// Why placing a BLACK stone at `move` would be a Renju foul. This is the
// single decision procedure: callers (Board::isPlayable, the executor, the
// GUI) must not implement a second verdict. Five takes precedence: a move
// forming exactly five is never a foul. cells[move] must be Empty.
enum class ForbiddenReason {
  None,
  Overline,     // 长连 (>= 6)
  DoubleFour,   // 四四
  DoubleThree,  // 三三
};

// Implementation notes (ticket 03):
// - Fours are counted as distinct four-stone groups, not windows: the two
//   five-windows of a straight four share the same four stones and count once.
// - An "open three" is only counted when some extension point makes a straight
//   four AND that extension move is itself not a foul (nested / false three).
//   Three counting never recurses further, so evaluation is bounded.
// - Analysis works on a 225-byte copy of the board; search (ticket 04) will
//   replace this with incremental bitboard logic if it becomes a bottleneck.
ForbiddenReason classifyForbidden(const std::array<Cell, kCells>& cells,
                                  int move);

// True iff classifyForbidden() != None. Kept as a thin wrapper so Board and
// the search path share exactly the same verdict as the reason reported by
// the executor.
bool isForbiddenMove(const std::array<Cell, kCells>& cells, int move);

// Win detection under a rule mode. In Renju black needs EXACTLY five
// (an overline is a foul and does not win); white and all freestyle wins
// accept five or more. Five wins even if it coincides with a foul pattern.
bool isWin(const std::array<Cell, kCells>& cells, int move, Color c, RuleMode mode);

}  // namespace renju
}  // namespace gomoku
