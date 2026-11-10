#include "renju.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace gomoku {
namespace renju {

namespace {

using Cells = std::array<Cell, kCells>;

struct Direction {
  int dx, dy;
};
constexpr Direction kDirections[4] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

bool inside(int x, int y) {
  return x >= 0 && x < kBoardSize && y >= 0 && y < kBoardSize;
}

// Maximal consecutive run of `c` through `move` along direction d.
int runLength(const Cells& cells, int move, Color c, Direction d) {
  const int x = Board::xOf(move);
  const int y = Board::yOf(move);
  const Cell target = cellOf(c);
  int n = 1;
  for (int i = 1; i < kBoardSize; ++i) {
    const int nx = x + d.dx * i;
    const int ny = y + d.dy * i;
    if (!inside(nx, ny) || cells[Board::indexOf(nx, ny)] != target) break;
    ++n;
  }
  for (int i = 1; i < kBoardSize; ++i) {
    const int nx = x - d.dx * i;
    const int ny = y - d.dy * i;
    if (!inside(nx, ny) || cells[Board::indexOf(nx, ny)] != target) break;
    ++n;
  }
  return n;
}

bool makesExactlyFive(const Cells& cells, int move, Color c) {
  for (const Direction& d : kDirections) {
    if (runLength(cells, move, c, d) == 5) return true;
  }
  return false;
}

bool makesOverline(const Cells& cells, int move, Color c) {
  for (const Direction& d : kDirections) {
    if (runLength(cells, move, c, d) >= 6) return true;
  }
  return false;
}

// A line segment of `len` cells containing `move`, starting `startOffset`
// cells before it along d. Returns false if the segment leaves the board.
bool segment(int move, Direction d, int len, int startOffset,
             std::array<int, 6>& out) {
  const int x = Board::xOf(move);
  const int y = Board::yOf(move);
  const int sx = x + d.dx * startOffset;
  const int sy = y + d.dy * startOffset;
  const int ex = sx + d.dx * (len - 1);
  const int ey = sy + d.dy * (len - 1);
  if (!inside(sx, sy) || !inside(ex, ey)) return false;
  for (int i = 0; i < len; ++i) {
    out[i] = Board::indexOf(sx + d.dx * i, sy + d.dy * i);
  }
  return true;
}

// Straight four: a maximal run of exactly four black stones through `move`
// along d, with both ends on-board and empty (i.e. _BBBB_).
bool isStraightFour(const Cells& cells, int move, Direction d) {
  if (runLength(cells, move, Color::Black, d) != 4) return false;
  int x = Board::xOf(move);
  int y = Board::yOf(move);
  while (inside(x - d.dx, y - d.dy) &&
         cells[Board::indexOf(x - d.dx, y - d.dy)] == Cell::Black) {
    x -= d.dx;
    y -= d.dy;
  }
  const int bx = x - d.dx;
  const int by = y - d.dy;
  const int ax = x + d.dx * 4;
  const int ay = y + d.dy * 4;
  return inside(bx, by) && cells[Board::indexOf(bx, by)] == Cell::Empty &&
         inside(ax, ay) && cells[Board::indexOf(ax, ay)] == Cell::Empty;
}

uint32_t signature(const int* idx, int n) {
  std::array<int, 6> a{};
  for (int i = 0; i < n; ++i) a[i] = idx[i];
  std::sort(a.begin(), a.begin() + n);
  uint32_t sig = 0;
  for (int i = 0; i < n; ++i) {
    sig = (sig << 8) | static_cast<uint32_t>(a[i]);
  }
  return sig;
}

void addUnique(std::vector<uint32_t>& seen, uint32_t sig) {
  if (std::find(seen.begin(), seen.end(), sig) == seen.end()) seen.push_back(sig);
}

// Distinct four-groups created by `move`: five-cell windows containing `move`
// with 4 black + 1 empty, where filling the empty makes exactly five.
// Two windows of a straight four share the same 4 stones -> deduplicated.
int countFours(const Cells& cells, int move) {
  std::vector<uint32_t> seen;
  for (const Direction& d : kDirections) {
    for (int s = -4; s <= 0; ++s) {
      std::array<int, 6> w{};
      if (!segment(move, d, 5, s, w)) continue;
      int blacks[6];
      int nb = 0;
      int empties[6];
      int ne = 0;
      bool has_white = false;
      for (int i = 0; i < 5; ++i) {
        const Cell c = cells[w[i]];
        if (c == Cell::Black) {
          blacks[nb++] = w[i];
        } else if (c == Cell::Empty) {
          empties[ne++] = w[i];
        } else {
          has_white = true;
        }
      }
      if (has_white || nb != 4 || ne != 1) continue;
      Cells copy = cells;
      copy[empties[0]] = Cell::Black;
      if (!makesExactlyFive(copy, empties[0], Color::Black)) continue;
      addUnique(seen, signature(blacks, 4));
    }
  }
  return static_cast<int>(seen.size());
}

// A three is "open" only if some empty extension point makes a straight four
// and that extension move is not itself a foul (overline / double-four).
// Recursion is intentionally cut here: double-three at the extension point is
// not evaluated, matching the Renju rule for false (nested) threes.
bool isFoulIgnoringThrees(const Cells& cells, int move) {
  Cells v = cells;
  v[move] = Cell::Black;
  if (makesExactlyFive(v, move, Color::Black)) return false;
  if (makesOverline(v, move, Color::Black)) return true;
  return countFours(v, move) >= 2;
}

// Distinct open-three groups created by `move`: six-cell windows with
// 3 black + 3 empty (no white), whose extension yields a legal straight four.
int countThrees(const Cells& cells, int move) {
  std::vector<uint32_t> seen;
  for (const Direction& d : kDirections) {
    for (int s = -5; s <= 0; ++s) {
      std::array<int, 6> w{};
      if (!segment(move, d, 6, s, w)) continue;
      int blacks[6];
      int nb = 0;
      int empties[6];
      int ne = 0;
      bool has_white = false;
      for (int i = 0; i < 6; ++i) {
        const Cell c = cells[w[i]];
        if (c == Cell::Black) {
          blacks[nb++] = w[i];
        } else if (c == Cell::Empty) {
          empties[ne++] = w[i];
        } else {
          has_white = true;
        }
      }
      if (has_white || nb != 3 || ne != 3) continue;
      bool valid = false;
      for (int i = 0; i < ne && !valid; ++i) {
        Cells copy = cells;
        copy[empties[i]] = Cell::Black;
        if (!isStraightFour(copy, empties[i], d)) continue;
        if (isFoulIgnoringThrees(cells, empties[i])) continue;
        valid = true;
      }
      if (valid) addUnique(seen, signature(blacks, 3));
    }
  }
  return static_cast<int>(seen.size());
}

}  // namespace

ForbiddenReason classifyForbidden(const Cells& cells, int move) {
  Cells v = cells;
  v[move] = Cell::Black;
  if (makesExactlyFive(v, move, Color::Black)) {
    return ForbiddenReason::None;  // five beats every foul
  }
  if (makesOverline(v, move, Color::Black)) return ForbiddenReason::Overline;
  if (countFours(v, move) >= 2) return ForbiddenReason::DoubleFour;
  if (countThrees(v, move) >= 2) return ForbiddenReason::DoubleThree;
  return ForbiddenReason::None;
}

bool isForbiddenMove(const Cells& cells, int move) {
  return classifyForbidden(cells, move) != ForbiddenReason::None;
}

bool isWin(const Cells& cells, int move, Color c, RuleMode mode) {
  if (mode == RuleMode::Renju && c == Color::Black) {
    return makesExactlyFive(cells, move, c);
  }
  for (const Direction& d : kDirections) {
    if (runLength(cells, move, c, d) >= 5) return true;
  }
  return false;
}

}  // namespace renju
}  // namespace gomoku
