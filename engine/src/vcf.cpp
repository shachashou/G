#include "vcf.h"

#include <array>
#include <vector>

namespace gomoku {

namespace {

using Cells = std::array<Cell, kCells>;

struct Direction {
  int dx, dy;
};
constexpr Direction kDirections[4] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

bool inside(int x, int y) {
  return x >= 0 && x < kBoardSize && y >= 0 && y < kBoardSize;
}

// Would placing `c` at empty point `e` win immediately?
bool wouldWin(const Cells& cells, int e, Color c, RuleMode mode) {
  const Cell target = cellOf(c);
  const int x = Board::xOf(e);
  const int y = Board::yOf(e);
  for (const Direction& d : kDirections) {
    int count = 1;  // the stone at e itself
    for (int s = 1; s < kBoardSize; ++s) {
      const int nx = x + d.dx * s;
      const int ny = y + d.dy * s;
      if (!inside(nx, ny) || cells[Board::indexOf(nx, ny)] != target) break;
      ++count;
    }
    for (int s = 1; s < kBoardSize; ++s) {
      const int nx = x - d.dx * s;
      const int ny = y - d.dy * s;
      if (!inside(nx, ny) || cells[Board::indexOf(nx, ny)] != target) break;
      ++count;
    }
    if (mode == RuleMode::Renju && c == Color::Black) {
      if (count == 5) return true;  // black needs EXACTLY five
    } else if (count >= 5) {
      return true;
    }
  }
  return false;
}

// Distinct empty points where `c` would win on the next move.
int winningPoints(const Board& board, Color c, int* out, int maxOut) {
  const Cells& cells = board.rawCells();
  const Cell target = cellOf(c);
  std::array<bool, kCells> near{};
  for (int m = 0; m < kCells; ++m) {
    if (cells[m] != target) continue;
    const int x = Board::xOf(m);
    const int y = Board::yOf(m);
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx;
        const int ny = y + dy;
        if (inside(nx, ny)) near[Board::indexOf(nx, ny)] = true;
      }
    }
  }
  int n = 0;
  for (int e = 0; e < kCells; ++e) {
    if (!near[e] || cells[e] != Cell::Empty) continue;
    if (wouldWin(cells, e, c, board.ruleMode())) {
      if (n < maxOut) out[n] = e;
      ++n;
    }
  }
  return n;
}

template <typename F>
void forEachWindowThrough(int move, F&& f) {
  const int x = Board::xOf(move);
  const int y = Board::yOf(move);
  for (const Direction& d : kDirections) {
    for (int s = -4; s <= 0; ++s) {
      const int sx = x + d.dx * s;
      const int sy = y + d.dy * s;
      const int ex = sx + d.dx * 4;
      const int ey = sy + d.dy * 4;
      if (sx < 0 || sx >= kBoardSize || sy < 0 || sy >= kBoardSize ||
          ex < 0 || ex >= kBoardSize || ey < 0 || ey >= kBoardSize) {
        continue;
      }
      std::array<int, 5> w{};
      for (int i = 0; i < 5; ++i) {
        w[i] = Board::indexOf(sx + d.dx * i, sy + d.dy * i);
      }
      f(w);
    }
  }
}

// Cheap prefilter: does placing `c` at `move` leave a five-window with 4 own
// stones + 1 empty (i.e. a four)? Avoids the expensive global scan for most
// candidate moves.
bool createsFour(const Cells& cells, int move, Color c) {
  const Cell own = cellOf(c);
  bool four = false;
  forEachWindowThrough(move, [&](const std::array<int, 5>& w) {
    int ownCount = 0;
    int oppCount = 0;
    int emptyCount = 0;
    for (int idx : w) {
      const Cell cell = idx == move ? own : cells[idx];
      if (cell == own) {
        ++ownCount;
      } else if (cell == Cell::Empty) {
        ++emptyCount;
      } else {
        ++oppCount;
      }
    }
    if (ownCount == 4 && oppCount == 0 && emptyCount == 1) four = true;
  });
  return four;
}

void collectCandidates(const Board& board, std::vector<int>& out) {
  const Cells& cells = board.rawCells();
  std::array<bool, kCells> near{};
  for (int m = 0; m < kCells; ++m) {
    if (cells[m] == Cell::Empty) continue;
    const int x = Board::xOf(m);
    const int y = Board::yOf(m);
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = x + dx;
        const int ny = y + dy;
        if (inside(nx, ny)) near[Board::indexOf(nx, ny)] = true;
      }
    }
  }
  out.clear();
  for (int m = 0; m < kCells; ++m) {
    if (near[m] && board.isPlayable(m)) out.push_back(m);
  }
}

bool vcfRecurse(Board& board, Color attacker, int pliesLeft, uint64_t& nodes,
                const VcfLimits& limits, int* firstMove) {
  if (pliesLeft <= 0 || nodes >= limits.nodeLimit) return false;
  std::vector<int> candidates;
  collectCandidates(board, candidates);
  const GameResult attackerWin =
      attacker == Color::Black ? GameResult::BlackWin : GameResult::WhiteWin;

  for (int move : candidates) {
    if (nodes >= limits.nodeLimit) return false;
    ++nodes;
    const Cells& cells = board.rawCells();
    if (!createsFour(cells, move, attacker) &&
        !wouldWin(cells, move, attacker, board.ruleMode())) {
      continue;
    }
    board.place(move);
    bool win = false;
    if (board.result() == attackerWin) {
      win = true;  // immediate five
    } else {
      int wp[16];
      const int n = winningPoints(board, attacker, wp, 16);
      if (n >= 2) {
        win = true;  // open four / double four: defender cannot cover both
      } else if (n == 1) {
        int dp[16];
        const int dn = winningPoints(board, opponent(attacker), dp, 16);
        if (dn == 0) {  // defender has no immediate five of their own
          const int block = wp[0];
          if (!board.isPlayable(block)) {
            win = true;  // Renju: the forced defense is a forbidden point
          } else {
            board.place(block);
            if (board.result() == GameResult::Ongoing) {
              win = vcfRecurse(board, attacker, pliesLeft - 1, nodes, limits,
                               nullptr);
            }
            board.undo();
          }
        }
      }
    }
    board.undo();
    if (win) {
      if (firstMove) *firstMove = move;
      return true;
    }
  }
  return false;
}

}  // namespace

VcfResult solveVcf(Board& board, const VcfLimits& limits) {
  VcfResult result;
  if (board.result() != GameResult::Ongoing) return result;
  uint64_t nodes = 0;
  int firstMove = -1;
  if (vcfRecurse(board, board.sideToMove(), limits.maxAttackPlies, nodes,
                 limits, &firstMove)) {
    result.found = true;
    result.firstMove = firstMove;
  }
  result.nodes = nodes;
  return result;
}

}  // namespace gomoku
