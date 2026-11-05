#include "movegen.h"

#include <algorithm>
#include <array>

namespace gomoku {

void generateCandidates(const Board& board, const Evaluator& eval,
                        std::vector<ScoredMove>& out, int radius) {
  out.clear();
  constexpr int kCenter = kBoardSize / 2;
  std::array<bool, kCells> near{};
  bool anyStone = false;
  for (int m = 0; m < kCells; ++m) {
    if (board.cell(m) == Cell::Empty) continue;
    anyStone = true;
    const int x = Board::xOf(m);
    const int y = Board::yOf(m);
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        const int nx = x + dx;
        const int ny = y + dy;
        if (nx < 0 || nx >= kBoardSize || ny < 0 || ny >= kBoardSize) continue;
        near[Board::indexOf(nx, ny)] = true;
      }
    }
  }
  if (!anyStone) {
    out.push_back({Board::indexOf(kCenter, kCenter), 0});
    return;
  }

  const Color stm = board.sideToMove();
  for (int m = 0; m < kCells; ++m) {
    if (!near[m] || !board.isPlayable(m)) continue;
    const int dx = Board::xOf(m) - kCenter;
    const int dy = Board::yOf(m) - kCenter;
    const int score = eval.moveHeuristic(board, m, stm) + 10 - (dx * dx + dy * dy) / 10;
    out.push_back({m, score});
  }
  std::sort(out.begin(), out.end(),
            [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });
}

}  // namespace gomoku
