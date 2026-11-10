#include "board.h"

#include "renju.h"

namespace gomoku {

namespace {

struct Direction {
  int dx, dy;
};
constexpr Direction kDirections[4] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

}  // namespace

Board::Board(RuleMode mode) {
  reset(mode);
}

void Board::reset(RuleMode mode) {
  cells_.fill(Cell::Empty);
  sideToMove_ = Color::Black;
  mode_ = mode;
  result_ = GameResult::Ongoing;
  moveCount_ = 0;
  history_.clear();
}

bool Board::isPlayable(int move) const {
  if (!inBounds(move) || cells_[move] != Cell::Empty) return false;
  // Renju: black's forbidden points are simply not legal moves.
  if (mode_ == RuleMode::Renju && sideToMove_ == Color::Black) {
    return !renju::isForbiddenMove(cells_, move);
  }
  return true;
}

bool Board::createsWin(const std::array<Cell, kCells>& cells, int move, Color c) {
  // Freestyle semantics (five or more). Renju-aware checks live in renju.h.
  return renju::isWin(cells, move, c, RuleMode::Freestyle);
}

void Board::place(int move) {
  history_.push_back({move, result_});
  cells_[move] = cellOf(sideToMove_);
  ++moveCount_;
  if (renju::isWin(cells_, move, sideToMove_, mode_)) {
    result_ = sideToMove_ == Color::Black ? GameResult::BlackWin : GameResult::WhiteWin;
  } else if (moveCount_ == kCells) {
    result_ = GameResult::Draw;
  }
  sideToMove_ = opponent(sideToMove_);
}

void Board::undo() {
  const HistoryEntry e = history_.back();
  history_.pop_back();
  cells_[e.move] = Cell::Empty;
  --moveCount_;
  result_ = e.resultBefore;
  sideToMove_ = opponent(sideToMove_);
}

std::vector<int> Board::moveHistory() const {
  std::vector<int> out;
  out.reserve(history_.size());
  for (const HistoryEntry& e : history_) out.push_back(e.move);
  return out;
}

void Board::setCell(int move, Cell c) {
  cells_[move] = c;
}

void Board::finalizeSync() {
  int black = 0;
  int white = 0;
  for (Cell c : cells_) {
    if (c == Cell::Black) {
      ++black;
    } else if (c == Cell::White) {
      ++white;
    }
  }
  moveCount_ = black + white;
  // Black moves first: equal counts -> black to move, else white to move.
  sideToMove_ = black == white ? Color::Black : Color::White;
  result_ = computeResult();
  history_.clear();
}

GameResult Board::computeResult() const {
  for (int m = 0; m < kCells; ++m) {
    const Cell c = cells_[m];
    if (c == Cell::Empty) continue;
    const int x = xOf(m);
    const int y = yOf(m);
    for (const Direction& d : kDirections) {
      // Only count runs starting at m to avoid re-visiting.
      const int px = x - d.dx;
      const int py = y - d.dy;
      if (px >= 0 && px < kBoardSize && py >= 0 && py < kBoardSize &&
          cells_[indexOf(px, py)] == c) {
        continue;
      }
      int count = 0;
      int nx = x;
      int ny = y;
      while (nx >= 0 && nx < kBoardSize && ny >= 0 && ny < kBoardSize &&
             cells_[indexOf(nx, ny)] == c) {
        ++count;
        nx += d.dx;
        ny += d.dy;
      }
      // Renju: black wins only with EXACTLY five (overline is a foul);
      // white and all freestyle wins accept five or more.
      const bool win = (mode_ == RuleMode::Renju && c == Cell::Black)
                           ? (count == 5)
                           : (count >= 5);
      if (win) {
        return c == Cell::Black ? GameResult::BlackWin : GameResult::WhiteWin;
      }
    }
  }
  if (moveCount_ == kCells) return GameResult::Draw;
  return GameResult::Ongoing;
}

}  // namespace gomoku
