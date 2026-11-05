#include "board.h"

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
  // Renju forbidden-move exclusion for black arrives with ticket 03.
  return true;
}

bool Board::createsWin(const std::array<Cell, kCells>& cells, int move, Color c) {
  const int x = xOf(move);
  const int y = yOf(move);
  const Cell target = cellOf(c);
  for (const Direction& d : kDirections) {
    int count = 1;
    for (int i = 1; i < kBoardSize; ++i) {
      const int nx = x + d.dx * i;
      const int ny = y + d.dy * i;
      if (nx < 0 || nx >= kBoardSize || ny < 0 || ny >= kBoardSize) break;
      if (cells[indexOf(nx, ny)] != target) break;
      ++count;
    }
    for (int i = 1; i < kBoardSize; ++i) {
      const int nx = x - d.dx * i;
      const int ny = y - d.dy * i;
      if (nx < 0 || nx >= kBoardSize || ny < 0 || ny >= kBoardSize) break;
      if (cells[indexOf(nx, ny)] != target) break;
      ++count;
    }
    if (count >= 5) return true;
  }
  return false;
}

void Board::place(int move) {
  history_.push_back({move, result_});
  cells_[move] = cellOf(sideToMove_);
  ++moveCount_;
  if (createsWin(cells_, move, sideToMove_)) {
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
      if (count >= 5) {
        return c == Cell::Black ? GameResult::BlackWin : GameResult::WhiteWin;
      }
    }
  }
  if (moveCount_ == kCells) return GameResult::Draw;
  return GameResult::Ongoing;
}

}  // namespace gomoku
