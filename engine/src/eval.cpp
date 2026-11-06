#include "eval.h"

namespace gomoku {

namespace {

struct Direction {
  int dx, dy;
};
constexpr Direction kDirections[4] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

// Score for n own stones in an uncontested five-cell window.
// An open four spans two such windows, an open three up to three.
constexpr int kPattern[6] = {0, 1, 12, 600, 15000, 500000};

using Window = std::array<int, 5>;

int windowValue(const std::array<Cell, kCells>& cells, const Window& w,
                Color c, int overrideMove, Cell overrideCell) {
  int black = 0;
  int white = 0;
  for (int idx : w) {
    const Cell cell = (idx == overrideMove) ? overrideCell : cells[idx];
    if (cell == Cell::Black) {
      ++black;
    } else if (cell == Cell::White) {
      ++white;
    }
  }
  if (c == Color::Black) return white == 0 ? kPattern[black] : 0;
  return black == 0 ? kPattern[white] : 0;
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
      Window w{};
      for (int i = 0; i < 5; ++i) {
        w[i] = Board::indexOf(sx + d.dx * i, sy + d.dy * i);
      }
      f(w);
    }
  }
}

template <typename F>
void forEachWindowOnBoard(F&& f) {
  for (const Direction& d : kDirections) {
    for (int y = 0; y < kBoardSize; ++y) {
      for (int x = 0; x < kBoardSize; ++x) {
        const int ex = x + d.dx * 4;
        const int ey = y + d.dy * 4;
        if (ex < 0 || ex >= kBoardSize || ey < 0 || ey >= kBoardSize) continue;
        Window w{};
        for (int i = 0; i < 5; ++i) {
          w[i] = Board::indexOf(x + d.dx * i, y + d.dy * i);
        }
        f(w);
      }
    }
  }
}

}  // namespace

void Evaluator::init(const Board& board) {
  blackScore_ = 0;
  whiteScore_ = 0;
  const auto& cells = board.rawCells();
  forEachWindowOnBoard([&](const Window& w) {
    blackScore_ += windowValue(cells, w, Color::Black, -1, Cell::Empty);
    whiteScore_ += windowValue(cells, w, Color::White, -1, Cell::Empty);
  });
  if (network_) {
    if (network_->formatVersion() == 3) {
      network_->initAccumulator(board, nnueAccF_);
    } else {
      network_->initAccumulator(board, nnueAcc_);
    }
  }
}

void Evaluator::update(const Board& board, int move, Cell to) {
  const auto& cells = board.rawCells();
  forEachWindowThrough(move, [&](const Window& w) {
    const int b0 = windowValue(cells, w, Color::Black, -1, Cell::Empty);
    const int s0 = windowValue(cells, w, Color::White, -1, Cell::Empty);
    const int b1 = windowValue(cells, w, Color::Black, move, to);
    const int s1 = windowValue(cells, w, Color::White, move, to);
    blackScore_ += b1 - b0;
    whiteScore_ += s1 - s0;
  });
  if (network_) {
    if (network_->formatVersion() == 3) {
      network_->updateAccumulator(board, move, to, nnueAccF_);
    } else {
      network_->updateAccumulator(board, move, to, nnueAcc_);
    }
  }
}

int Evaluator::moveHeuristic(const Board& board, int move, Color c) const {
  const auto& cells = board.rawCells();
  const Cell stone = cellOf(c);
  int dBlack = 0;
  int dWhite = 0;
  forEachWindowThrough(move, [&](const Window& w) {
    dBlack += windowValue(cells, w, Color::Black, move, stone) -
              windowValue(cells, w, Color::Black, -1, Cell::Empty);
    dWhite += windowValue(cells, w, Color::White, move, stone) -
              windowValue(cells, w, Color::White, -1, Cell::Empty);
  });
  return c == Color::Black ? (dBlack - dWhite) : (dWhite - dBlack);
}

int Evaluator::evaluate(const Board& board) const {
  if (network_) {
    // The network's head is selected by the side to move and its output is
    // already in the side-to-move/negamax convention used by the search.
    const int stm = board.sideToMove() == Color::Black ? 0 : 1;
    if (network_->formatVersion() == 3) {
      return network_->inferAccumulator(nnueAccF_, stm);
    }
    return network_->inferAccumulator(nnueAcc_, stm);
  }
  const int my = board.sideToMove() == Color::Black ? blackScore_ : whiteScore_;
  const int opp = board.sideToMove() == Color::Black ? whiteScore_ : blackScore_;
  // Small tempo bonus for the side to move.
  return (my * 115) / 100 - opp;
}

}  // namespace gomoku
