// Board + random player unit tests (ticket 02). Zero-dependency: a failed
// CHECK prints the line and fails the run; green run exits 0.
#include <iostream>

#include "board.h"
#include "random_player.h"

static int g_failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::cerr << "FAIL line " << __LINE__ << ": " << #cond << "\n"; \
      ++g_failures;                                                    \
    }                                                                  \
  } while (0)

using namespace gomoku;

int main() {
  {
    // Fresh board.
    Board b;
    CHECK(b.sideToMove() == Color::Black);
    CHECK(b.result() == GameResult::Ongoing);
    CHECK(b.moveCount() == 0);
    bool all_empty = true;
    for (int m = 0; m < kCells; ++m) all_empty = all_empty && b.cell(m) == Cell::Empty;
    CHECK(all_empty);
  }
  {
    // Place / undo roundtrip.
    Board b;
    const int m = Board::indexOf(7, 7);
    b.place(m);
    CHECK(b.cell(m) == Cell::Black);
    CHECK(b.sideToMove() == Color::White);
    CHECK(b.moveCount() == 1);
    b.undo();
    CHECK(b.cell(m) == Cell::Empty);
    CHECK(b.sideToMove() == Color::Black);
    CHECK(b.moveCount() == 0);
    CHECK(b.result() == GameResult::Ongoing);
  }
  {
    // Horizontal five wins (black); undo restores Ongoing.
    Board b;
    for (int i = 0; i < 4; ++i) {
      b.place(Board::indexOf(3 + i, 7));
      b.place(Board::indexOf(3 + i, 8));
    }
    CHECK(b.result() == GameResult::Ongoing);
    b.place(Board::indexOf(7, 7));
    CHECK(b.result() == GameResult::BlackWin);
    b.undo();
    CHECK(b.result() == GameResult::Ongoing);
    b.place(Board::indexOf(7, 7));
    CHECK(b.result() == GameResult::BlackWin);
  }
  {
    // Vertical five wins (white); black stones kept off any line.
    const int bx[] = {0, 1, 2, 0, 1};
    const int by[] = {0, 0, 0, 1, 1};
    Board b;
    for (int i = 0; i < 5; ++i) {
      b.place(Board::indexOf(bx[i], by[i]));
      b.place(Board::indexOf(7, 3 + i));
    }
    CHECK(b.result() == GameResult::WhiteWin);
  }
  {
    // Diagonal five wins (black).
    Board b;
    for (int i = 0; i < 4; ++i) {
      b.place(Board::indexOf(2 + i, 2 + i));
      b.place(Board::indexOf(10 + i, 0));
    }
    b.place(Board::indexOf(6, 6));
    CHECK(b.result() == GameResult::BlackWin);
  }
  {
    // Anti-diagonal five wins (black).
    Board b;
    const int bx[] = {3, 4, 5, 6, 7};
    const int by[] = {11, 10, 9, 8, 7};
    for (int i = 0; i < 4; ++i) {
      b.place(Board::indexOf(bx[i], by[i]));
      b.place(Board::indexOf(i, 0));
    }
    b.place(Board::indexOf(7, 7));
    CHECK(b.result() == GameResult::BlackWin);
  }
  {
    // Overline (six in a row) also wins in freestyle semantics.
    Board b;
    const int wx[] = {0, 0, 0, 0, 1};
    const int wy[] = {0, 1, 2, 3, 0};
    for (int i = 0; i < 5; ++i) {
      b.place(Board::indexOf(5 + i, 5));
      b.place(Board::indexOf(wx[i], wy[i]));
    }
    b.place(Board::indexOf(10, 5));
    CHECK(b.result() == GameResult::BlackWin);
  }
  {
    // isPlayable rejects occupied and out-of-bounds.
    Board b;
    b.place(Board::indexOf(7, 7));
    CHECK(!b.isPlayable(Board::indexOf(7, 7)));
    CHECK(!b.isPlayable(-1));
    CHECK(!b.isPlayable(kCells));
    CHECK(b.isPlayable(Board::indexOf(0, 0)));
  }
  {
    // BOARD sync: full board with no five-in-a-row pattern -> Draw.
    // Pattern cell(x,y) = black iff (x + 2y) mod 5 < 2: every direction
    // cycles through the period-5 coloring, so runs stay <= 2.
    Board b;
    for (int y = 0; y < kBoardSize; ++y) {
      for (int x = 0; x < kBoardSize; ++x) {
        const int t = (x + 2 * y) % 5;
        b.setCell(Board::indexOf(x, y), t < 2 ? Cell::Black : Cell::White);
      }
    }
    b.finalizeSync();
    CHECK(b.moveCount() == kCells);
    CHECK(b.result() == GameResult::Draw);
  }
  {
    // BOARD sync: existing five -> BlackWin, count and side tracked.
    Board b;
    for (int i = 0; i < 5; ++i) b.setCell(Board::indexOf(3 + i, 3 + i), Cell::Black);
    b.setCell(Board::indexOf(0, 0), Cell::White);
    b.setCell(Board::indexOf(1, 0), Cell::White);
    b.setCell(Board::indexOf(0, 1), Cell::White);
    b.finalizeSync();
    CHECK(b.result() == GameResult::BlackWin);
    CHECK(b.moveCount() == 8);
  }
  {
    // Side-to-move inference from stone counts.
    Board b;
    b.setCell(Board::indexOf(0, 0), Cell::Black);
    b.setCell(Board::indexOf(1, 0), Cell::White);
    b.setCell(Board::indexOf(2, 0), Cell::Black);
    b.setCell(Board::indexOf(3, 0), Cell::White);
    b.finalizeSync();
    CHECK(b.sideToMove() == Color::Black);  // equal counts -> black
    b.setCell(Board::indexOf(4, 0), Cell::Black);
    b.finalizeSync();
    CHECK(b.sideToMove() == Color::White);  // black ahead -> white
    CHECK(b.result() == GameResult::Ongoing);
  }
  {
    // Random player: legal, deterministic per seed, -1 when full.
    Board b;
    RandomPlayer rp(42);
    const int m = rp.pickMove(b);
    CHECK(m >= 0 && Board::inBounds(m) && b.isPlayable(m));
    RandomPlayer rp2(42);
    CHECK(rp2.pickMove(b) == m);

    Board full;
    for (int y = 0; y < kBoardSize; ++y) {
      for (int x = 0; x < kBoardSize; ++x) {
        const int t = (x + 2 * y) % 5;
        full.setCell(Board::indexOf(x, y), t < 2 ? Cell::Black : Cell::White);
      }
    }
    full.setCell(Board::indexOf(0, 0), Cell::Empty);
    full.finalizeSync();
    RandomPlayer rp3(7);
    CHECK(rp3.pickMove(full) == Board::indexOf(0, 0));
    full.setCell(Board::indexOf(0, 0), Cell::Black);
    full.finalizeSync();
    CHECK(rp3.pickMove(full) == -1);
  }

  if (g_failures > 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "board_tests: all checks passed\n";
  return 0;
}