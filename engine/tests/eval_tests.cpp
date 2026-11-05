// Incremental evaluation consistency tests (ticket 04).
#include <iostream>

#include "board.h"
#include "eval.h"

static int g_failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::cerr << "FAIL line " << __LINE__ << ": " << #cond << "\n"; \
      ++g_failures;                                                    \
    }                                                                  \
  } while (0)

using namespace gomoku;

static int p(int x, int y) { return Board::indexOf(x, y); }

int main() {
  {
    // Incremental place/undo must match a full recompute at every step.
    Board b(RuleMode::Freestyle);
    Evaluator e;
    e.init(b);
    const int moves[] = {p(7, 7), p(7, 8), p(8, 7), p(6, 8), p(8, 8),
                         p(5, 5), p(9, 9), p(6, 6), p(10, 10), p(4, 4)};
    constexpr int kCount = sizeof(moves) / sizeof(moves[0]);
    for (int i = 0; i < kCount; ++i) {
      const Color c = b.sideToMove();
      const int myBefore = c == Color::Black ? e.blackScore() - e.whiteScore()
                                             : e.whiteScore() - e.blackScore();
      const int h = e.moveHeuristic(b, moves[i], c);
      e.update(b, moves[i], cellOf(c));
      b.place(moves[i]);
      const int myAfter = c == Color::Black ? e.blackScore() - e.whiteScore()
                                            : e.whiteScore() - e.blackScore();
      CHECK(h == myAfter - myBefore);  // ordering heuristic == real delta
      Evaluator fresh;
      fresh.init(b);
      CHECK(e.blackScore() == fresh.blackScore());
      CHECK(e.whiteScore() == fresh.whiteScore());
    }
    for (int i = kCount - 1; i >= 0; --i) {
      e.update(b, moves[i], Cell::Empty);
      b.undo();
      Evaluator fresh;
      fresh.init(b);
      CHECK(e.blackScore() == fresh.blackScore());
      CHECK(e.whiteScore() == fresh.whiteScore());
    }
    CHECK(e.blackScore() == 0);
    CHECK(e.whiteScore() == 0);
  }
  {
    // Placing a stone next to the opponent kills their open windows.
    Board b(RuleMode::Freestyle);
    b.setCell(p(7, 7), Cell::White);
    b.setCell(p(8, 7), Cell::White);
    b.finalizeSync();
    b.setSideToMove(Color::Black);
    Evaluator e;
    e.init(b);
    const int before = e.whiteScore();
    CHECK(before > 0);
    e.update(b, p(9, 7), Cell::Black);
    b.setCell(p(9, 7), Cell::Black);
    b.finalizeSync();
    Evaluator fresh;
    fresh.init(b);
    CHECK(e.whiteScore() == fresh.whiteScore());
    CHECK(e.whiteScore() < before);
  }
  {
    // evaluate() is from the side-to-move's perspective.
    Board b(RuleMode::Freestyle);
    b.setCell(p(7, 7), Cell::Black);
    b.finalizeSync();
    Evaluator e;
    e.init(b);
    b.setSideToMove(Color::Black);
    const int asBlack = e.evaluate(b);
    b.setSideToMove(Color::White);
    const int asWhite = e.evaluate(b);
    CHECK(asBlack > 0);
    CHECK(asWhite < 0);
  }

  if (g_failures > 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "eval_tests: all checks passed\n";
  return 0;
}
