// Renju forbidden-move unit tests (ticket 03).
// Positions are constructed from classic problem shapes: overline,
// double-four, double-three, nested (false) three, five precedence,
// D4 symmetry, rule switching and white-side asymmetry.
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

#include "board.h"

static int g_failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::cerr << "FAIL line " << __LINE__ << ": " << #cond << "\n"; \
      ++g_failures;                                                    \
    }                                                                  \
  } while (0)

using namespace gomoku;
using Pt = std::pair<int, int>;

static int p(int x, int y) { return Board::indexOf(x, y); }

static Board makeBoard(RuleMode mode, std::initializer_list<Pt> blacks,
                       std::initializer_list<Pt> whites,
                       Color stm = Color::Black) {
  Board b(mode);
  for (const Pt& q : blacks) b.setCell(p(q.first, q.second), Cell::Black);
  for (const Pt& q : whites) b.setCell(p(q.first, q.second), Cell::White);
  b.finalizeSync();
  b.setSideToMove(stm);
  return b;
}

// D4 transforms: 4 rotations x (optional) mirror, all preserve Renju rules.
static Pt xf(Pt q, int rot, bool mirror) {
  int x = q.first;
  int y = q.second;
  if (mirror) x = kBoardSize - 1 - x;
  for (int i = 0; i < rot; ++i) {
    const int nx = y;
    const int ny = kBoardSize - 1 - x;
    x = nx;
    y = ny;
  }
  return {x, y};
}

static Board transformBoard(const std::vector<Pt>& blacks,
                            const std::vector<Pt>& whites, int rot,
                            bool mirror, Pt& transformedProbe) {
  Board b(RuleMode::Renju);
  for (const Pt& q : blacks) {
    const Pt t = xf(q, rot, mirror);
    b.setCell(p(t.first, t.second), Cell::Black);
  }
  for (const Pt& q : whites) {
    const Pt t = xf(q, rot, mirror);
    b.setCell(p(t.first, t.second), Cell::White);
  }
  transformedProbe = xf({7, 7}, rot, mirror);
  b.finalizeSync();
  b.setSideToMove(Color::Black);
  return b;
}

// A crossing of two straight fours -> double-four foul.
static const std::vector<Pt> kDoubleFourBlacks = {{5, 7}, {6, 7}, {8, 7},
                                                  {7, 5}, {7, 6}, {7, 8}};
// A single straight four -> legal.
static const std::vector<Pt> kStraightFourBlacks = {{5, 7}, {6, 7}, {8, 7}};
static const std::vector<Pt> kFarWhites = {{0, 0}, {2, 0}};

int main() {
  {
    // Overline: placing at (6,7) makes six in a row -> foul in Renju,
    // legal winning move in freestyle.
    const std::initializer_list<Pt> blacks = {{3, 7}, {4, 7}, {5, 7},
                                              {7, 7}, {8, 7}};
    auto b = makeBoard(RuleMode::Renju, blacks, {{0, 0}, {2, 0}});
    CHECK(!b.isPlayable(p(6, 7)));
    auto f = makeBoard(RuleMode::Freestyle, blacks, {{0, 0}, {2, 0}});
    CHECK(f.isPlayable(p(6, 7)));
  }
  {
    // Five takes precedence: (7,7) makes exactly five horizontally plus
    // two straight fours (vertical + diagonal). Legal, and black wins.
    auto b = makeBoard(RuleMode::Renju,
                       {{5, 7}, {6, 7}, {8, 7}, {9, 7},
                        {7, 5}, {7, 6}, {7, 8},
                        {5, 5}, {6, 6}, {8, 8}},
                       {{0, 0}, {2, 0}});
    CHECK(b.isPlayable(p(7, 7)));
    b.place(p(7, 7));
    CHECK(b.result() == GameResult::BlackWin);
  }
  {
    // Double four: two straight fours cross at (7,7).
    auto b = makeBoard(RuleMode::Renju, {{5, 7}, {6, 7}, {8, 7},
                                        {7, 5}, {7, 6}, {7, 8}},
                       {{0, 0}, {2, 0}});
    CHECK(!b.isPlayable(p(7, 7)));
    auto f = makeBoard(RuleMode::Freestyle, {{5, 7}, {6, 7}, {8, 7},
                                            {7, 5}, {7, 6}, {7, 8}},
                       {{0, 0}, {2, 0}});
    CHECK(f.isPlayable(p(7, 7)));
  }
  {
    // A single straight four is NOT a double-four foul.
    auto b = makeBoard(RuleMode::Renju, {{5, 7}, {6, 7}, {8, 7}}, {{0, 0}});
    CHECK(b.isPlayable(p(7, 7)));
  }
  {
    // Double three: two open threes cross at (7,7).
    auto b = makeBoard(RuleMode::Renju,
                       {{6, 7}, {8, 7}, {7, 6}, {7, 8}},
                       {{0, 0}, {2, 0}});
    CHECK(!b.isPlayable(p(7, 7)));
  }
  {
    // Sleeping three (one end blocked by white) is not an open three:
    // only one valid three remains -> legal.
    auto b = makeBoard(RuleMode::Renju,
                       {{6, 7}, {8, 7}, {7, 6}, {7, 8}},
                       {{9, 7}, {0, 0}});
    CHECK(b.isPlayable(p(7, 7)));
  }
  {
    // Double JUMP three: two 跳活三 (B_BB shapes) cross at (7,7).
    auto b = makeBoard(RuleMode::Renju,
                       {{5, 7}, {8, 7}, {7, 4}, {7, 6}},
                       {{0, 0}, {2, 0}});
    CHECK(!b.isPlayable(p(7, 7)));
  }
  {
    // A single jump three is legal.
    auto b = makeBoard(RuleMode::Renju, {{5, 7}, {8, 7}}, {{0, 0}});
    CHECK(b.isPlayable(p(7, 7)));
  }
  {
    // Four-three (四三) is legal: one four + one open three is not a foul.
    auto b = makeBoard(RuleMode::Renju,
                       {{5, 7}, {6, 7}, {8, 7}, {7, 6}, {7, 8}},
                       {{0, 0}, {2, 0}});
    CHECK(b.isPlayable(p(7, 7)));
  }
  {
    // Nested / false three (嵌跳三): the horizontal "three" has NO legal
    // four-making move -- both extensions (5,7) and (9,7) would create a
    // double-four because of the vertical stones in those columns.
    // Only the vertical three counts, so (7,7) is legal.
    auto b = makeBoard(RuleMode::Renju,
                       {{6, 7}, {8, 7}, {7, 6}, {7, 8},
                        {5, 5}, {5, 6}, {5, 8},
                        {9, 5}, {9, 6}, {9, 8}},
                       {{0, 0}, {2, 0}});
    CHECK(b.isPlayable(p(7, 7)));
  }
  {
    // White has no fouls at all: the same double-four shape is legal.
    auto b = makeBoard(RuleMode::Renju, {{0, 0}, {2, 0}},
                       {{5, 7}, {6, 7}, {8, 7}, {7, 5}, {7, 6}, {7, 8}},
                       Color::White);
    CHECK(b.isPlayable(p(7, 7)));
  }
  {
    // White overline wins in Renju; black stones stay off any line.
    Board b(RuleMode::Renju);
    const int bx[] = {0, 1, 2, 0, 1, 0};
    const int by[] = {0, 0, 0, 1, 1, 2};
    for (int i = 0; i < 6; ++i) {
      b.place(p(bx[i], by[i]));
      b.place(p(3 + i, 7));
    }
    CHECK(b.result() == GameResult::WhiteWin);
  }
  {
    // Black overline is a foul, not a win (BOARD-sync scan path).
    Board b(RuleMode::Renju);
    for (int x = 3; x <= 8; ++x) b.setCell(p(x, 7), Cell::Black);
    b.setCell(p(0, 0), Cell::White);
    b.finalizeSync();
    CHECK(b.result() == GameResult::Ongoing);
    Board f(RuleMode::Freestyle);
    for (int x = 3; x <= 8; ++x) f.setCell(p(x, 7), Cell::Black);
    f.setCell(p(0, 0), Cell::White);
    f.finalizeSync();
    CHECK(f.result() == GameResult::BlackWin);
  }
  {
    // INFO rule switching takes effect at runtime on the same position.
    auto b = makeBoard(RuleMode::Renju, {{5, 7}, {6, 7}, {8, 7},
                                        {7, 5}, {7, 6}, {7, 8}},
                       {{0, 0}, {2, 0}});
    CHECK(!b.isPlayable(p(7, 7)));
    b.setRuleMode(RuleMode::Freestyle);
    CHECK(b.isPlayable(p(7, 7)));
    b.setRuleMode(RuleMode::Renju);
    CHECK(!b.isPlayable(p(7, 7)));
  }
  {
    // D4 symmetry: forbidden stays forbidden, legal stays legal under all
    // eight rotations/mirrors.
    for (int rot = 0; rot < 4; ++rot) {
      for (bool mirror : {false, true}) {
        Pt q;
        auto b = transformBoard(kDoubleFourBlacks, kFarWhites, rot, mirror, q);
        CHECK(!b.isPlayable(p(q.first, q.second)));
        auto f = transformBoard(kStraightFourBlacks, kFarWhites, rot, mirror, q);
        CHECK(f.isPlayable(p(q.first, q.second)));
      }
    }
  }

  if (g_failures > 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "renju_tests: all checks passed\n";
  return 0;
}
