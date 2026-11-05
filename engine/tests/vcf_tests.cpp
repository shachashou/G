// VCF (Victory by Continuous Fours) tactic tests (ticket 05).
#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>

#include "board.h"
#include "vcf.h"

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

static Board makeBoard(std::initializer_list<Pt> blacks,
                       std::initializer_list<Pt> whites, Color stm,
                       RuleMode mode = RuleMode::Renju) {
  Board b(mode);
  for (const Pt& q : blacks) b.setCell(p(q.first, q.second), Cell::Black);
  for (const Pt& q : whites) b.setCell(p(q.first, q.second), Cell::White);
  b.finalizeSync();
  b.setSideToMove(stm);
  return b;
}

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

// A 5-ply VCF: (8,7) is a simple four forcing black to block at (9,7);
// then the diagonal (6,5)(7,6)(8,7) becomes an unblockable open four.
static const std::vector<Pt> kDeepBlacks = {{4, 7}, {0, 0}, {2, 0}, {4, 0}};
static const std::vector<Pt> kDeepWhites = {{5, 7}, {6, 7}, {7, 7},
                                           {6, 5}, {7, 6}};

int main() {
  {
    // Immediate five.
    Board b = makeBoard({{2, 7}, {0, 0}}, {{3, 7}, {4, 7}, {5, 7}, {6, 7}},
                        Color::White);
    CHECK(b.result() == GameResult::Ongoing);
    const VcfResult r = solveVcf(b);
    CHECK(r.found);
    CHECK(b.isPlayable(r.firstMove));
  }
  {
    // Open four: either end extension creates two winning points.
    Board b = makeBoard({{0, 0}}, {{5, 7}, {6, 7}, {7, 7}}, Color::White);
    const VcfResult r = solveVcf(b);
    CHECK(r.found);
    CHECK(r.firstMove == p(4, 7) || r.firstMove == p(8, 7));
    CHECK(b.isPlayable(r.firstMove));
  }
  {
    // Deep continuous-four sequence (needs >=5 plies; shallow Alpha-Beta
    // cannot see it).
    Board b = makeBoard({kDeepBlacks[0], kDeepBlacks[1], kDeepBlacks[2], kDeepBlacks[3]},
                  {kDeepWhites[0], kDeepWhites[1], kDeepWhites[2],
                   kDeepWhites[3], kDeepWhites[4]},
                  Color::White);
    CHECK(b.result() == GameResult::Ongoing);
    const auto before = b.rawCells();
    const VcfResult r = solveVcf(b);
    CHECK(r.found);
    CHECK(r.firstMove == p(8, 7));
    CHECK(b.isPlayable(r.firstMove));
    CHECK(b.rawCells() == before);  // board restored
    CHECK(b.moveCount() == 9);
  }
  {
    // Quiet position: no four sequence at all -> no false positive.
    Board b = makeBoard({{7, 8}, {0, 0}}, {{7, 7}, {8, 8}}, Color::White);
    const VcfResult r = solveVcf(b);
    CHECK(!r.found);
  }
  {
    // 禁手杀: white's simple four has a single winning point (7,7), but
    // black cannot block there -- it would be a double-four foul in Renju.
    Board b = makeBoard({{2, 7}, {5, 5}, {6, 6}, {8, 8},
                         {5, 9}, {6, 8}, {8, 6}},
                        {{3, 7}, {4, 7}, {5, 7}},
                        Color::White);
    CHECK(b.result() == GameResult::Ongoing);
    b.setSideToMove(Color::Black);
    CHECK(!b.isPlayable(p(7, 7)));  // double-four foul for black
    b.setSideToMove(Color::White);
    const VcfResult r = solveVcf(b);
    CHECK(r.found);
    CHECK(r.firstMove == p(6, 7));
  }
  {
    // Node/depth limits keep the solver bounded.
    Board b = makeBoard({{7, 8}, {0, 0}}, {{7, 7}, {8, 8}}, Color::White);
    const VcfResult r = solveVcf(b, VcfLimits{3, 1});
    CHECK(!r.found);
    CHECK(r.nodes <= 3);
  }
  {
    // D4 symmetry: the deep VCF survives all eight transforms.
    for (int rot = 0; rot < 4; ++rot) {
      for (bool mirror : {false, true}) {
        Board b(RuleMode::Renju);
        for (const Pt& q : kDeepBlacks) {
          const Pt t = xf(q, rot, mirror);
          b.setCell(p(t.first, t.second), Cell::Black);
        }
        for (const Pt& q : kDeepWhites) {
          const Pt t = xf(q, rot, mirror);
          b.setCell(p(t.first, t.second), Cell::White);
        }
        b.finalizeSync();
        b.setSideToMove(Color::White);
        const VcfResult r = solveVcf(b);
        CHECK(r.found);
        if (r.found) CHECK(b.isPlayable(r.firstMove));
      }
    }
  }

  if (g_failures > 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "vcf_tests: all checks passed\n";
  return 0;
}
