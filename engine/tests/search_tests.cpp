// Search tactical behaviour tests (ticket 04).
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <utility>

#include "board.h"
#include "search.h"

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
                       std::initializer_list<Pt> whites, Color stm) {
  Board b(RuleMode::Renju);
  for (const Pt& q : blacks) b.setCell(p(q.first, q.second), Cell::Black);
  for (const Pt& q : whites) b.setCell(p(q.first, q.second), Cell::White);
  b.finalizeSync();
  b.setSideToMove(stm);
  return b;
}

int main() {
  {
    // Immediate win: black has an open four, search must complete five.
    Board b = makeBoard({{5, 7}, {6, 7}, {7, 7}, {8, 7}},
                        {{0, 0}, {2, 0}, {4, 0}, {6, 0}}, Color::Black);
    Search s(1 << 20);
    const SearchResult r = s.findBestMove(b, SearchLimits{3, 0});
    CHECK(r.move == p(4, 7) || r.move == p(9, 7));
    CHECK(r.nodes > 0);
  }
  {
    // Must block: white has a single four whose only winning point is (4,7)
    // ((9,7) is occupied by black). Any other black move loses next turn.
    Board b = makeBoard({{9, 7}}, {{5, 7}, {6, 7}, {7, 7}, {8, 7}},
                        Color::Black);
    Search s(1 << 20);
    const SearchResult r = s.findBestMove(b, SearchLimits{3, 0});
    CHECK(r.move == p(4, 7));
  }
  {
    // Empty board: legal move, deterministic, board fully restored.
    Board b;
    const auto before = b.rawCells();
    Search s1(1 << 20);
    const SearchResult r1 = s1.findBestMove(b, SearchLimits{3, 0});
    CHECK(r1.move >= 0 && b.isPlayable(r1.move));
    CHECK(b.rawCells() == before);
    CHECK(b.moveCount() == 0);
    CHECK(r1.nodes > 0);
    Search s2(1 << 20);
    const SearchResult r2 = s2.findBestMove(b, SearchLimits{3, 0});
    CHECK(r1.move == r2.move);
  }
  {
    // A short time budget must still return a legal move (no hang).
    Board b;
    Search s(1 << 20);
    const SearchResult r = s.findBestMove(b, SearchLimits{8, 20});
    CHECK(r.move >= 0 && b.isPlayable(r.move));
    CHECK(b.moveCount() == 0);
  }

  {
    // VCF A/B: a 5-ply continuous-four win is found at the root even at
    // depth 2, while the M2 baseline (--no-vcf) sees no mate.
    Board b = makeBoard({{4, 7}, {0, 0}, {2, 0}, {4, 0}},
                        {{5, 7}, {6, 7}, {7, 7}, {6, 5}, {7, 6}},
                        Color::White);
    Search s1(1 << 20);
    Search s2(1 << 20);
    const SearchResult r1 = s1.findBestMove(b, SearchLimits{2, 0, true});
    const SearchResult r2 = s2.findBestMove(b, SearchLimits{2, 0, false});
    CHECK(r1.fromVcf);
    CHECK(r1.score > 900000);
    CHECK(r1.move == p(8, 7));
    CHECK(!r2.fromVcf);
    CHECK(std::abs(r2.score) < 900000);
  }

  if (g_failures > 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "search_tests: all checks passed\n";
  return 0;
}
