// Search tactical behaviour tests (ticket 04), including the VCF call sites
// the search uses (ticket 03).
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

// Coordinates such as "H8" -> (7,7), the same convention the upstream
// positions use.
static std::vector<Pt> coords(const char* text) {
  std::vector<Pt> out;
  const std::string input(text);
  size_t start = 0;
  while (start < input.size()) {
    size_t comma = input.find(',', start);
    if (comma == std::string::npos) comma = input.size();
    const std::string token = input.substr(start, comma - start);
    if (token.size() >= 2) {
      out.emplace_back(token[0] - 'A', std::stoi(token.substr(1)) - 1);
    }
    start = comma + 1;
  }
  return out;
}

static Board position(const char* blacks, const char* whites, Color stm) {
  Board b(RuleMode::Renju);
  for (const Pt& q : coords(blacks)) b.setCell(p(q.first, q.second), Cell::Black);
  for (const Pt& q : coords(whites)) b.setCell(p(q.first, q.second), Cell::White);
  b.finalizeSync();
  b.setSideToMove(stm);
  return b;
}

static SearchLimits limitsOf(int depth, int ms = 0,
                             VcfPolicy policy = VcfPolicy::Root) {
  SearchLimits limits;
  limits.maxDepth = depth;
  limits.moveTimeMs = ms;
  limits.vcfPolicy = policy;
  return limits;
}

int main() {
  {
    // Immediate win: black has an open four, search must complete five.
    Board b = makeBoard({{5, 7}, {6, 7}, {7, 7}, {8, 7}},
                        {{0, 0}, {2, 0}, {4, 0}, {6, 0}}, Color::Black);
    Search s(1 << 20);
    const SearchResult r = s.findBestMove(b, limitsOf(3));
    CHECK(r.move == p(4, 7) || r.move == p(9, 7));
    CHECK(r.nodes > 0);
  }
  {
    // Must block: white has a single four whose only winning point is (4,7)
    // ((9,7) is occupied by black). Any other black move loses next turn.
    Board b = makeBoard({{9, 7}}, {{5, 7}, {6, 7}, {7, 7}, {8, 7}},
                        Color::Black);
    Search s(1 << 20);
    const SearchResult r = s.findBestMove(b, limitsOf(3));
    CHECK(r.move == p(4, 7));
  }
  {
    // Empty board: legal move, deterministic, board fully restored.
    Board b;
    const auto before = b.rawCells();
    Search s1(1 << 20);
    const SearchResult r1 = s1.findBestMove(b, limitsOf(3));
    CHECK(r1.move >= 0 && b.isPlayable(r1.move));
    CHECK(b.rawCells() == before);
    CHECK(b.moveCount() == 0);
    CHECK(r1.nodes > 0);
    Search s2(1 << 20);
    const SearchResult r2 = s2.findBestMove(b, limitsOf(3));
    CHECK(r1.move == r2.move);
  }
  {
    // A short time budget must still return a legal move (no hang).
    Board b;
    Search s(1 << 20);
    const SearchResult r = s.findBestMove(b, limitsOf(8, 20));
    CHECK(r.move >= 0 && b.isPlayable(r.move));
    CHECK(b.moveCount() == 0);
  }

  {
    // VCF A/B: a 5-ply continuous-four win is found at the root even at
    // depth 2, while the baseline (policy off) sees no mate.
    Board b = makeBoard({{4, 7}, {0, 0}, {2, 0}, {4, 0}},
                        {{5, 7}, {6, 7}, {7, 7}, {6, 5}, {7, 6}},
                        Color::White);
    Search s1(1 << 20);
    Search s2(1 << 20);
    const SearchResult r1 = s1.findBestMove(b, limitsOf(2, 0, VcfPolicy::Root));
    const SearchResult r2 = s2.findBestMove(b, limitsOf(2, 0, VcfPolicy::Off));
    CHECK(r1.fromVcf);
    CHECK(r1.score > 900000);
    CHECK(r1.move == p(8, 7));
    CHECK(r1.vcf.status == VcfStatus::Proven);
    CHECK(!r1.vcf.variation.empty());
    CHECK(r2.vcf.status == VcfStatus::InvalidInput);  // never queried
    CHECK(!r2.fromVcf);
    CHECK(std::abs(r2.score) < 900000);
  }
  {
    // RootAndChildren: the second call site asks whether the opponent has a
    // proven VCF after each top root move. White to move here; Black has the
    // counter-four win from the position, so most White replies are proven
    // losses and the counters must show it.
    Board b = position("H8,J8,K8,I7,L7,I6", "H9,L8,J7,K6,I5,L5",
                       Color::White);
    const auto before = b.rawCells();
    Search s(1 << 20);
    SearchLimits limits = limitsOf(3, 0, VcfPolicy::RootAndChildren);
    limits.vcfChildChecks = 12;
    const SearchResult r = s.findBestMove(b, limits);
    CHECK(r.vcfChildChecks > 0);
    CHECK(r.vcfChildLosses > 0);
    CHECK(r.move >= 0 && b.isPlayable(r.move));
    CHECK(b.rawCells() == before);  // search restores the board
  }
  {
    // Never prune every root move: even when every checked child is a proven
    // loss the search still has to return a legal move.
    Board b = position("H8,J8,K8,I7,L7,I6", "H9,L8,J7,K6,I5,L5",
                       Color::White);
    const auto before = b.rawCells();
    Search s(1 << 20);
    SearchLimits limits = limitsOf(2, 0, VcfPolicy::RootAndChildren);
    limits.vcfChildChecks = 64;  // every candidate
    const SearchResult r = s.findBestMove(b, limits);
    CHECK(r.move >= 0 && b.isPlayable(r.move));
    CHECK(b.rawCells() == before);
  }

  if (g_failures > 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "search_tests: all checks passed\n";
  return 0;
}

