// VCF adapter tests over the vendored upstream solver (ticket 03).
//
// The solver itself is upstream quintet; these tests check our adapter
// boundary: result vocabulary (proven / disproven / budget / interrupted /
// unsupported / invalid), the recorded budget and variation, the evidence
// re-check through the authoritative board, Renju fouls on both attack and
// defence, and D4 symmetry.
#include <initializer_list>
#include <iostream>
#include <string>
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

// Parses quintet-style coordinates ("H8,I9") into board indices. Row 8 is
// y = 7: the same convention the upstream test positions use.
static std::vector<Pt> coords(const char* text) {
  std::vector<Pt> out;
  const std::string input(text);
  size_t start = 0;
  while (start < input.size()) {
    size_t comma = input.find(',', start);
    if (comma == std::string::npos) comma = input.size();
    const std::string token = input.substr(start, comma - start);
    if (token.size() >= 2) {
      const int x = token[0] - 'A';
      const int y = std::stoi(token.substr(1)) - 1;
      out.emplace_back(x, y);
    }
    start = comma + 1;
  }
  return out;
}

static Board position(const char* blacks, const char* whites, Color stm,
                      RuleMode mode = RuleMode::Renju) {
  Board b(mode);
  for (const Pt& q : coords(blacks)) b.setCell(p(q.first, q.second), Cell::Black);
  for (const Pt& q : coords(whites)) b.setCell(p(q.first, q.second), Cell::White);
  b.finalizeSync();
  b.setSideToMove(stm);
  return b;
}

static bool samePath(const VcfResult& r, const char* expected) {
  const std::vector<Pt> want = coords(expected);
  if (r.variation.size() != want.size()) return false;
  for (size_t i = 0; i < want.size(); ++i) {
    if (r.variation[i] != p(want[i].first, want[i].second)) return false;
  }
  return true;
}

// Counter-fours: Black H8,J8,K8,I7,L7,I6 against White H9,L8,J7,K6,I5,L5.
// The forced line is I8,G8,I10,I9,J9 (upstream test_vcf_counter).
static const char* kCounterBlack = "H8,J8,K8,I7,L7,I6";
static const char* kCounterWhite = "H9,L8,J7,K6,I5,L5";

// Puzzle 3040: a 12-attack Black VCF (upstream test_vcf_black).
static const char* kPuzzleBlack =
    "L14,E13,H13,H12,I11,G10,H10,K10,J9,H8,I8";
static const char* kPuzzleWhite =
    "I14,G13,K12,H11,L11,I10,F9,H9,I9,L9,J8";

// Puzzle 2990: a 5-attack White VCF (upstream test_vcf_white).
static const char* kWhiteBlack = "K13,M12,M11,J10,G9,I9,J9,M9,G8,H8";
static const char* kWhiteWhite = "L14,J12,L12,I11,H10,I10,L10,H9,F8";

// "shadows and fog": a very large VCF tree (upstream test_vcf_long), used to
// exercise the budget and wall-clock exits.
static const char* kLongBlack =
    "B15,F15,H15,M15,D14,H14,I14,N14,O14,C13,M13,O13,A12,C11,N11,B10,N10,"
    "O10,B9,D9,L9,A8,H8,H6,A5,E5,O5,B4,D4,I4,I3,J3,A2,J2,N2,D1,G1,K1,O1";
static const char* kLongWhite =
    "C15,E15,I15,J15,K15,L15,N15,O15,G14,J14,M14,A13,E12,F12,N12,O12,O11,"
    "A10,C10,D10,A9,G9,B8,C8,D8,J8,O8,A7,B7,O7,A6,G6,I6,O6,J5,H4,G3,M2,H1";

// White's four has one winning point (7,7); Black may not block it because it
// would be a double-four foul, so White wins by that forbidden defence.
static const Board kForbiddenDefence() {
  return makeBoard({{2, 7}, {5, 5}, {6, 6}, {8, 8}, {5, 9}, {6, 8}, {8, 6}},
                   {{3, 7}, {4, 7}, {5, 7}}, Color::White);
}

int main() {
  {
    // Proven VCF: a five-ply line, with the first move and the line reported.
    Board b = position(kCounterBlack, kCounterWhite, Color::Black);
    const auto before = b.rawCells();
    const VcfResult r = solveVcf(b, VcfLimits{50000, 10});
    CHECK(r.status == VcfStatus::Proven);
    CHECK(r.firstMove == p(8, 7));  // I8
    CHECK(samePath(r, "I8,G8,I10,I9,J9"));
    CHECK(r.evidence.kind == VcfEndKind::Fours);
    CHECK(r.nodes > 0);
    CHECK(r.limits.nodeLimit == 50000);  // budget echoed for records
    CHECK(checkVcfEvidence(b, r).empty());
    CHECK(b.rawCells() == before);  // adapter never mutates the board
  }
  {
    // A longer proven line with a tighter attacker-move budget.
    Board b = position(kPuzzleBlack, kPuzzleWhite, Color::Black);
    const VcfResult r = solveVcf(b, VcfLimits{200000, 12});
    CHECK(r.status == VcfStatus::Proven);
    CHECK(r.firstMove == p(9, 11));  // J12
    CHECK(r.variation.size() == 23);  // 12 attacks, 11 forced replies
    CHECK(checkVcfEvidence(b, r).empty());
    CHECK(r.evidence.kind == VcfEndKind::Fours);
  }
  {
    // White attacks: the same adapter serves either side to move.
    Board b = position(kWhiteBlack, kWhiteWhite, Color::White);
    const VcfResult r = solveVcf(b, VcfLimits{200000, 5});
    CHECK(r.status == VcfStatus::Proven);
    CHECK(r.firstMove == p(11, 12));  // L13
    CHECK(checkVcfEvidence(b, r).empty());
  }
  {
    // 禁手杀: the proven ending is a forbidden block for Black.
    Board b = kForbiddenDefence();
    Board asBlack = b;
    asBlack.setSideToMove(Color::Black);
    CHECK(!asBlack.isPlayable(p(7, 7)));  // the board agrees it is a foul
    const VcfResult r = solveVcf(b);
    CHECK(r.status == VcfStatus::Proven);
    CHECK(r.evidence.kind == VcfEndKind::Forbidden);
    CHECK(r.evidence.firstPoint == p(7, 7));
    CHECK(checkVcfEvidence(b, r).empty());
  }
  {
    // Depth bound: the same line is not proven when the attacker may only
    // play two fours, and "not found" is reported as Disproven, not unknown.
    Board b = position(kCounterBlack, kCounterWhite, Color::Black);
    const VcfResult r = solveVcf(b, VcfLimits{50000, 2});
    CHECK(r.status == VcfStatus::Disproven);
    CHECK(r.firstMove < 0);
    CHECK(r.variation.empty());
  }
  {
    // Quiet position: no VCF at all, budget untouched -> Disproven.
    Board b = makeBoard({{7, 8}, {0, 0}}, {{7, 7}, {8, 8}}, Color::White);
    const VcfResult r = solveVcf(b);
    CHECK(r.status == VcfStatus::Disproven);
    CHECK(r.variation.empty());
  }
  {
    // Node budget: a hard stop reports BudgetExhausted (unknown), never
    // "no mate".
    Board b = position(kLongBlack, kLongWhite, Color::Black);
    const VcfResult r = solveVcf(b, VcfLimits{5, 10});
    CHECK(r.status == VcfStatus::BudgetExhausted);
    CHECK(!r.proven());
    CHECK(r.unknown());
    CHECK(r.nodes <= 6);
    CHECK(r.variation.empty());
  }
  // The wall-clock exit (Interrupted) is covered by the Rust bridge's own
  // unit tests (engine/vcf-bridge: `cargo test`), where a deliberately slow
  // task can be scheduled deterministically; no real VCF position on this
  // machine reliably exceeds a 1 ms deadline.
  {
    // Freestyle is unsupported by the upstream Renju solver: say so instead
    // of silently returning "no VCF".
    Board b = makeBoard({{7, 7}}, {{8, 8}}, Color::Black, RuleMode::Freestyle);
    const VcfResult r = solveVcf(b);
    CHECK(r.status == VcfStatus::UnsupportedRule);
  }
  {
    // Input errors are distinguished from tactical verdicts.
    Board b = position(kCounterBlack, kCounterWhite, Color::Black);
    CHECK(solveVcf(b, VcfLimits{0, 10}).status == VcfStatus::InvalidInput);
    CHECK(solveVcf(b, VcfLimits{100, 0}).status == VcfStatus::InvalidInput);
    CHECK(solveVcf(b, VcfLimits{100, 256}).status == VcfStatus::InvalidInput);
    CHECK(solveVcf(b, VcfLimits{100, 10, -1}).status ==
          VcfStatus::InvalidInput);
    Board finished = makeBoard({{3, 7}, {4, 7}, {5, 7}, {6, 7}, {7, 7}},
                               {{0, 0}}, Color::White);
    CHECK(finished.result() != GameResult::Ongoing);
    CHECK(solveVcf(finished).status == VcfStatus::InvalidInput);
  }
  {
    // Evidence re-check: a line the authoritative board rejects is refused,
    // whether the point is occupied, out of bounds, after game end, or a
    // Renju foul for Black.
    Board b = kForbiddenDefence();
    b.setSideToMove(Color::Black);
    VcfResult forged;
    forged.status = VcfStatus::Proven;
    forged.firstMove = p(7, 7);  // double-four for Black
    forged.variation = {p(7, 7)};
    CHECK(!checkVcfEvidence(b, forged).empty());

    VcfResult occupied;
    occupied.status = VcfStatus::Proven;
    occupied.firstMove = p(3, 7);  // already White's stone
    occupied.variation = {p(3, 7)};
    CHECK(!checkVcfEvidence(b, occupied).empty());

    VcfResult mismatch;
    mismatch.status = VcfStatus::Proven;
    mismatch.firstMove = p(0, 0);
    mismatch.variation = {p(1, 0)};
    CHECK(!checkVcfEvidence(b, mismatch).empty());

    VcfResult emptyLine;
    emptyLine.status = VcfStatus::Proven;
    emptyLine.firstMove = -1;
    CHECK(checkVcfEvidence(b, emptyLine).empty());
  }
  {
    // D4 symmetry: the same forced line is proven under all eight transforms.
    const std::vector<Pt> blacks = coords(kCounterBlack);
    const std::vector<Pt> whites = coords(kCounterWhite);
    auto transform = [](Pt q, int rot, bool mirror) {
      int x = q.first;
      int y = q.second;
      if (mirror) x = kBoardSize - 1 - x;
      for (int i = 0; i < rot; ++i) {
        const int nx = y;
        const int ny = kBoardSize - 1 - x;
        x = nx;
        y = ny;
      }
      return Pt{x, y};
    };
    for (int rot = 0; rot < 4; ++rot) {
      for (bool mirror : {false, true}) {
        Board b(RuleMode::Renju);
        for (const Pt& q : blacks) {
          const Pt t = transform(q, rot, mirror);
          b.setCell(p(t.first, t.second), Cell::Black);
        }
        for (const Pt& q : whites) {
          const Pt t = transform(q, rot, mirror);
          b.setCell(p(t.first, t.second), Cell::White);
        }
        b.finalizeSync();
        b.setSideToMove(Color::Black);
        const VcfResult r = solveVcf(b);
        CHECK(r.status == VcfStatus::Proven);
        CHECK(checkVcfEvidence(b, r).empty());
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
