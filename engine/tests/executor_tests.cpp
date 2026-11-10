// Authoritative executor unit tests (ticket 02).
// Reuses the classic Renju problem shapes from renju_tests.cpp and asserts
// externally observable behaviour of the shared executor.
#include <array>
#include <initializer_list>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "executor.h"

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

static std::array<Cell, kCells> makeCells(
    std::initializer_list<Pt> blacks, std::initializer_list<Pt> whites) {
  std::array<Cell, kCells> cells{};
  for (const Pt& q : blacks) cells[p(q.first, q.second)] = Cell::Black;
  for (const Pt& q : whites) cells[p(q.first, q.second)] = Cell::White;
  return cells;
}

static bool sameState(const PositionState& a, const PositionState& b) {
  return a.cells == b.cells && a.sideToMove == b.sideToMove &&
         a.result == b.result && a.rule == b.rule &&
         a.moveCount == b.moveCount && a.history == b.history &&
         a.lastMove == b.lastMove;
}

// D4 transforms: 4 rotations x optional mirror.
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

static std::array<Cell, kCells> transform(
    const std::vector<Pt>& blacks, const std::vector<Pt>& whites, int rot,
    bool mirror, Pt& probe) {
  std::array<Cell, kCells> cells{};
  for (const Pt& q : blacks) {
    const Pt t = xf(q, rot, mirror);
    cells[p(t.first, t.second)] = Cell::Black;
  }
  for (const Pt& q : whites) {
    const Pt t = xf(q, rot, mirror);
    cells[p(t.first, t.second)] = Cell::White;
  }
  probe = xf({7, 7}, rot, mirror);
  return cells;
}

int main() {
  {
    // Fresh board: complete legal set vs search candidate pruning.
    Executor ex;
    CHECK(ex.legalMoves().size() == kCells);
    CHECK(ex.candidates().size() == 1);
    CHECK(ex.candidates()[0] == p(7, 7));
    bool allLegal = true;
    for (int m = 0; m < kCells; ++m) {
      allLegal = allLegal && ex.moveReason(m) == RejectReason::None;
    }
    CHECK(allLegal);
    CHECK(ex.moveReason(-1) == RejectReason::OutOfBounds);
    CHECK(ex.moveReason(kCells) == RejectReason::OutOfBounds);
    CHECK(ex.moveReason(p(0, 0)) == RejectReason::None);
    CHECK(ex.moveReason(p(14, 14)) == RejectReason::None);
  }
  {
    // Candidates are strictly a subset of the complete legal set.
    Executor ex;
    CHECK(ex.play(p(7, 7)).ok);
    CHECK(ex.play(p(8, 8)).ok);
    const std::vector<int> legal = ex.legalMoves();
    const std::vector<int> candidates = ex.candidates(1);
    CHECK(candidates.size() < legal.size());
    std::vector<bool> isLegal(kCells, false);
    for (int m : legal) isLegal[m] = true;
    bool subset = true;
    for (int m : candidates) subset = subset && isLegal[m];
    CHECK(subset);
  }
  {
    // Illegal requests (out of bounds / occupied) change nothing at all.
    Executor ex;
    const PositionState before = ex.state();
    const MoveOutcome oob = ex.play(Board::indexOf(99, 99));
    CHECK(!oob.ok && oob.reason == RejectReason::OutOfBounds);
    CHECK(sameState(before, ex.state()));
    CHECK(ex.play(p(7, 7)).ok);
    const PositionState after = ex.state();
    const MoveOutcome occupied = ex.play(p(7, 7));
    CHECK(!occupied.ok && occupied.reason == RejectReason::Occupied);
    CHECK(sameState(after, ex.state()));
  }
  {
    // Rejection reasons come from the single Renju decision procedure.
    // Shapes are the classic examples from renju_tests.cpp.
    Executor ex;
    ex.rebuild(makeCells({{3, 7}, {4, 7}, {5, 7}, {7, 7}, {8, 7}},
                         {{0, 0}, {2, 0}}),
               Color::Black, RuleMode::Renju);
    CHECK(ex.moveReason(p(6, 7)) == RejectReason::ForbiddenOverline);

    // Five takes precedence over any coinciding foul pattern.
    ex.rebuild(makeCells({{5, 7}, {6, 7}, {8, 7}, {9, 7}, {7, 5},
                          {7, 6}, {7, 8}, {5, 5}, {6, 6}, {8, 8}},
                         {{0, 0}, {2, 0}}),
               Color::Black, RuleMode::Renju);
    CHECK(ex.moveReason(p(7, 7)) == RejectReason::None);

    const auto doubleFour =
        makeCells({{5, 7}, {6, 7}, {8, 7}, {7, 5}, {7, 6}, {7, 8}},
                  {{0, 0}, {2, 0}});
    ex.rebuild(doubleFour, Color::Black, RuleMode::Renju);
    CHECK(ex.moveReason(p(7, 7)) == RejectReason::ForbiddenDoubleFour);

    ex.rebuild(makeCells({{5, 7}, {6, 7}, {8, 7}}, {{0, 0}}), Color::Black,
               RuleMode::Renju);
    CHECK(ex.moveReason(p(7, 7)) == RejectReason::None);  // single four

    ex.rebuild(makeCells({{6, 7}, {8, 7}, {7, 6}, {7, 8}}, {{0, 0}, {2, 0}}),
               Color::Black, RuleMode::Renju);
    CHECK(ex.moveReason(p(7, 7)) == RejectReason::ForbiddenDoubleThree);

    // Sleeping three (one end blocked) is not an open three.
    ex.rebuild(makeCells({{6, 7}, {8, 7}, {7, 6}, {7, 8}}, {{9, 7}, {0, 0}}),
               Color::Black, RuleMode::Renju);
    CHECK(ex.moveReason(p(7, 7)) == RejectReason::None);

    ex.rebuild(makeCells({{5, 7}, {8, 7}, {7, 4}, {7, 6}}, {{0, 0}, {2, 0}}),
               Color::Black, RuleMode::Renju);
    CHECK(ex.moveReason(p(7, 7)) ==
          RejectReason::ForbiddenDoubleThree);  // jump three

    ex.rebuild(makeCells({{5, 7}, {6, 7}, {8, 7}, {7, 6}, {7, 8}},
                         {{0, 0}, {2, 0}}),
               Color::Black, RuleMode::Renju);
    CHECK(ex.moveReason(p(7, 7)) == RejectReason::None);  // four-three

    // Nested / false three: only one genuine open three remains.
    ex.rebuild(makeCells({{6, 7}, {8, 7}, {7, 6}, {7, 8},
                          {5, 5}, {5, 6}, {5, 8},
                          {9, 5}, {9, 6}, {9, 8}},
                         {{0, 0}, {2, 0}}),
               Color::Black, RuleMode::Renju);
    CHECK(ex.moveReason(p(7, 7)) == RejectReason::None);

    // White has no fouls; freestyle ignores black fouls too.
    ex.rebuild(makeCells({{0, 0}, {2, 0}},
                         {{5, 7}, {6, 7}, {8, 7}, {7, 5}, {7, 6}, {7, 8}}),
               Color::White, RuleMode::Renju);
    CHECK(ex.moveReason(p(7, 7)) == RejectReason::None);
    ex.rebuild(doubleFour, Color::Black, RuleMode::Freestyle);
    CHECK(ex.moveReason(p(7, 7)) == RejectReason::None);
  }
  {
    // A forbidden move is rejected and leaves the whole position untouched.
    Executor ex;
    ex.rebuild(makeCells({{5, 7}, {6, 7}, {8, 7}, {7, 5}, {7, 6}, {7, 8}},
                         {{0, 0}, {2, 0}}),
               Color::Black, RuleMode::Renju);
    const PositionState before = ex.state();
    const MoveOutcome foul = ex.play(p(7, 7));
    CHECK(!foul.ok && foul.reason == RejectReason::ForbiddenDoubleFour);
    CHECK(sameState(before, ex.state()));
    const std::vector<int> legal = ex.legalMoves();
    bool excluded = true;
    for (int m : legal) excluded = excluded && m != p(7, 7);
    CHECK(excluded);
  }
  {
    // Terminal rejection: moves after a win are refused and change nothing.
    Executor ex;
    const int bx[] = {3, 4, 5, 6, 7};
    const int wx[] = {0, 1, 2, 3};
    for (int i = 0; i < 4; ++i) {
      CHECK(ex.play(p(bx[i], 7)).ok);
      CHECK(ex.play(p(wx[i], 0)).ok);
    }
    CHECK(ex.play(p(bx[4], 7)).ok);
    CHECK(ex.state().result == GameResult::BlackWin);
    const PositionState before = ex.state();
    const MoveOutcome afterEnd = ex.play(p(10, 10));
    CHECK(!afterEnd.ok && afterEnd.reason == RejectReason::GameOver);
    CHECK(sameState(before, ex.state()));
    CHECK(ex.legalMoves().empty());
    CHECK(ex.candidates().empty());
    CHECK(ex.undo());  // undo still works after a rejected move
    CHECK(ex.state().result == GameResult::Ongoing);
  }
  {
    // White overline (>= 5) wins in Renju; black stones stay off any line.
    Executor ex;
    ex.rebuild(makeCells({{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {0, 2}},
                         {{3, 7}, {4, 7}, {5, 7}, {6, 7}, {7, 7}, {8, 7}}),
               Color::Black, RuleMode::Renju);
    CHECK(ex.state().result == GameResult::WhiteWin);
  }
  {
    // Undo restores the exact previous state; empty history is refused.
    Executor ex;
    const PositionState initial = ex.state();
    CHECK(!ex.undo());
    CHECK(sameState(initial, ex.state()));
    CHECK(ex.play(p(7, 7)).ok);
    CHECK(ex.play(p(8, 8)).ok);
    CHECK(ex.state().history.size() == 2);
    CHECK(ex.undo());
    CHECK(ex.state().history.size() == 1);
    CHECK(ex.state().sideToMove == Color::White);
    CHECK(ex.undo());
    CHECK(sameState(initial, ex.state()));
  }
  {
    // Rebuild validation: corrupt cells are refused without side effects,
    // a valid snapshot rebuilds without history, and a finished snapshot
    // reports its result.
    Executor ex;
    CHECK(ex.play(p(7, 7)).ok);
    const PositionState before = ex.state();
    std::array<Cell, kCells> bad{};
    bad[0] = static_cast<Cell>(7);
    CHECK(ex.rebuild(bad, Color::Black, RuleMode::Renju).has_value());
    CHECK(sameState(before, ex.state()));
    CHECK(!ex.rebuild(makeCells({{3, 3}, {4, 4}}, {{10, 10}}), Color::White,
                      RuleMode::Renju)
               .has_value());
    CHECK(ex.state().moveCount == 3);
    CHECK(ex.state().sideToMove == Color::White);
    CHECK(ex.state().history.empty());
    CHECK(ex.state().lastMove == -1);
    CHECK(ex.state().cells[p(3, 3)] == Cell::Black);
    CHECK(!ex.undo());
    CHECK(!ex.rebuild(makeCells({{3, 7}, {4, 7}, {5, 7}, {6, 7}, {7, 7}},
                                {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}}),
                      Color::White, RuleMode::Renju)
               .has_value());
    CHECK(ex.state().result == GameResult::BlackWin);
  }
  {
    // D4 symmetry: forbidden stays forbidden, legal stays legal.
    const std::vector<Pt> dfBlacks = {{5, 7}, {6, 7}, {8, 7},
                                      {7, 5}, {7, 6}, {7, 8}};
    const std::vector<Pt> sfBlacks = {{5, 7}, {6, 7}, {8, 7}};
    const std::vector<Pt> whites = {{0, 0}, {2, 0}};
    for (int rot = 0; rot < 4; ++rot) {
      for (bool mirror : {false, true}) {
        Pt probe{};
        Executor ex;
        ex.rebuild(transform(dfBlacks, whites, rot, mirror, probe), Color::Black,
                   RuleMode::Renju);
        CHECK(ex.moveReason(p(probe.first, probe.second)) ==
              RejectReason::ForbiddenDoubleFour);
        ex.rebuild(transform(sfBlacks, whites, rot, mirror, probe),
                   Color::Black, RuleMode::Renju);
        CHECK(ex.moveReason(p(probe.first, probe.second)) ==
              RejectReason::None);
      }
    }
  }
  {
    // Replay: a full game reproduces exactly the same final state.
    Executor ex;
    std::mt19937 rng(20261110u);
    std::vector<int> moves;
    PositionState state = ex.state();
    for (int step = 0; step < kCells && state.result == GameResult::Ongoing;
         ++step) {
      const std::vector<int> legal = ex.legalMoves();
      if (legal.empty()) break;
      const int m = legal[rng() % legal.size()];
      const MoveOutcome out = ex.play(m);
      CHECK(out.ok);
      if (!out.ok) break;
      moves.push_back(m);
      state = ex.state();
    }
    CHECK(state.result != GameResult::Ongoing);
    const ReplayResult replayed = ex.replay(moves, RuleMode::Renju);
    CHECK(replayed.ok);
    CHECK(replayed.finalState.cells == state.cells);
    CHECK(replayed.finalState.result == state.result);
    CHECK(replayed.finalState.history == state.history);
    CHECK(replayed.finalState.sideToMove == state.sideToMove);
  }
  {
    // Replay rejects illegal sequences and moves after the game ended.
    Executor ex;
    const ReplayResult empty = ex.replay({}, RuleMode::Renju);
    CHECK(empty.ok && empty.finalState.moveCount == 0 &&
          empty.finalState.result == GameResult::Ongoing);
    const ReplayResult oob =
        ex.replay({p(7, 7), Board::indexOf(99, 99)}, RuleMode::Renju);
    CHECK(!oob.ok && oob.failedAt == 1 &&
          oob.reason == RejectReason::OutOfBounds);

    std::vector<int> seq;
    const int bx[] = {3, 4, 5, 6, 7};
    const int wx[] = {0, 1, 2, 3};
    for (int i = 0; i < 4; ++i) {
      seq.push_back(p(bx[i], 7));
      seq.push_back(p(wx[i], 0));
    }
    seq.push_back(p(bx[4], 7));  // black wins
    seq.push_back(p(10, 10));    // extra move
    const ReplayResult afterEnd = ex.replay(seq, RuleMode::Renju);
    CHECK(!afterEnd.ok && afterEnd.failedAt == 9 &&
          afterEnd.reason == RejectReason::GameOver);
    seq.pop_back();
    const ReplayResult clean = ex.replay(seq, RuleMode::Renju);
    CHECK(clean.ok && clean.finalState.result == GameResult::BlackWin);
  }

  if (g_failures > 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "executor_tests: all checks passed\n";
  return 0;
}
