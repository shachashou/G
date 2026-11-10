#include "executor.h"

#include <utility>

#include "eval.h"
#include "movegen.h"
#include "renju.h"

namespace gomoku {

namespace {

PositionState stateOf(const Board& board) {
  PositionState s;
  s.cells = board.rawCells();
  s.sideToMove = board.sideToMove();
  s.result = board.result();
  s.rule = board.ruleMode();
  s.moveCount = board.moveCount();
  s.history = board.moveHistory();
  s.lastMove = board.lastMove();
  return s;
}

RejectReason forbiddenToReject(renju::ForbiddenReason reason) {
  switch (reason) {
    case renju::ForbiddenReason::Overline:
      return RejectReason::ForbiddenOverline;
    case renju::ForbiddenReason::DoubleFour:
      return RejectReason::ForbiddenDoubleFour;
    case renju::ForbiddenReason::DoubleThree:
      return RejectReason::ForbiddenDoubleThree;
    case renju::ForbiddenReason::None:
      return RejectReason::None;
  }
  return RejectReason::None;
}

// The one legality gate, shared by play(), legalMoves() and replay().
// It mirrors Board::isPlayable() and adds the reason for a rejection.
RejectReason reasonOf(const Board& board, int move) {
  if (!Board::inBounds(move)) return RejectReason::OutOfBounds;
  if (board.result() != GameResult::Ongoing) return RejectReason::GameOver;
  if (!board.isEmpty(move)) return RejectReason::Occupied;
  if (board.ruleMode() == RuleMode::Renju &&
      board.sideToMove() == Color::Black) {
    return forbiddenToReject(renju::classifyForbidden(board.rawCells(), move));
  }
  return RejectReason::None;
}

}  // namespace

const char* rejectReasonName(RejectReason reason) {
  switch (reason) {
    case RejectReason::None: return "legal";
    case RejectReason::OutOfBounds: return "out_of_bounds";
    case RejectReason::Occupied: return "occupied";
    case RejectReason::ForbiddenOverline: return "forbidden_overline";
    case RejectReason::ForbiddenDoubleFour: return "forbidden_double_four";
    case RejectReason::ForbiddenDoubleThree: return "forbidden_double_three";
    case RejectReason::GameOver: return "game_over";
  }
  return "unknown";
}

Executor::Executor(RuleMode rule) : board_(rule) {}

PositionState Executor::state() const { return stateOf(board_); }

void Executor::newGame(RuleMode rule) { board_.reset(rule); }

RejectReason Executor::moveReason(int move) const {
  return reasonOf(board_, move);
}

std::vector<int> Executor::legalMoves() const {
  std::vector<int> out;
  if (board_.result() != GameResult::Ongoing) return out;
  for (int m = 0; m < kCells; ++m) {
    if (reasonOf(board_, m) == RejectReason::None) out.push_back(m);
  }
  return out;
}

MoveOutcome Executor::play(int move) {
  const RejectReason reason = reasonOf(board_, move);
  if (reason != RejectReason::None || !board_.isPlayable(move)) {
    return {false, reason == RejectReason::None ? RejectReason::Occupied
                                                : reason};
  }
  board_.place(move);
  return {true, RejectReason::None};
}

bool Executor::undo() {
  if (board_.moveHistory().empty()) return false;
  board_.undo();
  return true;
}

std::optional<std::string> Executor::rebuild(
    const std::array<Cell, kCells>& cells, Color sideToMove, RuleMode rule) {
  for (Cell c : cells) {
    if (c != Cell::Black && c != Cell::White && c != Cell::Empty) {
      return "invalid cell value (expected 0, 1 or 2)";
    }
  }
  Board fresh(rule);
  for (int m = 0; m < kCells; ++m) fresh.setCell(m, cells[m]);
  fresh.finalizeSync();
  // A snapshot cannot restore move history. Honour the caller's side to
  // move: tactic-evidence review and Renju problem positions often have
  // stone counts no legal game history could produce. The protocol layer
  // reports whether the snapshot is consistent with black-first play.
  fresh.setSideToMove(sideToMove);
  board_ = std::move(fresh);
  return std::nullopt;
}

ReplayResult Executor::replay(const std::vector<int>& moves,
                              RuleMode rule) const {
  ReplayResult out;
  Board board(rule);
  for (size_t i = 0; i < moves.size(); ++i) {
    const RejectReason reason = reasonOf(board, moves[i]);
    if (reason != RejectReason::None) {
      out.error = "illegal move in sequence";
      out.failedAt = static_cast<int>(i);
      out.reason = reason;
      return out;
    }
    board.place(moves[i]);
  }
  out.ok = true;
  out.finalState = stateOf(board);
  return out;
}

std::vector<int> Executor::candidates(int radius) const {
  std::vector<int> out;
  if (board_.result() != GameResult::Ongoing) return out;
  Evaluator eval;
  eval.init(board_);
  std::vector<ScoredMove> scored;
  generateCandidates(board_, eval, scored, radius);
  for (const ScoredMove& s : scored) out.push_back(s.move);
  return out;
}

}  // namespace gomoku
