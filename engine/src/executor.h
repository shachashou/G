#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "board.h"

namespace gomoku {

// Why a requested move was rejected. Every verdict comes from the single
// Board / renju::classifyForbidden implementation; the GUI and the Python
// referee never compute legality themselves.
enum class RejectReason {
  None,
  OutOfBounds,
  Occupied,
  ForbiddenOverline,
  ForbiddenDoubleFour,
  ForbiddenDoubleThree,
  GameOver,
};

const char* rejectReasonName(RejectReason reason);

struct MoveOutcome {
  bool ok = false;
  RejectReason reason = RejectReason::None;
};

// Immutable snapshot of the authoritative position.
struct PositionState {
  std::array<Cell, kCells> cells{};
  Color sideToMove = Color::Black;
  GameResult result = GameResult::Ongoing;
  RuleMode rule = RuleMode::Renju;
  int moveCount = 0;
  std::vector<int> history;  // move indices, oldest first
  int lastMove = -1;
};

struct ReplayResult {
  bool ok = false;
  std::string error;
  int failedAt = -1;  // index into the replayed sequence
  RejectReason reason = RejectReason::None;
  PositionState finalState;
};

// Authoritative rules executor. One implementation of legality, forbidden
// moves and results is shared by the GUI, data collection, tactic-evidence
// review and match arbitration (spec section 1, ticket 02).
class Executor {
 public:
  explicit Executor(RuleMode rule = RuleMode::Renju);

  PositionState state() const;
  void newGame(RuleMode rule);

  // COMPLETE legal move set: every empty, non-foul point. This is deliberately
  // NOT the search's candidate pruning (see candidates()).
  std::vector<int> legalMoves() const;
  // Explains why a point is or is not playable, for UI and referees.
  RejectReason moveReason(int move) const;

  // Applies a move. A rejected request changes nothing: cells, side to move,
  // result and history all stay exactly as they were.
  MoveOutcome play(int move);
  // Reverts the last move recorded in this session; false when history empty.
  bool undo();

  // Rebuilds the position from a snapshot and validates it. Returns an error
  // message on invalid input, leaving the current position untouched.
  std::optional<std::string> rebuild(const std::array<Cell, kCells>& cells,
                                     Color sideToMove, RuleMode rule);

  // Replays a move sequence from an empty board using the same rules. Does
  // not modify this executor's own position.
  ReplayResult replay(const std::vector<int>& moves, RuleMode rule) const;

  // Search-side candidate pruning (neighbourhood + threat ordering), always
  // a subset of legalMoves(). Kept separate so nobody mistakes it for the
  // complete legal set.
  std::vector<int> candidates(int radius = 1) const;

  const Board& board() const { return board_; }

 private:
  Board board_;
};

}  // namespace gomoku
