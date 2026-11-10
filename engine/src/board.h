#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace gomoku {

constexpr int kBoardSize = 15;
constexpr int kCells = kBoardSize * kBoardSize;

enum class Cell : uint8_t { Empty = 0, Black = 1, White = 2 };
enum class Color : uint8_t { Black = 1, White = 2 };
// Values match the engine's INFO rule contract: 1 = freestyle, 2 = Renju.
enum class RuleMode : uint8_t { Freestyle = 1, Renju = 2 };
enum class GameResult : uint8_t { Ongoing, BlackWin, WhiteWin, Draw };

constexpr Color opponent(Color c) {
  return c == Color::Black ? Color::White : Color::Black;
}
constexpr Color colorOf(Cell c) { return static_cast<Color>(c); }
constexpr Cell cellOf(Color c) { return static_cast<Cell>(c); }

// 15x15 gomoku board with turn tracking and rule-aware win detection.
// In Renju mode black's forbidden points are excluded from isPlayable() and
// black needs EXACTLY five; in freestyle five or more wins for either side.
class Board {
 public:
  explicit Board(RuleMode mode = RuleMode::Renju);

  static bool inBounds(int move) { return move >= 0 && move < kCells; }
  static int indexOf(int x, int y) { return y * kBoardSize + x; }
  static int xOf(int move) { return move % kBoardSize; }
  static int yOf(int move) { return move / kBoardSize; }

  Cell cell(int move) const { return cells_[move]; }
  const std::array<Cell, kCells>& rawCells() const { return cells_; }
  bool isEmpty(int move) const { return cells_[move] == Cell::Empty; }
  // Empty and legal under the current rule mode (Renju fouls arrive in 03).
  bool isPlayable(int move) const;

  // Turn-based moves. place() assumes isPlayable(move) holds.
  void place(int move);
  // Reverts the most recent place(); undefined if history is empty.
  void undo();

  // BOARD-sync API: set cells directly, then call finalizeSync().
  void setCell(int move, Cell c);
  void finalizeSync();  // recount, infer side to move, recompute result

  void reset(RuleMode mode);
  void setSideToMove(Color c) { sideToMove_ = c; }
  void setRuleMode(RuleMode mode) { mode_ = mode; }

  Color sideToMove() const { return sideToMove_; }
  RuleMode ruleMode() const { return mode_; }
  GameResult result() const { return result_; }
  int moveCount() const { return moveCount_; }

  // Indices of moves played since the last reset/finalizeSync, oldest first.
  // A rebuilt BOARD snapshot has no move history. Undo is only valid when
  // this is non-empty.
  std::vector<int> moveHistory() const;
  int lastMove() const { return history_.empty() ? -1 : history_.back().move; }

  // True if the stone of color c assumed to sit at `move` belongs to a line
  // of >= 5 (freestyle win semantics).
  static bool createsWin(const std::array<Cell, kCells>& cells, int move, Color c);

 private:
  GameResult computeResult() const;  // full-board scan (undo/sync paths)

  struct HistoryEntry {
    int move;
    GameResult resultBefore;
  };

  std::array<Cell, kCells> cells_{};
  std::vector<HistoryEntry> history_;
  Color sideToMove_ = Color::Black;
  RuleMode mode_ = RuleMode::Renju;
  GameResult result_ = GameResult::Ongoing;
  int moveCount_ = 0;
};

}  // namespace gomoku
