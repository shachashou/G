#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "board.h"
#include "nnue.h"
#include "random_player.h"
#include "search.h"

namespace gomoku {

struct EngineConfig {
  bool random = false;    // --random: baseline random legal mover
  int fixedDepth = 0;     // --depth N: fixed depth, no clock
  int moveTimeMs = 1000;  // --movetime N: per-move budget (INFO overrides)
  bool useVcf = true;     // --no-vcf: M2 baseline for A/B comparisons
  std::string nnueWeights;  // --nnue PATH: NNUE evaluation (ticket 09)
  // --nnue-backend scalar|avx2: inference backend (ticket 10, default avx2).
  std::string nnueBackend = "avx2";
};

// Gomocup/piskvork protocol subset over stdin/stdout:
//   START [size]      -> OK / ERROR
//   RESTART           -> OK
//   BEGIN             -> engine's first move as black
//   TURN x,y          -> applies opponent move, replies with engine move
//   BOARD / x,y,c ... / DONE -> sync position, replies with engine move
//   INFO key value    -> no reply (rule 1=freestyle, 2=renju; time controls)
//   ABOUT             -> engine info line
//   END               -> exits
class Protocol {
 public:
  explicit Protocol(const EngineConfig& cfg);
  // Reads commands until END or EOF; returns process exit code.
  int run(std::istream& in, std::ostream& out);

 private:
  void handle(const std::string& line, std::istream& in, std::ostream& out);
  // Consumes "x,y,c" lines until DONE. Returns false after sending an error.
  bool readBoard(std::istream& in, std::ostream& out);
  void handleInfo(const std::string& arg);
  void playEngineMove(std::ostream& out);
  int effectiveMoveTimeMs() const;

  EngineConfig cfg_;
  Board board_;
  RandomPlayer player_;
  // Weights are loaded once at startup and shared with Search via a borrowed
  // pointer. Declared before search_ so it outlives the search object.
  std::unique_ptr<NnueNetwork> nnue_;
  bool nnueActive_ = false;
  // Allocated only for the search engine: the random baseline stays tiny.
  std::unique_ptr<Search> search_;
  int timeoutTurnMs_ = 0;
  int timeoutMatchMs_ = 0;
  int timeLeftMs_ = 0;
};

}  // namespace gomoku
