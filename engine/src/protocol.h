#pragma once

#include <iosfwd>
#include <string>

#include "board.h"
#include "random_player.h"

namespace gomoku {

// Gomocup/piskvork protocol subset over stdin/stdout (ticket 02):
//   START [size]      -> OK / ERROR
//   RESTART           -> OK
//   BEGIN             -> engine's first move as black
//   TURN x,y          -> applies opponent move, replies with engine move
//   BOARD / x,y,c ... / DONE -> sync position, replies with engine move
//   INFO key value    -> no reply (rule 1=freestyle, 2=renju)
//   ABOUT             -> engine info line
//   END               -> exits
class Protocol {
 public:
  // Reads commands until END or EOF; returns process exit code.
  int run(std::istream& in, std::ostream& out);

 private:
  void handle(const std::string& line, std::istream& in, std::ostream& out);
  // Consumes "x,y,c" lines until DONE. Returns false after sending an error.
  bool readBoard(std::istream& in, std::ostream& out);
  void handleInfo(const std::string& arg);
  void playEngineMove(std::ostream& out);

  Board board_;
  RandomPlayer player_;
};

}  // namespace gomoku
