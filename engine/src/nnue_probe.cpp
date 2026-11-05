// Cross-language parity probe for ticket 09. Not part of the engine itself:
// Python tests feed board/feature lines on stdin and compare the outputs with
// trainer/features.py and trainer/weights_format.py.
//
//   nnue_probe --weights PATH features   < "stm <225 board chars>"
//   nnue_probe --weights PATH infer      < "stm <sparse indices...>"
//   nnue_probe --weights PATH board      < "stm <225 board chars>"
//
// Board chars: '.'/'0' empty, 'x'/'1' black, 'o'/'2' white.
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "board.h"
#include "eval.h"
#include "nnue.h"

namespace {

using namespace gomoku;

bool parseBoard(const std::string& text, Board& board, int stm,
                std::string& error) {
  if (text.size() != static_cast<size_t>(kCells)) {
    error = "board string must have exactly 225 characters";
    return false;
  }
  for (int m = 0; m < kCells; ++m) {
    const char c = text[static_cast<size_t>(m)];
    Cell cell = Cell::Empty;
    if (c == 'x' || c == '1') {
      cell = Cell::Black;
    } else if (c == 'o' || c == '2') {
      cell = Cell::White;
    } else if (c != '.' && c != '0') {
      error = std::string("invalid board character '") + c + "'";
      return false;
    }
    board.setCell(m, cell);
  }
  board.finalizeSync();
  board.setSideToMove(stm == 0 ? Color::Black : Color::White);
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  using namespace gomoku;
  std::string weightsPath;
  std::string mode;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a == "--weights" && i + 1 < argc) {
      weightsPath = argv[++i];
    } else if (mode.empty()) {
      mode = a;
    }
  }
  if (weightsPath.empty() ||
      (mode != "features" && mode != "infer" && mode != "board")) {
    std::cerr << "usage: nnue_probe --weights PATH "
                 "{features|infer|board}\n";
    return 2;
  }

  NnueNetwork network;
  try {
    network.load(weightsPath);
  } catch (const NnueError& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::istringstream in(line);
    int stm = -1;
    if (!(in >> stm) || (stm != 0 && stm != 1)) {
      std::cerr << "bad stm in line: " << line << "\n";
      return 2;
    }

    if (mode == "infer") {
      std::vector<int> indices;
      int index = 0;
      while (in >> index) indices.push_back(index);
      try {
        std::cout << network.infer(indices, stm) << "\n";
      } catch (const NnueError& error) {
        std::cerr << error.what() << "\n";
        return 2;
      }
      continue;
    }

    std::string text;
    in >> text;
    Board board(RuleMode::Freestyle);
    std::string error;
    if (!parseBoard(text, board, stm, error)) {
      std::cerr << error << "\n";
      return 2;
    }
    if (mode == "features") {
      const std::vector<int> indices = nnueSparseIndices(board);
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i != 0) std::cout << ' ';
        std::cout << indices[i];
      }
      std::cout << "\n";
    } else {
      Evaluator eval;
      eval.setNetwork(&network);
      eval.init(board);
      std::cout << eval.evaluate(board) << "\n";
    }
  }
  return 0;
}
