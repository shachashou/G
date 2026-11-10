// Cross-language parity probe for the deployed value network (ticket 07).
//
// This is NOT part of the engine: tests feed board/feature lines on stdin and
// compare the output with trainer/value_format.py (the integer reference) and
// with the exported PyTorch model. Every path here is the reference path; the
// incremental accumulator of ticket 11 must match it bit for bit.
//
//   value_probe --model PATH features < "stm <225 chars>"
//   value_probe --model PATH infer    < "<indices...>"
//   value_probe --model PATH board    < "stm <225 chars>"
//   value_probe --model PATH acc      < "stm <225 chars>"
//   value_probe --model PATH score    < "stm <225 chars>"
//   value_probe --model PATH bench    < "stm <225 chars> N"
//
// Board chars: '.'/'0' empty, 'x'/'1' black, 'o'/'2' white.
// ``board`` rebuilds the whole first layer and runs the scalar forward;
// ``acc`` rebuilds through the public accumulator API and runs the same
// forward, so both must be identical. ``score`` prints the black-absolute z,
// the black-side search score and the side-to-move search score as JSON.
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "board.h"
#include "nnue.h"
#include "value_net.h"

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
  std::string modelPath;
  std::string mode;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a == "--model" && i + 1 < argc) {
      modelPath = argv[++i];
    } else if (mode.empty() && a.rfind("--", 0) != 0) {
      mode = a;
    } else {
      std::cerr << "unknown/incomplete argument: " << a << "\n";
      return 2;
    }
  }
  if (modelPath.empty() || (mode != "features" && mode != "infer" &&
                            mode != "board" && mode != "acc" &&
                            mode != "score" && mode != "bench")) {
    std::cerr << "usage: value_probe --model PATH "
                 "{features|infer|board|acc|score|bench}\n";
    return 2;
  }

  ValueNetwork network;
  try {
    network.load(modelPath);
  } catch (const ValueNetError& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;

    if (mode == "infer") {
      std::istringstream in(line);
      std::vector<int> indices;
      int index = 0;
      while (in >> index) indices.push_back(index);
      try {
        std::cout << network.inferSparse(indices) << "\n";
      } catch (const ValueNetError& error) {
        std::cerr << error.what() << "\n";
        return 2;
      }
      continue;
    }

    std::istringstream in(line);
    int stm = -1;
    std::string text;
    if (!(in >> stm) || (stm != 0 && stm != 1) || !(in >> text)) {
      std::cerr << "bad line (need 'stm board'): " << line << "\n";
      return 2;
    }
    Board board(RuleMode::Renju);
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
      continue;
    }

    if (mode == "bench") {
      int iterations = 0;
      if (!(in >> iterations) || iterations <= 0 || iterations > 1000000) {
        std::cerr << "bench needs 1..1000000 iterations\n";
        return 2;
      }
      int64_t sink = 0;
      for (int i = 0; i < 1000; ++i) sink += network.evaluateBoard(board);
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < iterations; ++i) {
        sink += network.evaluateBoard(board);
      }
      const auto stop = std::chrono::steady_clock::now();
      const double totalNs =
          std::chrono::duration<double, std::nano>(stop - start).count();
      std::cout << "backend=scalar-full-rebuild iterations=" << iterations
                << " total_us=" << std::fixed << std::setprecision(1)
                << totalNs / 1000.0 << " ns_per_eval=" << std::setprecision(2)
                << totalNs / iterations << " sink=" << sink << "\n";
      continue;
    }

    int64_t z = 0;
    if (mode == "acc") {
      ValueAccumulator acc{};
      network.rebuildAccumulator(board, acc);
      z = network.inferAccumulator(acc);
    } else {
      z = network.evaluateBoard(board);
    }
    if (mode == "score") {
      const int black_score =
          network.sideToMoveScore(z, Color::Black);
      const int stm_score = network.sideToMoveScore(
          z, stm == 0 ? Color::Black : Color::White);
      std::cout << "{\"z\":" << z << ",\"black_score\":" << black_score
                << ",\"stm_score\":" << stm_score
                << ",\"side_to_move\":" << stm
                << ",\"result\":" << static_cast<int>(board.result()) << "}\n";
    } else {
      std::cout << z << "\n";
    }
  }
  return 0;
}
