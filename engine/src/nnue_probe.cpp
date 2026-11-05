// Cross-language parity probe for ticket 09. Not part of the engine itself:
// Python tests feed board/feature lines on stdin and compare the outputs with
// trainer/features.py and trainer/weights_format.py.
//
//   nnue_probe --weights PATH --backend NAME features < "stm <225 chars>"
//   nnue_probe --weights PATH --backend NAME infer    < "stm <indices...>"
//   nnue_probe --weights PATH --backend NAME board    < "stm <225 chars>"
//   nnue_probe --weights PATH --backend NAME bench    < "stm <225 chars> N"
//
// Board chars: '.'/'0' empty, 'x'/'1' black, 'o'/'2' white.
// NAME is avx2 (default) or scalar. bench prints ns_per_eval for N evaluations.
#include <chrono>
#include <iomanip>
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
  std::string backendName = "avx2";
  std::string mode;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a == "--weights" && i + 1 < argc) {
      weightsPath = argv[++i];
    } else if (a == "--backend" && i + 1 < argc) {
      backendName = argv[++i];
    } else if (mode.empty()) {
      mode = a;
    }
  }
  if (weightsPath.empty() ||
      (mode != "features" && mode != "infer" && mode != "board" &&
       mode != "bench")) {
    std::cerr << "usage: nnue_probe --weights PATH [--backend avx2|scalar] "
                 "{features|infer|board|bench}\n";
    return 2;
  }
  NnueBackend backend = NnueBackend::Avx2;
  if (!nnueBackendFromName(backendName, backend)) {
    std::cerr << "unknown NNUE backend '" << backendName << "'\n";
    return 2;
  }

  NnueNetwork network;
  try {
    network.setBackend(backend);
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
    int iterations = 0;
    if (mode == "bench") {
      if (!(in >> iterations) || iterations <= 0 || iterations > 5000000) {
        std::cerr << "bench needs 1..5000000 iterations\n";
        return 2;
      }
    }
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
    } else if (mode == "board") {
      Evaluator eval;
      eval.setNetwork(&network);
      eval.init(board);
      std::cout << eval.evaluate(board) << "\n";
    } else {
      NnueAccumulator acc;
      network.initAccumulator(board, acc);
      volatile int32_t jitter = 0;  // stays 0; defeats loop hoisting
      int64_t sink = 0;
      for (int i = 0; i < 1000; ++i) {
        sink += network.inferAccumulator(acc, stm);
      }
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < iterations; ++i) {
        acc[0] += jitter;
        sink += network.inferAccumulator(acc, stm);
      }
      const auto stop = std::chrono::steady_clock::now();
      const double totalNs =
          std::chrono::duration<double, std::nano>(stop - start).count();
      std::cout << "backend=" << nnueBackendName(network.backend())
                << " iterations=" << iterations
                << " total_us=" << std::fixed << std::setprecision(1)
                << totalNs / 1000.0
                << " ns_per_eval=" << std::setprecision(2)
                << totalNs / iterations
                << " sink=" << sink << "\n";
    }
  }
  return 0;
}
