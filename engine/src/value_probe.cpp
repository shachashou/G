// Cross-language parity and performance probe for the deployed value network
// (tickets 07 and 11).
//
// This is NOT part of the engine: tests feed board/feature lines on stdin and
// compare the output with trainer/value_format.py (the integer reference) and
// with the exported PyTorch model. `board` and `infer` always run the scalar
// reference path; `adv`/`walk`/`bench` follow --backend so the incremental and
// AVX2 paths can be compared against the reference position by position.
//
//   value_probe --model PATH [--backend avx2|scalar|reference] MODE < stdin
//
// Modes:
//   features < "stm <225 chars>"      -> sparse feature indices
//   infer    < "<indices...>"         -> scalar reference z
//   board    < "stm <225 chars>"      -> full rebuild + scalar reference z
//   acc      < "stm <225 chars>"      -> rebuild + selected backend z
//   score    < "stm <225 chars>"      -> z, black score, side-to-move score JSON
//   bench    < "stm <225 chars> N [full|incremental]" -> timing
//   walk     < load/place/undo/state commands -> incremental session (JSON)
//
// Walk commands (one per line):
//   load <stm> <225 chars>   reset the board and rebuild the first layer
//   place <cell>             legally place side-to-move's stone at cell
//   undo                     undo the last placement
//   state                    report current z/reference/match without a change
//
// Board chars: '.'/'0' empty, 'x'/'1' black, 'o'/'2' white.
#include <algorithm>
#include <chrono>
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
  board.reset(RuleMode::Renju);
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

int sideToMoveInt(const Board& board) {
  return board.sideToMove() == Color::Black ? 0 : 1;
}

void printEval(const ValueNetwork& network, const Board& board,
               const ValueAccumulator& acc, const char* op, int cell) {
  const int64_t z = network.inferAccumulator(acc);
  const int64_t reference = network.evaluateBoard(board);
  std::cout << "{\"op\":\"" << op << "\"";
  if (cell >= 0) std::cout << ",\"cell\":" << cell;
  std::cout << ",\"side_to_move\":" << sideToMoveInt(board)
            << ",\"ply\":" << board.moveCount()
            << ",\"z\":" << z << ",\"reference\":" << reference
            << ",\"match\":" << (z == reference ? "true" : "false")
            << ",\"result\":" << static_cast<int>(board.result()) << "}\n";
}

void printWalkError(const std::string& message) {
  std::cout << "{\"op\":\"error\",\"ok\":false,\"error\":\"" << message
            << "\"}\n";
}

void runWalk(const ValueNetwork& network, std::istream& input) {
  Board board(RuleMode::Renju);
  ValueAccumulator acc{};
  bool initialized = false;
  // Stones placed since the last `load`.  Undo may only walk back over those;
  // the loaded position itself has no move history, so an underflow used to
  // call updateAccumulator with lastMove() == -1 and abort the probe.
  int placed = 0;

  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.size() >= 3 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
      line.erase(0, 3);  // tolerate a UTF-8 BOM from PowerShell pipelines
    }
    if (line.empty()) continue;
    std::istringstream in(line);
    std::string op;
    if (!(in >> op)) continue;

    if (op == "load") {
      int stm = -1;
      std::string text;
      if (!(in >> stm) || (stm != 0 && stm != 1) || !(in >> text)) {
        printWalkError("load needs 'stm board'");
        continue;
      }
      std::string error;
      if (!parseBoard(text, board, stm, error)) {
        printWalkError(error);
        continue;
      }
      network.rebuildAccumulator(board, acc);
      initialized = true;
      placed = 0;
      printEval(network, board, acc, "load", -1);
    } else if (op == "place") {
      int cell = -1;
      if (!(in >> cell)) {
        printWalkError("place needs a cell index");
        continue;
      }
      if (!initialized) {
        printWalkError("no position loaded");
        continue;
      }
      if (!Board::inBounds(cell) || !board.isPlayable(cell)) {
        printWalkError("illegal placement");
        continue;
      }
      const Color side = board.sideToMove();
      network.updateAccumulator(board, cell, cellOf(side), acc);
      board.place(cell);
      ++placed;
      printEval(network, board, acc, "place", cell);
    } else if (op == "undo") {
      if (!initialized || placed == 0) {
        printWalkError("nothing to undo");
        continue;
      }
      const int move = board.lastMove();
      // updateAccumulator expects the board in its pre-change state (the
      // stone is still present), then the board is rolled back.
      network.updateAccumulator(board, move, Cell::Empty, acc);
      board.undo();
      --placed;
      printEval(network, board, acc, "undo", move);
    } else if (op == "state") {
      if (!initialized) {
        printWalkError("no position loaded");
        continue;
      }
      printEval(network, board, acc, "state", -1);
    } else {
      printWalkError("unknown walk op '" + op + "'");
    }
  }
}

bool parseBenchKind(const std::string& text, std::string& kind) {
  if (text == "full" || text == "incremental") {
    kind = text;
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char* argv[]) {
  using namespace gomoku;
  std::string modelPath;
  std::string backendName = "reference";
  std::string mode;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a == "--model" && i + 1 < argc) {
      modelPath = argv[++i];
    } else if (a == "--backend" && i + 1 < argc) {
      backendName = argv[++i];
    } else if (mode.empty() && a.rfind("--", 0) != 0) {
      mode = a;
    } else {
      std::cerr << "unknown/incomplete argument: " << a << "\n";
      return 2;
    }
  }
  if (modelPath.empty() || (mode != "features" && mode != "infer" &&
                            mode != "board" && mode != "acc" &&
                            mode != "score" && mode != "bench" &&
                            mode != "walk")) {
    std::cerr << "usage: value_probe --model PATH [--backend NAME] "
                 "{features|infer|board|acc|score|bench|walk}\n";
    return 2;
  }

  ValueNetwork network;
  try {
    network.load(modelPath);
  } catch (const ValueNetError& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
  ValueBackend backend = ValueBackend::Scalar;
  ValueAccumulatorMode accumulator = ValueAccumulatorMode::FullRebuild;
  if (!valueEvalModeFromName(backendName, backend, accumulator)) {
    std::cerr << "--backend must be avx2, scalar or reference\n";
    return 2;
  }
  network.setBackend(backend);
  network.setAccumulatorMode(accumulator);

  if (mode == "walk") {
    runWalk(network, std::cin);
    return 0;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.size() >= 3 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) {
      line.erase(0, 3);
    }
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
      std::string kind = "full";
      if (!(in >> iterations) || iterations <= 0 || iterations > 1000000) {
        std::cerr << "bench needs 1..1000000 iterations\n";
        return 2;
      }
      if (in >> kind) {
        std::string parsed;
        if (!parseBenchKind(kind, parsed)) {
          std::cerr << "bench kind must be full or incremental\n";
          return 2;
        }
        kind = parsed;
      }
      int64_t sink = 0;
      const int warmup = std::min(1000, iterations);
      if (kind == "full") {
        for (int i = 0; i < warmup; ++i) sink += network.evaluateBoard(board);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
          sink += network.evaluateBoard(board);
        }
        const auto stop = std::chrono::steady_clock::now();
        const double totalNs =
            std::chrono::duration<double, std::nano>(stop - start).count();
        std::cout << "backend=full-rebuild-scalar kind=full iterations="
                  << iterations << " total_us=" << std::fixed
                  << std::setprecision(1) << totalNs / 1000.0
                  << " ns_per_eval=" << std::setprecision(2)
                  << totalNs / iterations << " sink=" << sink << "\n";
      } else {
        ValueAccumulator acc{};
        network.rebuildAccumulator(board, acc);
        for (int i = 0; i < warmup; ++i) sink += network.inferAccumulator(acc);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
          sink += network.inferAccumulator(acc);
        }
        const auto stop = std::chrono::steady_clock::now();
        const double totalNs =
            std::chrono::duration<double, std::nano>(stop - start).count();
        std::cout << "backend=" << network.evalModeName()
                  << " kind=incremental iterations=" << iterations
                  << " total_us=" << std::fixed << std::setprecision(1)
                  << totalNs / 1000.0 << " ns_per_eval=" << std::setprecision(2)
                  << totalNs / iterations << " sink=" << sink << "\n";
      }
      continue;
    }

    int64_t z = 0;
    if (mode == "acc") {
      ValueAccumulator acc{};
      network.rebuildAccumulator(board, acc);
      z = network.inferAccumulator(acc);
    } else {
      // board / score: the frozen scalar reference path.
      z = network.evaluateBoard(board);
    }
    if (mode == "score") {
      const int black_score = network.sideToMoveScore(z, Color::Black);
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
