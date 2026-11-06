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
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "board.h"
#include "eval.h"
#include "nnue.h"
#include "movegen.h"
#include "search.h"

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
  std::string rule = "renju";
  int radius = 1;
  int searchDepth = 2;
  int ttMb = 16;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a == "--weights" && i + 1 < argc) {
      weightsPath = argv[++i];
    } else if (a == "--backend" && i + 1 < argc) {
      backendName = argv[++i];
    } else if (a == "--depth" && i + 1 < argc) {
      searchDepth = std::stoi(argv[++i]);
    } else if (a == "--tt-mb" && i + 1 < argc) {
      ttMb = std::stoi(argv[++i]);
    } else if (a == "--rule" && i + 1 < argc) {
      rule = argv[++i];
    } else if (a == "--radius" && i + 1 < argc) {
      const std::string r = argv[++i];
      if (r != "1") { std::cerr << "radius must be 1\n"; return 2; }
      radius = 1;
    } else if (mode.empty() && !a.starts_with("--")) {
      mode = a;
    } else {
      std::cerr << "unknown/incomplete argument: " << a << "\n";
      return 2;
    }
  }
  const bool handcrafted = mode == "handcrafted";
  const bool structured = mode == "labels" || mode == "candidates";
  const bool searchscore = mode == "searchscore";
  if (rule != "renju") { std::cerr << "rule must be renju\n"; return 2; }
  if ((!handcrafted && !structured && !searchscore && weightsPath.empty()) ||
      (mode != "features" && mode != "infer" && mode != "board" &&
       mode != "bench" && mode != "searchscore" && !handcrafted && !structured)) {
    std::cerr << "usage: nnue_probe [--weights PATH] [--backend avx2|scalar] "
                 "{features|infer|board|bench|handcrafted|labels|candidates|searchscore}\n";
    return 2;
  }
  if (searchscore && (searchDepth < 1 || searchDepth > 8 || ttMb < 1 || ttMb > 256)) {
    std::cerr << "searchscore needs --depth 1..8 and --tt-mb 1..256\n";
    return 2;
  }
  NnueBackend backend = NnueBackend::Avx2;
  if (!nnueBackendFromName(backendName, backend)) {
    std::cerr << "unknown NNUE backend '" << backendName << "'\n";
    return 2;
  }

  NnueNetwork network;
  if (!handcrafted && !structured && !searchscore) {
    try {
      network.setBackend(backend);
      network.load(weightsPath);
    } catch (const NnueError& error) {
      std::cerr << error.what() << "\n";
      return 2;
    }
  }

  // Ticket 23: one reusable Search for the whole stdin batch. Each line gets
  // a clean TT (reset) so the score is reproducible, exactly like datagen.
  Search search(static_cast<uint64_t>(ttMb) * 1024ull * 1024ull);
  SearchLimits searchLimits;
  searchLimits.maxDepth = searchDepth;
  searchLimits.moveTimeMs = 0;  // fixed depth, no clock
  searchLimits.useVcf = true;

  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    std::istringstream in(line);
    int recordId = -1;
    if (structured && (!(in >> recordId) || recordId < 0)) {
      std::cerr << "bad record id\n"; return 2;
    }
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
    Board board(structured || searchscore ? RuleMode::Renju : RuleMode::Freestyle);
    std::string error;
    if (!parseBoard(text, board, stm, error)) {
      std::cerr << error << "\n";
      return 2;
    }
    if (searchscore) {
      search.reset();
      const SearchResult result = search.findBestMove(board, searchLimits);
      std::cout << result.score << "\n";
      continue;
    }
    if (structured) {
      int black = 0, white = 0;
      for (Cell c : board.rawCells()) {
        black += c == Cell::Black;
        white += c == Cell::White;
      }
      std::string extra;
      if ((in >> extra) || (stm == 0 ? black != white : black != white + 1)) {
        std::cerr << "inconsistent board/side to move\n"; return 2;
      }
      Evaluator eval;
      eval.init(board);
      std::cout << "{\"record_id\":" << recordId << ",\"result\":"
                << static_cast<int>(board.result()) << ",\"eval\":"
                << eval.evaluate(board);
      if (mode == "candidates") {
        std::vector<ScoredMove> moves;
        if (board.result() == GameResult::Ongoing) generateCandidates(board, eval, moves, radius);
        std::sort(moves.begin(), moves.end(), [](const ScoredMove& a, const ScoredMove& b) {
          return a.move < b.move;
        });
        std::cout << ",\"candidates\":[";
        bool first = true;
        for (const auto& move : moves) {
          board.place(move.move);
          if (!first) std::cout << ',';
          first = false;
          std::cout << "{\"move\":" << move.move << ",\"result\":"
                    << static_cast<int>(board.result()) << '}';
          board.undo();
        }
        std::cout << ']';
      }
      std::cout << "}\n";
    } else if (mode == "features") {
      const std::vector<int> indices = nnueSparseIndices(board);
      for (size_t i = 0; i < indices.size(); ++i) {
        if (i != 0) std::cout << ' ';
        std::cout << indices[i];
      }
      std::cout << "\n";
    } else if (mode == "board" || handcrafted) {
      Evaluator eval;
      if (!handcrafted) eval.setNetwork(&network);
      eval.init(board);
      std::cout << eval.evaluate(board) << "\n";
    } else if (network.formatVersion() == 3) {
      NnueFloatAccumulator acc;
      network.initAccumulator(board, acc);
      volatile float jitter = 0.0f;  // stays 0; defeats loop hoisting
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
