// Ticket 07: self-play data generation worker.
//
// One process owns one shard file, one RNG stream and one Search object. Many
// processes are launched by tools/datagen.py; they share no mutable state.
//
// Modes:
//   generate: --out FILE --games N [--depth 6 ...]
//   verify:   --verify FILE [--verify-samples 32]
//
// Offline verification replays sampled records: the board is rebuilt from the
// two bit planes and searched again with the config stored in the file header.
// Generation clears the TT before every move, so a fresh single-position search
// reproduces the stored score exactly.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "binpack.h"
#include "board.h"
#include "eval.h"
#include "search.h"
#include "version.h"

namespace {

using namespace gomoku;

struct Config {
  std::string out;
  std::string verifyPath;
  int games = 0;
  uint64_t seed = 0;
  uint32_t shardId = 0;
  int depth = 6;
  uint16_t ttMb = 16;
  bool useVcf = true;
  uint16_t vcfNodeLimit = 5000;
  RuleMode rule = RuleMode::Renju;
  int openingPlies = 3;
  int openingRegion = 7;
  double openingTemp = 1.0;
  int maxPlies = 225;
  int verifySamples = 32;
};

void printUsage() {
  std::cout << "Usage: datagen [options]\n"
            << "Generate mode:\n"
            << "  --out FILE           output shard (.binpack), required\n"
            << "  --games N            self-play games to play, required\n"
            << "  --seed N             base RNG seed (default 0)\n"
            << "  --shard-id N         shard/worker id (default 0)\n"
            << "  --depth N            fixed search depth (default 6)\n"
            << "  --tt-mb N            transposition table MB (default 16)\n"
            << "  --no-vcf             disable root VCF solver\n"
            << "  --vcf-nodes N        VCF node limit (default 5000)\n"
            << "  --rule 1|2           1=freestyle, 2=Renju (default 2)\n"
            << "  --opening-plies N    random opening plies, 0..8 (default 3)\n"
            << "  --opening-region N   odd side length of centre square (default 7)\n"
            << "  --opening-temp T     softmax temperature, 0=greedy (default 1.0)\n"
            << "  --max-plies N        cap game length, 1..225 (default 225)\n"
            << "Verify mode:\n"
            << "  --verify FILE        replay sampled records and compare scores\n"
            << "  --verify-samples N   number of positions to replay (default 32)\n"
            << "  --help               show this help\n"
            << "  --version            print version and exit\n";
}

int usageError(const std::string& message) {
  std::cerr << "datagen: " << message << "\n";
  return 2;
}

int parseInt(const std::string& s) {
  try {
    return std::stoi(s);
  } catch (...) {
    return -1;
  }
}

double parseDouble(const std::string& s) {
  try {
    return std::stod(s);
  } catch (...) {
    return -1.0;
  }
}

uint64_t parseU64(const std::string& s) {
  try {
    return std::stoull(s);
  } catch (...) {
    return 0;
  }
}

uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t gameSeed(const Config& cfg, int gameIndex) {
  return mix64(cfg.seed ^ (static_cast<uint64_t>(cfg.shardId) *
                           0x9E3779B97F4A7C15ull) ^
               (static_cast<uint64_t>(gameIndex) * 0xD1B54A32D192ED03ull));
}

// Softmax over the evaluator's move heuristic restricted to the centre square.
// Higher temperature flattens the distribution; 0 picks the best move.
int sampleOpening(const Board& board, const Config& cfg,
                  std::mt19937_64& rng) {
  const int half = cfg.openingRegion / 2;
  const int centre = kBoardSize / 2;
  Evaluator eval;
  eval.init(board);

  std::vector<int> moves;
  std::vector<double> heuristics;
  for (int y = centre - half; y <= centre + half; ++y) {
    for (int x = centre - half; x <= centre + half; ++x) {
      const int m = Board::indexOf(x, y);
      if (!board.isPlayable(m)) continue;
      moves.push_back(m);
      heuristics.push_back(
          static_cast<double>(eval.moveHeuristic(board, m, board.sideToMove())));
    }
  }
  if (moves.empty()) return -1;

  const double best = *std::max_element(heuristics.begin(), heuristics.end());
  if (cfg.openingTemp <= 0.0) {
    std::vector<size_t> bestIndices;
    for (size_t i = 0; i < heuristics.size(); ++i) {
      if (heuristics[i] == best) bestIndices.push_back(i);
    }
    std::uniform_int_distribution<size_t> pick(0, bestIndices.size() - 1);
    return moves[bestIndices[pick(rng)]];
  }

  const double scale = std::max(0.05, 20.0 * cfg.openingTemp);
  std::vector<double> weights;
  weights.reserve(heuristics.size());
  for (double h : heuristics) weights.push_back(std::exp((h - best) / scale));
  std::discrete_distribution<size_t> pick(weights.begin(), weights.end());
  return moves[pick(rng)];
}

GameResult playGame(Board& board, Search& search,
                    binpack::ShardWriter& writer, const Config& cfg,
                    std::mt19937_64& rng, const SearchLimits& limits) {
  board.reset(cfg.rule);
  writer.beginGame();

  // Random opening: these positions are intentionally NOT recorded.
  for (int ply = 0; ply < cfg.openingPlies; ++ply) {
    const int move = sampleOpening(board, cfg, rng);
    if (move < 0) return GameResult::Draw;  // no legal move: draw
    board.place(move);
    if (board.result() != GameResult::Ongoing) return board.result();
  }

  while (board.result() == GameResult::Ongoing &&
         board.moveCount() < cfg.maxPlies) {
    // A clean TT makes this score reproducible by a standalone replay.
    search.reset();
    const SearchResult result = search.findBestMove(board, limits);
    if (result.move < 0 || !board.isPlayable(result.move)) {
      return GameResult::Draw;  // e.g. every remaining point is a Renju foul
    }
    writer.appendRecord(board, result.score, result.move);
    if (!writer.ok()) return GameResult::Draw;
    board.place(result.move);
  }
  // Hitting the ply cap without a winner is a draw (also covers full board).
  return board.result() == GameResult::Ongoing ? GameResult::Draw
                                               : board.result();
}

int runGenerate(const Config& cfg) {
  binpack::Header header;
  header.searchDepth = static_cast<uint32_t>(cfg.depth);
  header.shardId = cfg.shardId;
  header.baseSeed = cfg.seed;
  header.ruleMode = static_cast<uint8_t>(cfg.rule);
  header.useVcf = cfg.useVcf ? 1 : 0;
  header.ttMb = cfg.ttMb;
  header.vcfNodeLimit = cfg.vcfNodeLimit;
  header.openingTempX100 = static_cast<uint16_t>(
      std::llround(std::clamp(cfg.openingTemp, 0.0, 600.0) * 100.0));

  binpack::ShardWriter writer;
  std::string error;
  if (!writer.open(cfg.out, header, &error)) {
    std::cerr << "datagen: " << error << "\n";
    return 3;
  }

  SearchLimits limits;
  limits.maxDepth = cfg.depth;
  limits.moveTimeMs = 0;  // fixed depth, no clock
  limits.useVcf = cfg.useVcf;
  limits.vcfNodeLimit = cfg.vcfNodeLimit;

  Search search(static_cast<uint64_t>(cfg.ttMb) * 1024ull * 1024ull);
  Board board(cfg.rule);
  int blackWins = 0;
  int whiteWins = 0;
  int draws = 0;

  const auto start = std::chrono::steady_clock::now();
  for (int g = 0; g < cfg.games; ++g) {
    std::mt19937_64 rng(gameSeed(cfg, g));
    const GameResult result = playGame(board, search, writer, cfg, rng, limits);
    writer.endGame(result);
    if (result == GameResult::BlackWin) {
      ++blackWins;
    } else if (result == GameResult::WhiteWin) {
      ++whiteWins;
    } else {
      ++draws;
    }
    if (!writer.ok()) break;
  }
  const double seconds = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start)
                             .count();
  if (!writer.finish(&error)) {
    std::cerr << "datagen: " << error << "\n";
    return 3;
  }
  const uint64_t records = writer.recordCount();
  const double rate = seconds > 0.0 ? static_cast<double>(records) / seconds : 0.0;
  std::cout << "STATS"
            << " shard=" << cfg.shardId << " games=" << (blackWins + whiteWins + draws)
            << " records=" << records << " games_with_records=" << writer.gameCount()
            << " seconds=" << std::fixed << std::setprecision(3) << seconds
            << " pos_per_sec=" << std::setprecision(2) << rate
            << " black_wins=" << blackWins << " white_wins=" << whiteWins
            << " draws=" << draws << "\n";
  return 0;
}

int runVerify(const Config& cfg) {
  binpack::ShardReader reader;
  std::string error;
  if (!reader.open(cfg.verifyPath, &error)) {
    std::cerr << "datagen: " << error << "\n";
    return 3;
  }
  const binpack::Header& header = reader.header();
  const uint64_t total = reader.availableRecords();
  const uint64_t samples =
      std::min<uint64_t>(static_cast<uint64_t>(cfg.verifySamples), total);
  const uint64_t stride = std::max<uint64_t>(1, total / std::max<uint64_t>(1, samples));

  SearchLimits limits;
  limits.maxDepth = static_cast<int>(header.searchDepth);
  limits.moveTimeMs = 0;
  limits.useVcf = header.useVcf != 0;
  limits.vcfNodeLimit = header.vcfNodeLimit;

  Search search(static_cast<uint64_t>(header.ttMb) * 1024ull * 1024ull);
  Board board(header.ruleMode == 1 ? RuleMode::Freestyle : RuleMode::Renju);

  uint64_t checked = 0;
  uint64_t mismatches = 0;
  uint64_t saturated = 0;
  uint64_t unknownWdl = 0;
  uint64_t errors = 0;
  for (uint64_t i = 0; i < total && checked < samples; i += stride) {
    binpack::Record record;
    if (!reader.read(i, record, &error)) {
      ++errors;
      std::cerr << "datagen: record " << i << ": " << error << "\n";
      continue;
    }
    if (record.wdl == binpack::kWdlUnknown) ++unknownWdl;
    if (!binpack::boardFromRecord(record, board.ruleMode(), board, &error)) {
      ++errors;
      std::cerr << "datagen: record " << i << ": " << error << "\n";
      continue;
    }
    search.reset();
    const SearchResult result = search.findBestMove(board, limits);
    const int16_t expected = binpack::encodeScore(result.score);
    if (expected != record.score) {
      ++mismatches;
      if (mismatches <= 5) {
        std::cerr << "datagen: score mismatch at record " << i
                  << " stored=" << record.score << " replay=" << expected
                  << " (raw=" << result.score << ")\n";
      }
    }
    if (record.score == binpack::kScoreMax ||
        record.score == -binpack::kScoreMax) {
      ++saturated;
    }
    ++checked;
  }

  std::cout << "VERIFY"
            << " records=" << header.recordCount << " available=" << total
            << " sampled=" << checked << " mismatches=" << mismatches
            << " saturated=" << saturated << " unknown_wdl=" << unknownWdl
            << " errors=" << errors << " depth=" << header.searchDepth
            << " tt_mb=" << header.ttMb << " vcf=" << int(header.useVcf) << "\n";
  return (mismatches || errors || unknownWdl) ? 4 : 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a(argv[i]);
    auto take = [&](std::string& value) -> bool {
      if (i + 1 >= argc) return false;
      value = argv[++i];
      return true;
    };
    std::string v;
    if (a == "--help") {
      printUsage();
      return 0;
    }
    if (a == "--version") {
      std::cout << gomoku::kEngineName << " datagen " << gomoku::kEngineVersion
                << "\n";
      return 0;
    }
    if (a == "--no-vcf") {
      cfg.useVcf = false;
      continue;
    }
    if (!take(v)) return usageError(std::string(a) + " needs a value");
    if (a == "--out") {
      cfg.out = v;
    } else if (a == "--verify") {
      cfg.verifyPath = v;
    } else if (a == "--games") {
      cfg.games = parseInt(v);
    } else if (a == "--seed") {
      cfg.seed = parseU64(v);
    } else if (a == "--shard-id") {
      cfg.shardId = static_cast<uint32_t>(parseU64(v));
    } else if (a == "--depth") {
      cfg.depth = parseInt(v);
    } else if (a == "--tt-mb") {
      cfg.ttMb = static_cast<uint16_t>(parseInt(v));
    } else if (a == "--vcf-nodes") {
      cfg.vcfNodeLimit = static_cast<uint16_t>(parseInt(v));
    } else if (a == "--rule") {
      if (v != "1" && v != "2") return usageError("--rule must be 1 or 2");
      cfg.rule = v == "1" ? RuleMode::Freestyle : RuleMode::Renju;
    } else if (a == "--opening-plies") {
      cfg.openingPlies = parseInt(v);
    } else if (a == "--opening-region") {
      cfg.openingRegion = parseInt(v);
    } else if (a == "--opening-temp") {
      cfg.openingTemp = parseDouble(v);
    } else if (a == "--max-plies") {
      cfg.maxPlies = parseInt(v);
    } else if (a == "--verify-samples") {
      cfg.verifySamples = parseInt(v);
    } else {
      return usageError("unknown option " + std::string(a));
    }
  }

  if (cfg.depth < 1 || cfg.depth > 32) return usageError("--depth out of range");
  if (cfg.ttMb < 1 || cfg.ttMb > 1024) return usageError("--tt-mb out of range");
  if (cfg.vcfNodeLimit == 0) return usageError("--vcf-nodes must be positive");
  if (cfg.openingPlies < 0 || cfg.openingPlies > 8) {
    return usageError("--opening-plies must be 0..8");
  }
  if (cfg.openingRegion < 1 || cfg.openingRegion > kBoardSize ||
      cfg.openingRegion % 2 == 0) {
    return usageError("--opening-region must be an odd value 1..15");
  }
  if (cfg.openingTemp < 0.0 || cfg.openingTemp > 600.0) {
    return usageError("--opening-temp out of range");
  }
  if (cfg.maxPlies < 1 || cfg.maxPlies > kCells) {
    return usageError("--max-plies must be 1..225");
  }
  if (cfg.verifySamples < 1) return usageError("--verify-samples must be positive");

  if (!cfg.verifyPath.empty()) return runVerify(cfg);
  if (cfg.out.empty()) return usageError("generate mode needs --out FILE");
  if (cfg.games < 1) return usageError("generate mode needs --games N > 0");
  return runGenerate(cfg);
}
