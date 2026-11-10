#include "search.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>

#include "movegen.h"
#include "vcf.h"

namespace gomoku {

namespace {

constexpr int kInf = 2000000;
constexpr int kMate = 1000000;
constexpr int kMateBound = kMate - 1000;
constexpr uint8_t kExact = 0;
constexpr uint8_t kLower = 1;
constexpr uint8_t kUpper = 2;

struct Zobrist {
  std::array<std::array<uint64_t, 2>, kCells> stone{};
  uint64_t side = 0;

  Zobrist() {
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull);
    for (int m = 0; m < kCells; ++m) {
      stone[m][0] = rng();
      stone[m][1] = rng();
    }
    side = rng();
  }
};

const Zobrist& zobrist() {
  static const Zobrist z;
  return z;
}

uint64_t stoneKey(int move, Color c) {
  return zobrist().stone[move][c == Color::Black ? 0 : 1];
}

uint64_t hashOf(const Board& board) {
  uint64_t h = 0;
  for (int m = 0; m < kCells; ++m) {
    const Cell c = board.cell(m);
    if (c != Cell::Empty) h ^= stoneKey(m, colorOf(c));
  }
  if (board.sideToMove() == Color::White) h ^= zobrist().side;
  return h;
}

// Mate scores are ply-relative in the tree but absolute in the TT.
int scoreToTT(int score, int ply) {
  if (score > kMateBound) return score + ply;
  if (score < -kMateBound) return score - ply;
  return score;
}

int scoreFromTT(int score, int ply) {
  if (score > kMateBound) return score - ply;
  if (score < -kMateBound) return score + ply;
  return score;
}

}  // namespace

Search::Search(uint64_t ttBytes) {
  size_t count = static_cast<size_t>(ttBytes) / sizeof(TTEntry);
  size_t pow2 = 1;
  while (pow2 * 2 <= count) pow2 *= 2;
  tt_.resize(pow2);
  ttMask_ = pow2 - 1;
}

bool Search::timeUp() const {
  return std::chrono::steady_clock::now() >= deadline_;
}

bool Search::isRootVcfLoss(int move) const {
  return std::find(rootVcfLosses_.begin(), rootVcfLosses_.end(), move) !=
         rootVcfLosses_.end();
}

void Search::reset() {
  std::fill(tt_.begin(), tt_.end(), TTEntry{});
  generation_ = 1;
  for (auto& k : killers_) {
    k[0] = -1;
    k[1] = -1;
  }
  history_.fill(0);
  nodes_ = 0;
  stop_ = false;
  rootBestMove_ = -1;
}

SearchResult Search::findBestMove(Board& board, const SearchLimits& limits) {
  SearchResult result;
  Evaluator eval;
  eval.setNetwork(network_);
  eval.setValueNetwork(valueNetwork_);
  eval.init(board);
  const uint64_t rootHash = hashOf(board);

  ++generation_;
  if (generation_ == 0) generation_ = 1;
  for (auto& k : killers_) {
    k[0] = -1;
    k[1] = -1;
  }
  history_.fill(0);
  nodes_ = 0;
  stop_ = false;
  rootBestMove_ = -1;
  rootVcfLosses_.clear();
  useClock_ = limits.moveTimeMs > 0;
  if (useClock_) {
    deadline_ = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(limits.moveTimeMs);
  }

  std::vector<ScoredMove> rootMoves;
  generateCandidates(board, eval, rootMoves);
  if (rootMoves.empty()) return result;  // no legal move
  result.move = rootMoves[0].move;       // safe fallback

  // VCF: the vendored upstream solver is asked under a node budget. Only a
  // proven line may replace the search move; "not found", "budget exhausted"
  // and "interrupted" never do. (ticket 03)
  if (limits.vcfPolicy != VcfPolicy::Off) {
    const VcfLimits rootLimits{limits.vcfNodeLimit, limits.vcfMaxAttackPlies, 0};
    result.vcf = solveVcf(board, rootLimits);
    nodes_ += result.vcf.nodes;
    if (result.vcf.proven() && result.vcf.firstMove >= 0 &&
        board.isPlayable(result.vcf.firstMove)) {
      result.move = result.vcf.firstMove;
      result.score = kMate - 1;
      result.reachedDepth = 0;
      result.nodes = nodes_;
      result.fromVcf = true;
      return result;
    }

    // The other call site: a proven opponent VCF after a root move means that
    // move loses by force, so it is pruned at the root. Budget is per child;
    // the top candidates only, so the cost stays bounded.
    if (limits.vcfPolicy == VcfPolicy::RootAndChildren &&
        !(useClock_ && timeUp())) {
      const VcfLimits childLimits{limits.vcfChildNodeLimit,
                                  limits.vcfMaxAttackPlies, 0};
      const int checks =
          std::min<int>(limits.vcfChildChecks,
                        static_cast<int>(rootMoves.size()));
      for (int i = 0; i < checks && !stop_; ++i) {
        const int move = rootMoves[i].move;
        board.place(move);
        const VcfResult child = solveVcf(board, childLimits);
        const bool childWins = child.proven() && child.firstMove >= 0 &&
                               board.isPlayable(child.firstMove);
        board.undo();
        nodes_ += child.nodes;
        ++result.vcfChildChecks;
        if (childWins) {
          rootVcfLosses_.push_back(move);
          ++result.vcfChildLosses;
        }
        if (useClock_ && timeUp()) break;
      }
      // Never prune every root move: with no non-losing alternative the
      // search still has to return a legal move.
      if (rootVcfLosses_.size() >= rootMoves.size()) rootVcfLosses_.clear();
    }
  }

  for (int depth = 1; depth <= limits.maxDepth; ++depth) {
    rootBestMove_ = -1;
    const int score = alphaBeta(board, eval, depth, -kInf, kInf, 0, rootHash, -1);
    if (stop_) break;
    if (rootBestMove_ >= 0) result.move = rootBestMove_;
    result.score = score;
    result.reachedDepth = depth;
    if (std::abs(score) > kMateBound) break;  // forced win/loss found
  }
  result.nodes = nodes_;
  return result;
}

int Search::alphaBeta(Board& board, Evaluator& eval, int depth, int alpha,
                      int beta, int ply, uint64_t hash, int ttMove) {
  ++nodes_;
  if ((nodes_ & 1023) == 0 && useClock_ && timeUp()) stop_ = true;
  if (stop_) return 0;

  const GameResult res = board.result();
  if (res != GameResult::Ongoing) {
    if (res == GameResult::Draw) return 0;
    const Color winner = res == GameResult::BlackWin ? Color::Black : Color::White;
    return winner == board.sideToMove() ? (kMate - ply) : (-kMate + ply);
  }
  if (depth <= 0) return eval.evaluate(board);

  const int alphaOrig = alpha;
  TTEntry& entry = tt_[hash & ttMask_];
  if (ply > 0 && entry.key == hash && entry.depth >= depth) {
    const int s = scoreFromTT(entry.score, ply);
    if (entry.flag == kExact) return s;
    if (entry.flag == kLower && s > alpha) alpha = s;
    if (entry.flag == kUpper && s < beta) beta = s;
    if (alpha >= beta) return s;
  }

  std::vector<ScoredMove> moves;
  generateCandidates(board, eval, moves);
  if (moves.empty()) return eval.evaluate(board);

  for (ScoredMove& m : moves) {
    if (m.move == ttMove) {
      m.score += 10000000;
    } else if (ply < 64 && m.move == killers_[ply][0]) {
      m.score += 500000;
    } else if (ply < 64 && m.move == killers_[ply][1]) {
      m.score += 250000;
    }
    m.score += history_[m.move];
  }
  std::sort(moves.begin(), moves.end(),
            [](const ScoredMove& a, const ScoredMove& b) { return a.score > b.score; });

  int best = -kInf;
  int bestMove = moves.front().move;
  for (size_t i = 0; i < moves.size(); ++i) {
    const int move = moves[i].move;
    // A root move whose reply position is a proven opponent VCF loses by
    // force; searching it only burns nodes (ticket 03, RootAndChildren).
    if (ply == 0 && isRootVcfLoss(move)) continue;
    const Color c = board.sideToMove();
    eval.update(board, move, cellOf(c));
    const uint64_t childHash = hash ^ stoneKey(move, c) ^ zobrist().side;
    board.place(move);

    int score;
    if (i == 0) {
      score = -alphaBeta(board, eval, depth - 1, -beta, -alpha, ply + 1,
                         childHash, -1);
    } else {
      score = -alphaBeta(board, eval, depth - 1, -alpha - 1, -alpha, ply + 1,
                         childHash, -1);
      if (!stop_ && score > alpha && score < beta) {
        score = -alphaBeta(board, eval, depth - 1, -beta, -alpha, ply + 1,
                           childHash, -1);
      }
    }

    // update() needs the board in its pre-change state (stone still present),
    // so it must run BEFORE undo().
    eval.update(board, move, Cell::Empty);
    board.undo();
    if (stop_) return 0;

    if (score > best) {
      best = score;
      bestMove = move;
    }
    if (score > alpha) {
      alpha = score;
      if (ply == 0) rootBestMove_ = move;
    }
    if (alpha >= beta) {
      if (ply < 64 && killers_[ply][0] != move) {
        killers_[ply][1] = killers_[ply][0];
        killers_[ply][0] = move;
      }
      history_[move] += depth * depth;
      break;
    }
  }

  const uint8_t flag = best <= alphaOrig ? kUpper : (best >= beta ? kLower : kExact);
  if (entry.key != hash || entry.depth <= depth || entry.generation != generation_) {
    entry = TTEntry{hash, scoreToTT(best, ply), static_cast<int16_t>(bestMove),
                    static_cast<int8_t>(depth), flag, generation_};
  }
  return best;
}

}  // namespace gomoku
