#include "executor_protocol.h"

#include <array>
#include <exception>
#include <iostream>
#include <optional>
#include <utility>

#include "version.h"

namespace gomoku {

namespace {

using json::Value;

const char* resultName(GameResult result) {
  switch (result) {
    case GameResult::Ongoing: return "ongoing";
    case GameResult::BlackWin: return "black_win";
    case GameResult::WhiteWin: return "white_win";
    case GameResult::Draw: return "draw";
  }
  return "unknown";
}

RuleMode ruleFrom(const Value* value) {
  if (!value) return RuleMode::Renju;
  return value->asString("renju") == "freestyle" ? RuleMode::Freestyle
                                                 : RuleMode::Renju;
}

Value pointValue(int move) {
  std::vector<Value> point;
  point.push_back(json::numberValue(Board::xOf(move)));
  point.push_back(json::numberValue(Board::yOf(move)));
  return json::arrayValue(std::move(point));
}

bool parsePoint(const Value& request, int& x, int& y, std::string& error) {
  const Value* xv = request.find("x");
  const Value* yv = request.find("y");
  if (!xv || !yv || xv->type != Value::Type::Number ||
      yv->type != Value::Type::Number) {
    error = "x and y must be numbers";
    return false;
  }
  x = static_cast<int>(xv->asInt());
  y = static_cast<int>(yv->asInt());
  return true;
}

}  // namespace

// NNUE loading mirrors Protocol's constructor. The two front ends are
// mutually exclusive process modes (--executor vs default Gomocup), so this
// stays independent and the existing protocol path is not refactored.
ExecutorProtocol::ExecutorProtocol(const EngineConfig& cfg, std::ostream& out)
    : cfg_(cfg), out_(out), executor_(RuleMode::Renju) {
  search_ = std::make_unique<Search>();
  if (!cfg_.valueModel.empty()) {
    // Ticket 07: strict load. A malformed or incompatible model propagates to
    // main() so the process exits; it never plays with another evaluator while
    // the response still claims NNUE.
    valueNet_ = std::make_unique<ValueNetwork>();
    valueNet_->load(cfg_.valueModel);
    search_->setValueNetwork(valueNet_.get());
  }
  if (!cfg_.nnueWeights.empty()) {
    try {
      nnue_ = std::make_unique<NnueNetwork>();
      NnueBackend backend = NnueBackend::Avx2;
      if (!nnueBackendFromName(cfg_.nnueBackend, backend)) {
        std::cerr << "NNUE: unknown backend '" << cfg_.nnueBackend
                  << "', using avx2\n";
      }
      nnue_->setBackend(backend);
      nnue_->load(cfg_.nnueWeights);
      search_->setNetwork(nnue_.get());
      nnueActive_ = true;
    } catch (const NnueError& error) {
      nnue_.reset();
      std::cerr << error.what() << "\n"
                << "NNUE: falling back to handcrafted evaluation\n";
    }
  }
}

int ExecutorProtocol::run(std::istream& in) {
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    Value response;
    bool quit = false;
    try {
      response = dispatch(json::parse(line), quit);
    } catch (const std::exception& error) {
      response = errorValue(error.what());
    }
    out_ << json::dump(response) << '\n';
    out_.flush();
    if (quit) break;
  }
  return 0;
}

Value ExecutorProtocol::errorValue(const std::string& message) const {
  Value response = json::objectValue();
  json::set(response, "ok", json::booleanValue(false));
  json::set(response, "error", json::stringValue(message));
  return response;
}

Value ExecutorProtocol::valueModelValue() const {
  Value out = json::objectValue();
  if (!valueNet_) {
    json::set(out, "loaded", json::booleanValue(false));
    return out;
  }
  const ValueModelInfo& info = valueNet_->info();
  json::set(out, "loaded", json::booleanValue(true));
  json::set(out, "path", json::stringValue(info.path));
  json::set(out, "model_id", json::stringValue(info.model_id));
  json::set(out, "arch", json::stringValue(info.arch()));
  json::set(out, "rule", json::stringValue(info.rule));
  json::set(out, "feature_spec", json::stringValue(info.feature_spec));
  json::set(out, "value_contract", json::stringValue(info.value_contract));
  json::set(out, "ea", json::numberValue(info.ea));
  json::set(out, "act_max_int", json::numberValue(info.act_max_int));
  json::set(out, "scale", json::numberValue(1 << info.ea));
  json::set(out, "bytes", json::numberValue(info.file_bytes));
  json::set(out, "fnv1a",
            json::numberValue(static_cast<long long>(info.fnv1a)));
  json::set(out, "training_steps", json::numberValue(info.training_steps));
  json::set(out, "created_at", json::stringValue(info.created_at));
  std::vector<Value> labels;
  labels.reserve(info.label_versions.size());
  for (const std::string& version : info.label_versions) {
    labels.push_back(json::stringValue(version));
  }
  json::set(out, "label_versions", json::arrayValue(std::move(labels)));
  json::set(out, "backend", json::stringValue("scalar-full-rebuild"));
  // Ticket 08 owns promotion; a successfully loaded candidate is playable but
  // is never presented as the promoted official version.
  json::set(out, "promoted", json::booleanValue(false));
  json::set(out, "status", json::stringValue("candidate-unpromoted"));
  return out;
}

void ExecutorProtocol::addState(Value& target, const PositionState& s) const {
  std::vector<Value> cells;
  cells.reserve(kCells);
  for (Cell c : s.cells) {
    cells.push_back(json::numberValue(static_cast<int>(c)));
  }
  json::set(target, "cells", json::arrayValue(std::move(cells)));
  json::set(target, "side_to_move",
            json::stringValue(s.sideToMove == Color::Black ? "black" : "white"));
  json::set(target, "result", json::stringValue(resultName(s.result)));
  json::set(target, "rule",
            json::stringValue(s.rule == RuleMode::Renju ? "renju"
                                                        : "freestyle"));
  json::set(target, "move_count", json::numberValue(s.moveCount));
  std::vector<Value> history;
  history.reserve(s.history.size());
  for (int m : s.history) history.push_back(pointValue(m));
  json::set(target, "history", json::arrayValue(std::move(history)));
  json::set(target, "last_move",
            s.lastMove < 0 ? json::nullValue() : pointValue(s.lastMove));
}

Value ExecutorProtocol::vcfValue(const VcfResult& result) const {
  Value out = json::objectValue();
  json::set(out, "status", json::stringValue(vcfStatusName(result.status)));
  json::set(out, "proven", json::booleanValue(result.proven()));
  json::set(out, "unknown", json::booleanValue(result.unknown()));
  json::set(out, "first_move",
            result.firstMove < 0 ? json::nullValue()
                                 : pointValue(result.firstMove));
  std::vector<Value> variation;
  variation.reserve(result.variation.size());
  for (int m : result.variation) variation.push_back(pointValue(m));
  json::set(out, "variation", json::arrayValue(std::move(variation)));
  Value end = json::objectValue();
  json::set(end, "kind", json::stringValue(vcfEndKindName(result.evidence.kind)));
  json::set(end, "first_point",
            result.evidence.firstPoint < 0
                ? json::nullValue()
                : pointValue(result.evidence.firstPoint));
  json::set(end, "second_point",
            result.evidence.secondPoint < 0
                ? json::nullValue()
                : pointValue(result.evidence.secondPoint));
  json::set(out, "end", std::move(end));
  Value budget = json::objectValue();
  json::set(budget, "max_nodes",
            json::numberValue(static_cast<long long>(result.limits.nodeLimit)));
  json::set(budget, "max_attack_plies",
            json::numberValue(result.limits.maxAttackPlies));
  json::set(budget, "time_ms", json::numberValue(result.limits.timeLimitMs));
  json::set(out, "budget", std::move(budget));
  json::set(out, "nodes",
            json::numberValue(static_cast<long long>(result.nodes)));
  json::set(out, "elapsed_ms", json::numberValue(result.elapsedMs));
  json::set(out, "message", json::stringValue(result.message));
  return out;
}

Value ExecutorProtocol::vcfCommand(const Value& request) const {
  VcfLimits limits;
  limits.nodeLimit = cfg_.vcfNodeLimit;
  limits.maxAttackPlies = 10;
  limits.timeLimitMs = 0;
  if (const Value* value = request.find("max_nodes")) {
    if (value->type != Value::Type::Number) {
      return errorValue("max_nodes must be a number");
    }
    limits.nodeLimit = static_cast<uint64_t>(value->asInt(0));
  }
  if (const Value* value = request.find("max_attack_plies")) {
    if (value->type != Value::Type::Number) {
      return errorValue("max_attack_plies must be a number");
    }
    limits.maxAttackPlies = static_cast<int>(value->asInt(0));
  }
  if (const Value* value = request.find("time_ms")) {
    if (value->type != Value::Type::Number) {
      return errorValue("time_ms must be a number");
    }
    limits.timeLimitMs = static_cast<int>(value->asInt(0));
  }
  if (limits.nodeLimit < 1 || limits.nodeLimit > 100000000ull) {
    return errorValue("max_nodes must be in 1..100000000");
  }
  if (limits.maxAttackPlies < 1 || limits.maxAttackPlies > 255) {
    return errorValue("max_attack_plies must be in 1..255");
  }
  if (limits.timeLimitMs < 0 || limits.timeLimitMs > 600000) {
    return errorValue("time_ms must be in 0..600000");
  }

  const Board& board = executor_.board();
  VcfResult result = solveVcf(board, limits);
  Value response = json::objectValue();
  json::set(response, "ok", json::booleanValue(true));
  json::set(response, "rule",
            json::stringValue(board.ruleMode() == RuleMode::Renju ? "renju"
                                                                  : "freestyle"));
  json::set(response, "attacker",
            json::stringValue(board.sideToMove() == Color::Black ? "black"
                                                                 : "white"));
  json::set(response, "vcf", vcfValue(result));
  // The variation has already been replayed through Board::isPlayable by the
  // adapter; say so explicitly so clients can tell evidence was checked.
  json::set(response, "evidence_checked", json::booleanValue(true));
  return response;
}

Value ExecutorProtocol::think(const Value& request) const {
  const std::vector<int> legal = executor_.legalMoves();
  if (legal.empty()) {
    return errorValue("no legal move (game over or full board)");
  }
  SearchLimits limits;
  limits.vcfPolicy = cfg_.vcfPolicy;
  limits.vcfNodeLimit = cfg_.vcfNodeLimit;
  if (const Value* value = request.find("vcf_policy")) {
    if (value->type != Value::Type::String ||
        !vcfPolicyFromName(value->asString(), limits.vcfPolicy)) {
      return errorValue("vcf_policy must be off, root or root_and_children");
    }
  }
  if (const Value* value = request.find("vcf_nodes")) {
    const long long nodes = value->asInt(0);
    if (value->type != Value::Type::Number || nodes < 1) {
      return errorValue("vcf_nodes must be a positive number");
    }
    limits.vcfNodeLimit = static_cast<uint64_t>(nodes);
  }
  if (const Value* value = request.find("vcf_plies")) {
    const long long plies = value->asInt(0);
    if (value->type != Value::Type::Number || plies < 1 || plies > 255) {
      return errorValue("vcf_plies must be in 1..255");
    }
    limits.vcfMaxAttackPlies = static_cast<int>(plies);
  }
  const int depth = request.find("depth")
                        ? static_cast<int>(request.find("depth")->asInt(0))
                        : 0;
  const int moveTime = request.find("movetime_ms")
                           ? static_cast<int>(request.find("movetime_ms")->asInt(0))
                           : 0;
  if (depth > 0) {
    limits.maxDepth = depth;
    limits.moveTimeMs = 0;
  } else {
    limits.maxDepth = 8;
    limits.moveTimeMs = moveTime > 0 ? moveTime : cfg_.moveTimeMs;
  }
  Board board = executor_.board();  // search mutates its own copy
  const SearchResult result = search_->findBestMove(board, limits);
  int move = result.move;
  if (move < 0 || !executor_.board().isPlayable(move)) move = legal.front();
  Value response = json::objectValue();
  json::set(response, "ok", json::booleanValue(true));
  json::set(response, "move", pointValue(move));
  json::set(response, "score", json::numberValue(result.score));
  json::set(response, "depth", json::numberValue(result.reachedDepth));
  json::set(response, "nodes",
            json::numberValue(static_cast<long long>(result.nodes)));
  json::set(response, "from_vcf", json::booleanValue(result.fromVcf));
  json::set(response, "vcf_policy",
            json::stringValue(vcfPolicyName(limits.vcfPolicy)));
  json::set(response, "vcf",
            limits.vcfPolicy == VcfPolicy::Off
                ? json::nullValue()
                : vcfValue(result.vcf));
  json::set(response, "vcf_child_checks",
            json::numberValue(result.vcfChildChecks));
  json::set(response, "vcf_child_losses",
            json::numberValue(result.vcfChildLosses));
  json::set(response, "eval",
            json::stringValue(valueNet_ || nnueActive_ ? "nnue" : "handcrafted"));
  json::set(response, "eval_backend",
            json::stringValue(valueNet_ ? "value-gmvalue1-scalar"
                                        : (nnueActive_ ? "legacy-gmnnue"
                                                       : "handcrafted")));
  if (valueNet_) {
    json::set(response, "value_model_id",
              json::stringValue(valueNet_->info().model_id));
    json::set(response, "value_model_backend",
              json::stringValue("scalar-full-rebuild"));
  }
  return response;
}

Value ExecutorProtocol::dispatch(const Value& request, bool& quit) {
  if (!request.isObject()) return errorValue("request must be a JSON object");
  const Value* cmd = request.find("cmd");
  if (!cmd) return errorValue("missing cmd");
  const std::string name = cmd->asString();

  if (name == "quit") {
    quit = true;
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(true));
    return response;
  }

  if (name == "about") {
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(true));
    json::set(response, "engine", json::stringValue(kEngineName));
    json::set(response, "version", json::stringValue(kEngineVersion));
    json::set(response, "eval",
              json::stringValue(valueNet_ || nnueActive_ ? "nnue"
                                                         : "handcrafted"));
    json::set(response, "eval_backend",
              json::stringValue(valueNet_ ? "value-gmvalue1-scalar"
                                          : (nnueActive_ ? "legacy-gmnnue"
                                                         : "handcrafted")));
    json::set(response, "nnue_loaded", json::booleanValue(nnueActive_));
    json::set(response, "nnue_backend",
              json::stringValue(nnueActive_
                                    ? nnueBackendName(nnue_->backend())
                                    : "none"));
    json::set(response, "nnue_path",
              json::stringValue(nnueActive_ ? nnue_->path() : ""));
    // Ticket 07: identity of the actually loaded value model, including its
    // architecture, rule, feature spec, value contract and training identity.
    json::set(response, "value_model_loaded",
              json::booleanValue(valueNet_ != nullptr));
    json::set(response, "value_model", valueModelValue());
    // Ticket 03: which upstream VCF implementation the engine calls and how.
    json::set(response, "vcf_engine", json::stringValue(vcfUpstreamIdentity()));
    json::set(response, "vcf_policy",
              json::stringValue(vcfPolicyName(cfg_.vcfPolicy)));
    json::set(response, "vcf_nodes",
              json::numberValue(static_cast<long long>(cfg_.vcfNodeLimit)));
    // A handcrafted evaluation must never be presented as a trained NNUE.
    std::string description;
    if (valueNet_) {
      const ValueModelInfo& info = valueNet_->info();
      description =
          "new value network (GMVALUE1, " + info.arch() +
          ", scalar full-rebuild reference), model " + info.model_id +
          "; candidate for play, not promoted";
    } else if (nnueActive_) {
      description = std::string("legacy NNUE network (GMNNUE), ") +
                    nnueBackendName(nnue_->backend());
    } else {
      description =
          "handcrafted threat evaluation (no trained NNUE model loaded)";
    }
    json::set(response, "eval_description", json::stringValue(description));
    return response;
  }

  if (name == "new_game") {
    executor_.newGame(ruleFrom(request.find("rule")));
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(true));
    addState(response, executor_.state());
    return response;
  }

  if (name == "state") {
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(true));
    addState(response, executor_.state());
    return response;
  }

  if (name == "legal_moves") {
    const std::vector<int> moves = executor_.legalMoves();
    std::vector<Value> points;
    points.reserve(moves.size());
    for (int m : moves) points.push_back(pointValue(m));
    std::vector<Value> forbidden;
    const PositionState s = executor_.state();
    if (s.result == GameResult::Ongoing) {
      for (int m = 0; m < kCells; ++m) {
        if (s.cells[m] != Cell::Empty) continue;
        const RejectReason reason = executor_.moveReason(m);
        if (reason != RejectReason::ForbiddenOverline &&
            reason != RejectReason::ForbiddenDoubleFour &&
            reason != RejectReason::ForbiddenDoubleThree) {
          continue;
        }
        Value item = json::objectValue();
        json::set(item, "x", json::numberValue(Board::xOf(m)));
        json::set(item, "y", json::numberValue(Board::yOf(m)));
        json::set(item, "reason", json::stringValue(rejectReasonName(reason)));
        forbidden.push_back(std::move(item));
      }
    }
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(true));
    json::set(response, "moves", json::arrayValue(std::move(points)));
    json::set(response, "count",
              json::numberValue(static_cast<long long>(moves.size())));
    json::set(response, "forbidden",
              json::arrayValue(std::move(forbidden)));
    return response;
  }

  if (name == "move_reason") {
    int x = 0;
    int y = 0;
    std::string error;
    if (!parsePoint(request, x, y, error)) return errorValue(error);
    const RejectReason reason = executor_.moveReason(Board::indexOf(x, y));
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(true));
    json::set(response, "x", json::numberValue(x));
    json::set(response, "y", json::numberValue(y));
    json::set(response, "legal",
              json::booleanValue(reason == RejectReason::None));
    json::set(response, "reason",
              json::stringValue(rejectReasonName(reason)));
    return response;
  }

  if (name == "play") {
    int x = 0;
    int y = 0;
    std::string error;
    if (!parsePoint(request, x, y, error)) return errorValue(error);
    const MoveOutcome outcome = executor_.play(Board::indexOf(x, y));
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(outcome.ok));
    if (!outcome.ok) {
      json::set(response, "x", json::numberValue(x));
      json::set(response, "y", json::numberValue(y));
      json::set(response, "reason",
                json::stringValue(rejectReasonName(outcome.reason)));
    }
    addState(response, executor_.state());
    return response;
  }

  if (name == "undo") {
    const bool ok = executor_.undo();
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(ok));
    if (!ok) json::set(response, "reason", json::stringValue("no move to undo"));
    addState(response, executor_.state());
    return response;
  }

  if (name == "rebuild") {
    const Value* cellsValue = request.find("cells");
    if (!cellsValue || !cellsValue->isArray() ||
        cellsValue->array.size() != kCells) {
      return errorValue("cells must be an array of 225 integers (0/1/2)");
    }
    std::array<Cell, kCells> cells{};
    for (size_t i = 0; i < kCells; ++i) {
      const Value& c = cellsValue->array[i];
      if (c.type != Value::Type::Number) {
        return errorValue("cell must be a number");
      }
      const long long v = c.asInt();
      if (v < 0 || v > 2) return errorValue("cell must be 0, 1 or 2");
      cells[i] = static_cast<Cell>(v);
    }
    const Value* stmValue = request.find("side_to_move");
    const std::string stm = stmValue ? stmValue->asString("black") : "black";
    if (stm != "black" && stm != "white") {
      return errorValue("side_to_move must be black or white");
    }
    const Color color = stm == "black" ? Color::Black : Color::White;
    int blackCount = 0;
    int whiteCount = 0;
    for (Cell c : cells) {
      if (c == Cell::Black) {
        ++blackCount;
      } else if (c == Cell::White) {
        ++whiteCount;
      }
    }
    // Black moves first: equal counts imply black to move. Synthetic rule
    // positions are still rebuilt, but the caller is told the snapshot does
    // not describe a legal game history.
    const Color inferred =
        blackCount == whiteCount ? Color::Black : Color::White;
    const std::optional<std::string> error =
        executor_.rebuild(cells, color, ruleFrom(request.find("rule")));
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(!error.has_value()));
    if (error) json::set(response, "error", json::stringValue(*error));
    json::set(response, "inferred_side_to_move",
              json::stringValue(inferred == Color::Black ? "black" : "white"));
    json::set(response, "side_to_move_consistent",
              json::booleanValue(inferred == color));
    addState(response, executor_.state());
    return response;
  }

  if (name == "replay") {
    const Value* movesValue = request.find("moves");
    if (!movesValue || !movesValue->isArray()) {
      return errorValue("moves must be an array of [x,y]");
    }
    std::vector<int> moves;
    moves.reserve(movesValue->array.size());
    for (const Value& p : movesValue->array) {
      if (!p.isArray() || p.array.size() != 2 ||
          p.array[0].type != Value::Type::Number ||
          p.array[1].type != Value::Type::Number) {
        return errorValue("each move must be [x,y]");
      }
      moves.push_back(Board::indexOf(static_cast<int>(p.array[0].asInt()),
                                     static_cast<int>(p.array[1].asInt())));
    }
    const ReplayResult result =
        executor_.replay(moves, ruleFrom(request.find("rule")));
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(result.ok));
    if (!result.ok) {
      json::set(response, "error", json::stringValue(result.error));
      json::set(response, "failed_at", json::numberValue(result.failedAt));
      json::set(response, "reason",
                json::stringValue(rejectReasonName(result.reason)));
    } else {
      json::set(response, "replayed",
                json::numberValue(static_cast<long long>(moves.size())));
      addState(response, result.finalState);
    }
    return response;
  }

  if (name == "candidates") {    const Value* radiusValue = request.find("radius");
    const int radius =
        radiusValue ? static_cast<int>(radiusValue->asInt(1)) : 1;
    const std::vector<int> candidates = executor_.candidates(radius);
    std::vector<Value> points;
    points.reserve(candidates.size());
    for (int m : candidates) points.push_back(pointValue(m));
    Value response = json::objectValue();
    json::set(response, "ok", json::booleanValue(true));
    json::set(response, "moves", json::arrayValue(std::move(points)));
    json::set(response, "count",
              json::numberValue(static_cast<long long>(candidates.size())));
    json::set(response, "note",
              json::stringValue(
                  "search candidate pruning, NOT the complete legal move set"));
    return response;
  }

  if (name == "think") return think(request);

  if (name == "vcf") return vcfCommand(request);

  return errorValue("unknown cmd: " + name);
}

}  // namespace gomoku

