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

Value ExecutorProtocol::think(const Value& request) const {
  const std::vector<int> legal = executor_.legalMoves();
  if (legal.empty()) {
    return errorValue("no legal move (game over or full board)");
  }
  SearchLimits limits;
  limits.useVcf = cfg_.useVcf;
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
  json::set(response, "eval",
            json::stringValue(nnueActive_ ? "nnue" : "handcrafted"));
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
              json::stringValue(nnueActive_ ? "nnue" : "handcrafted"));
    json::set(response, "nnue_loaded", json::booleanValue(nnueActive_));
    json::set(response, "nnue_backend",
              json::stringValue(nnueActive_
                                    ? nnueBackendName(nnue_->backend())
                                    : "none"));
    json::set(response, "nnue_path",
              json::stringValue(nnueActive_ ? nnue_->path() : ""));
    // A handcrafted evaluation must never be presented as a trained NNUE.
    json::set(response, "eval_description",
              json::stringValue(
                  nnueActive_
                      ? std::string("NNUE network (GMNNUE01), ") +
                            nnueBackendName(nnue_->backend())
                      : "handcrafted threat evaluation (no trained NNUE model loaded)"));
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

  if (name == "candidates") {
    const Value* radiusValue = request.find("radius");
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

  return errorValue("unknown cmd: " + name);
}

}  // namespace gomoku
