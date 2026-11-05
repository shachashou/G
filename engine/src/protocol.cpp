#include "protocol.h"

#include <optional>
#include <sstream>
#include <string>

#include "version.h"

namespace gomoku {

namespace {

std::string moveToXY(int move) {
  return std::to_string(Board::xOf(move)) + "," + std::to_string(Board::yOf(move));
}

// Parses "x,y" (0-based column,row) into a cell index; validates bounds.
std::optional<int> parseXY(const std::string& arg) {
  const size_t comma = arg.find(',');
  if (comma == std::string::npos) return std::nullopt;
  try {
    const int x = std::stoi(arg.substr(0, comma));
    const int y = std::stoi(arg.substr(comma + 1));
    if (x < 0 || x >= kBoardSize || y < 0 || y >= kBoardSize) {
      return std::nullopt;
    }
    return Board::indexOf(x, y);
  } catch (...) {
    return std::nullopt;
  }
}

struct BoardLine {
  int move;
  Cell cell;
};

// Parses a BOARD data line "x,y,c" where c is 1=black, 2=white.
std::optional<BoardLine> parseBoardLine(const std::string& s) {
  const size_t c1 = s.find(',');
  if (c1 == std::string::npos) return std::nullopt;
  const size_t c2 = s.find(',', c1 + 1);
  if (c2 == std::string::npos) return std::nullopt;
  try {
    const int x = std::stoi(s.substr(0, c1));
    const int y = std::stoi(s.substr(c1 + 1, c2 - c1 - 1));
    const int color = std::stoi(s.substr(c2 + 1));
    if (x < 0 || x >= kBoardSize || y < 0 || y >= kBoardSize) {
      return std::nullopt;
    }
    if (color != 1 && color != 2) return std::nullopt;
    return BoardLine{Board::indexOf(x, y),
                     color == 1 ? Cell::Black : Cell::White};
  } catch (...) {
    return std::nullopt;
  }
}

void stripCR(std::string& line) {
  if (!line.empty() && line.back() == '\r') line.pop_back();
}

}  // namespace

int Protocol::run(std::istream& in, std::ostream& out) {
  std::string line;
  while (std::getline(in, line)) {
    stripCR(line);
    if (line.empty()) continue;
    if (line == "END") return 0;
    handle(line, in, out);
    out.flush();
  }
  return 0;
}

void Protocol::handle(const std::string& line, std::istream& in, std::ostream& out) {
  const size_t sp = line.find(' ');
  const std::string cmd = sp == std::string::npos ? line : line.substr(0, sp);
  const std::string arg = sp == std::string::npos ? std::string() : line.substr(sp + 1);

  if (cmd == "START") {
    int size = 0;
    bool parsed = false;
    try {
      size = std::stoi(arg);
      parsed = true;
    } catch (...) {
    }
    if (parsed && size == kBoardSize) {
      board_.reset(board_.ruleMode());
      out << "OK\n";
    } else {
      out << "ERROR unsupported board size (only 15x15)\n";
    }
  } else if (cmd == "RESTART") {
    board_.reset(board_.ruleMode());
    out << "OK\n";
  } else if (cmd == "BEGIN") {
    board_.reset(board_.ruleMode());
    playEngineMove(out);
  } else if (cmd == "TURN") {
    const std::optional<int> move = parseXY(arg);
    if (!move) {
      out << "ERROR malformed move\n";
    } else if (!board_.isPlayable(*move)) {
      // Occupied point or (Renju, black) a forbidden move.
      out << "ERROR illegal move\n";
    } else if (board_.result() != GameResult::Ongoing) {
      // Game already decided; nothing to answer.
    } else {
      board_.place(*move);
      if (board_.result() == GameResult::Ongoing) playEngineMove(out);
    }
  } else if (cmd == "BOARD") {
    if (readBoard(in, out)) playEngineMove(out);
  } else if (cmd == "INFO") {
    handleInfo(arg);
  } else if (cmd == "ABOUT") {
    out << "name=\"" << kEngineName << "\", version=\"" << kEngineVersion
        << "\", author=\"G\", country=\"CN\"\n";
  } else {
    out << "UNKNOWN\n";
  }
}

bool Protocol::readBoard(std::istream& in, std::ostream& out) {
  board_.reset(board_.ruleMode());
  bool ok = true;
  std::string line;
  while (std::getline(in, line)) {
    stripCR(line);
    if (line == "DONE") {
      if (!ok) {
        out << "ERROR invalid BOARD data\n";
        return false;
      }
      board_.finalizeSync();
      return true;
    }
    const std::optional<BoardLine> parsed = parseBoardLine(line);
    if (parsed) {
      board_.setCell(parsed->move, parsed->cell);
    } else {
      ok = false;
    }
  }
  // EOF before DONE.
  out << "ERROR BOARD terminated without DONE\n";
  return false;
}

void Protocol::handleInfo(const std::string& arg) {
  const size_t sp = arg.find(' ');
  if (sp == std::string::npos) return;
  const std::string key = arg.substr(0, sp);
  const std::string value = arg.substr(sp + 1);
  if (key == "rule") {
    // Engine contract: 1 = freestyle, 2 = Renju. Foul exclusion and
    // black's exact-five win are enforced in Board::isPlayable / place.
    try {
      const int v = std::stoi(value);
      if (v == static_cast<int>(RuleMode::Freestyle)) {
        board_.setRuleMode(RuleMode::Freestyle);
      } else if (v == static_cast<int>(RuleMode::Renju)) {
        board_.setRuleMode(RuleMode::Renju);
      }
    } catch (...) {
    }
  }
  // Other INFO keys (timeout_turn, timeout_match, ...) are consumed from
  // ticket 04 (time management); silently accepted for now.
}

void Protocol::playEngineMove(std::ostream& out) {
  const int move = player_.pickMove(board_);
  if (move < 0) {
    out << "ERROR no legal move\n";
    return;
  }
  board_.place(move);
  out << moveToXY(move) << "\n";
}

}  // namespace gomoku
