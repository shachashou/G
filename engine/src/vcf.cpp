#include "vcf.h"

#include <algorithm>
#include <cstring>

#include "vcf_bridge.h"

namespace gomoku {

namespace {

VcfEndKind endKindFrom(int kind) {
  switch (kind) {
    case VCF_BRIDGE_END_FOURS: return VcfEndKind::Fours;
    case VCF_BRIDGE_END_FORBIDDEN: return VcfEndKind::Forbidden;
    case VCF_BRIDGE_END_UNKNOWN: return VcfEndKind::Unknown;
    default: return VcfEndKind::None;
  }
}

VcfStatus statusFrom(int status) {
  switch (status) {
    case VCF_BRIDGE_STATUS_PROVEN: return VcfStatus::Proven;
    case VCF_BRIDGE_STATUS_DISPROVEN: return VcfStatus::Disproven;
    case VCF_BRIDGE_STATUS_BUDGET_EXHAUSTED:
      return VcfStatus::BudgetExhausted;
    case VCF_BRIDGE_STATUS_INTERRUPTED: return VcfStatus::Interrupted;
    case VCF_BRIDGE_STATUS_UNSUPPORTED: return VcfStatus::UnsupportedRule;
    default: return VcfStatus::InvalidInput;
  }
}

std::string messageOf(const VcfBridgeResult& raw) {
  char buffer[VCF_BRIDGE_MESSAGE_LEN + 1];
  std::memcpy(buffer, raw.message, VCF_BRIDGE_MESSAGE_LEN);
  buffer[VCF_BRIDGE_MESSAGE_LEN] = '\0';
  return std::string(buffer);
}

VcfResult rejected(const VcfLimits& limits, const std::string& message) {
  VcfResult result;
  result.status = VcfStatus::InvalidInput;
  result.limits = limits;
  result.message = message;
  return result;
}

}  // namespace

const char* vcfStatusName(VcfStatus status) {
  switch (status) {
    case VcfStatus::Proven: return "proven";
    case VcfStatus::Disproven: return "disproven";
    case VcfStatus::BudgetExhausted: return "budget_exhausted";
    case VcfStatus::Interrupted: return "interrupted";
    case VcfStatus::UnsupportedRule: return "unsupported_rule";
    case VcfStatus::InvalidInput: return "invalid_input";
  }
  return "invalid_input";
}

const char* vcfEndKindName(VcfEndKind kind) {
  switch (kind) {
    case VcfEndKind::Fours: return "fours";
    case VcfEndKind::Forbidden: return "forbidden";
    case VcfEndKind::Unknown: return "unknown";
    case VcfEndKind::None: return "none";
  }
  return "none";
}

const char* vcfUpstreamIdentity() { return vcf_bridge_version(); }

const char* vcfPolicyName(VcfPolicy policy) {
  switch (policy) {
    case VcfPolicy::Off: return "off";
    case VcfPolicy::Root: return "root";
    case VcfPolicy::RootAndChildren: return "root_and_children";
  }
  return "root";
}

bool vcfPolicyFromName(const std::string& name, VcfPolicy& out) {
  if (name == "off") {
    out = VcfPolicy::Off;
  } else if (name == "root") {
    out = VcfPolicy::Root;
  } else if (name == "root_and_children") {
    out = VcfPolicy::RootAndChildren;
  } else {
    return false;
  }
  return true;
}

std::string checkVcfEvidence(const Board& board, const VcfResult& result) {
  if (!result.proven()) return {};
  if (result.variation.empty()) {
    // Upstream reports "already has a four" as a proof without a line.
    return result.firstMove < 0
               ? std::string()
               : std::string("proven line is empty but names a first move");
  }
  if (result.firstMove != result.variation.front()) {
    return "first move does not match the start of the variation";
  }
  Board replay = board;
  for (size_t i = 0; i < result.variation.size(); ++i) {
    const int move = result.variation[i];
    if (!Board::inBounds(move)) {
      return "variation point " + std::to_string(i) + " is out of bounds";
    }
    if (replay.result() != GameResult::Ongoing) {
      return "variation continues after the game was decided at point " +
             std::to_string(i);
    }
    if (!replay.isPlayable(move)) {
      const bool occupied = !replay.isEmpty(move);
      return std::string(occupied ? "variation point " : "forbidden variation point ") +
             std::to_string(i) + " (" + std::to_string(Board::xOf(move)) + "," +
             std::to_string(Board::yOf(move)) + ") rejected by the board";
    }
    replay.place(move);
  }
  return {};
}

VcfResult solveVcf(const Board& board, const VcfLimits& limits) {
  if (limits.nodeLimit == 0) {
    return rejected(limits, "nodeLimit must be positive");
  }
  if (limits.maxAttackPlies < 1 || limits.maxAttackPlies > 255) {
    return rejected(limits, "maxAttackPlies must be in 1..32");
  }
  if (limits.timeLimitMs < 0) {
    return rejected(limits, "timeLimitMs must not be negative");
  }
  if (board.result() != GameResult::Ongoing) {
    return rejected(limits, "game is already decided");
  }

  VcfBridgeQuery query{};
  const std::array<Cell, kCells>& cells = board.rawCells();
  for (int m = 0; m < kCells; ++m) {
    query.cells[m] = static_cast<uint8_t>(cells[m]);
  }
  query.side_to_move = static_cast<uint8_t>(board.sideToMove());
  query.rule = static_cast<uint8_t>(board.ruleMode());
  query.max_attack_plies = static_cast<uint8_t>(limits.maxAttackPlies);
  query.max_nodes = limits.nodeLimit;
  query.time_limit_ms = static_cast<uint32_t>(limits.timeLimitMs);

  VcfBridgeResult raw{};
  vcf_bridge_solve(&query, &raw);

  VcfResult result;
  result.status = statusFrom(raw.status);
  result.limits = limits;
  result.nodes = raw.nodes;
  result.elapsedMs = static_cast<int>(raw.elapsed_ms);
  result.message = messageOf(raw);
  result.evidence.kind = endKindFrom(raw.end_kind);
  result.evidence.firstPoint = raw.end_a;
  result.evidence.secondPoint = raw.end_b;
  result.firstMove = raw.first_move;
  const int pathLength = std::clamp(raw.path_len, 0, VCF_BRIDGE_MAX_PATH);
  result.variation.assign(raw.path, raw.path + pathLength);

  if (result.status == VcfStatus::Proven) {
    const std::string problem = checkVcfEvidence(board, result);
    if (!problem.empty()) {
      // A proven line the authoritative board rejects must not be used.
      result.status = VcfStatus::InvalidInput;
      result.message = "upstream evidence rejected: " + problem;
      result.variation.clear();
      result.firstMove = -1;
      result.evidence = VcfEvidence{};
    }
  }
  return result;
}

}  // namespace gomoku

