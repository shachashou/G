#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "board.h"

namespace gomoku {

// Result of one VCF (victory by continuous fours) query.
//
// The search itself is the vendored upstream Renju solver quintet (MIT,
// pinned commit in vendor/quintet/UPSTREAM.md); this enum only maps the
// upstream vocabulary onto words the engine, GUI and data tools share.
// "No VCF" and "search did not finish" are deliberately different states: a
// budget-exhausted or interrupted query must never be read as "no mate".
enum class VcfStatus {
  Proven,           // a forced win by continuous fours was proven
  Disproven,        // no VCF within maxAttackPlies, budget not exhausted
  BudgetExhausted,  // node budget ran out: unknown
  Interrupted,      // wall-clock limit stopped the search: unknown
  UnsupportedRule,  // upstream models Renju only; freestyle VCF is disabled
  InvalidInput,     // malformed query or evidence rejected by the board
};

const char* vcfStatusName(VcfStatus status);

// How the proven line ends, as reported by the upstream evidence.
enum class VcfEndKind { None, Fours, Forbidden, Unknown };

const char* vcfEndKindName(VcfEndKind kind);

struct VcfEvidence {
  VcfEndKind kind = VcfEndKind::None;
  // Fours: the two unblockable winning points. Forbidden: the single block
  // that is a forbidden move for Black. -1 when not applicable.
  int firstPoint = -1;
  int secondPoint = -1;
};

struct VcfLimits {
  uint64_t nodeLimit = 50000;  // hard budget; must be positive
  int maxAttackPlies = 10;     // attacker moves in the main line, 1..255
  int timeLimitMs = 0;         // 0 = no wall-clock cap
};

// Where the search asks the upstream VCF solver. The default was chosen by
// measurement, not inherited from the old root-only solver (ticket 03).
enum class VcfPolicy {
  Off,              // no VCF calls (A/B baseline)
  Root,             // root query only: a proven win replaces the search move
  RootAndChildren,  // root query plus an opponent-VCF check on root children
};

const char* vcfPolicyName(VcfPolicy policy);
bool vcfPolicyFromName(const std::string& name, VcfPolicy& out);

struct VcfResult {
  VcfStatus status = VcfStatus::InvalidInput;
  int firstMove = -1;
  // Attacker, defender, attacker, ... as move indices (y * 15 + x). Empty
  // when nothing was proven; upstream also returns an empty line for a
  // position whose attacker already has a four.
  std::vector<int> variation;
  VcfEvidence evidence;
  uint64_t nodes = 0;
  int elapsedMs = 0;
  std::string message;
  VcfLimits limits;  // budget this query ran under, echoed for records

  bool proven() const { return status == VcfStatus::Proven; }
  bool unknown() const {
    return status == VcfStatus::BudgetExhausted ||
           status == VcfStatus::Interrupted;
  }
};

// Replays the reported variation on a copy of `board` through the
// authoritative Board legality gate. Returns an empty string when every
// point is playable in turn; otherwise a message naming the offending move.
// This is the evidence re-check required before a proven line may override
// normal evaluation.
std::string checkVcfEvidence(const Board& board, const VcfResult& result);

// Runs one VCF query on `board` for the side to move. Never mutates `board`.
VcfResult solveVcf(const Board& board, const VcfLimits& limits = {});

// Upstream identity, e.g. "quintet@8219b71... (MIT)", for about/logs.
const char* vcfUpstreamIdentity();

}  // namespace gomoku

