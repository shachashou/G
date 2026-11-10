// C ABI of the vendored upstream VCF solver bridge (engine/vcf-bridge).
//
// Must stay layout-compatible with the #[repr(C)] structs in
// engine/vcf-bridge/src/lib.rs. The C++ adapter in vcf.cpp is the only
// consumer; nothing else should include this header.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VCF_BRIDGE_MAX_PATH 512
#define VCF_BRIDGE_MESSAGE_LEN 192

enum {
  VCF_BRIDGE_STATUS_PROVEN = 0,
  VCF_BRIDGE_STATUS_DISPROVEN = 1,
  VCF_BRIDGE_STATUS_BUDGET_EXHAUSTED = 2,
  VCF_BRIDGE_STATUS_INTERRUPTED = 3,
  VCF_BRIDGE_STATUS_INVALID_INPUT = 4,
  VCF_BRIDGE_STATUS_UNSUPPORTED = 5
};

enum {
  VCF_BRIDGE_END_NONE = 0,
  VCF_BRIDGE_END_FOURS = 1,
  VCF_BRIDGE_END_FORBIDDEN = 2,
  VCF_BRIDGE_END_UNKNOWN = 3
};

typedef struct VcfBridgeQuery {
  uint8_t cells[225];       // engine layout: index = y * 15 + x, 0/1/2
  uint8_t side_to_move;     // 1 = black, 2 = white
  uint8_t rule;             // engine RuleMode: 1 freestyle, 2 renju
  uint8_t max_attack_plies;
  uint8_t reserved;
  uint64_t max_nodes;       // 0 = unlimited (callers pass a finite budget)
  uint32_t time_limit_ms;   // 0 = no wall-clock limit
} VcfBridgeQuery;

typedef struct VcfBridgeResult {
  int32_t status;
  int32_t end_kind;
  int32_t end_a;            // engine move index or -1
  int32_t end_b;
  int32_t first_move;       // engine move index or -1
  int32_t path_len;
  int32_t path[VCF_BRIDGE_MAX_PATH];
  uint64_t nodes;
  uint32_t elapsed_ms;
  uint32_t reserved;
  char message[VCF_BRIDGE_MESSAGE_LEN];
} VcfBridgeResult;

// Answers one VCF question; returns the status code and fills *out.
int32_t vcf_bridge_solve(const VcfBridgeQuery* query, VcfBridgeResult* out);

// "quintet@<commit> (MIT)" for provenance logging.
const char* vcf_bridge_version(void);

int32_t vcf_bridge_max_path(void);

#ifdef __cplusplus
}
#endif

