#pragma once

// Ticket 07: fixed-size self-play training records.
//
// File layout (little-endian throughout):
//   [0, 64)               file header (see binpack::Header)
//   [64, 64 + 64 * N)     N fixed-size records
//
// Record layout:
//   [0, 29)    black occupancy, 225 bits, LSB-first; bit i == cell i
//   [29, 58)   white occupancy, same encoding
//   [58, 60)   int16 search score (side-to-move perspective, saturated)
//   [60]       flags: bits 0-1 WDL, bit 2 stm, bit 3 game-start, bit 4 game-end
//   [61]       ply (stones already on the board, 0..225)
//   [62, 64)   uint16 move actually played from this position (0..224 / 0xFFFF)
//
// The authoritative format description lives in tools/README.md.

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "board.h"

namespace gomoku {
namespace binpack {

inline constexpr std::array<uint8_t, 8> kMagic{'G', 'O', 'M', 'O',
                                              'K', 'U', '0', '7'};
inline constexpr uint32_t kVersion = 1;
inline constexpr uint32_t kHeaderSize = 64;
inline constexpr uint32_t kRecordSize = 64;
inline constexpr uint32_t kPlaneBytes = 29;  // ceil(225 / 8)
inline constexpr uint32_t kBoardSize = 15;
inline constexpr uint32_t kMergedShardId = 0xFFFFFFFFu;
inline constexpr uint16_t kNoMove = 0xFFFF;

// flags byte at kFlagsOffset.
inline constexpr uint8_t kWdlMask = 0x03;
inline constexpr uint8_t kStmBit = 0x04;
inline constexpr uint8_t kGameStartBit = 0x08;
inline constexpr uint8_t kGameEndBit = 0x10;
inline constexpr uint8_t kReservedMask = 0xE0;
// WDL is always relative to the side to move of the recorded position.
inline constexpr uint8_t kWdlLoss = 0;
inline constexpr uint8_t kWdlDraw = 1;
inline constexpr uint8_t kWdlWin = 2;
inline constexpr uint8_t kWdlUnknown = 3;  // placeholder before result backfill

inline constexpr uint32_t kBlackPlaneOffset = 0;
inline constexpr uint32_t kWhitePlaneOffset = kPlaneBytes;
inline constexpr uint32_t kScoreOffset = 58;
inline constexpr uint32_t kFlagsOffset = 60;
inline constexpr uint32_t kPlyOffset = 61;
inline constexpr uint32_t kMoveOffset = 62;

// Engine scores are ints (mate = +/-999999, pattern table peaks at 500000),
// so they are saturated into int16. -32768 is never written.
inline constexpr int kScoreMax = 32767;

static_assert(kWhitePlaneOffset + kPlaneBytes == kScoreOffset);
static_assert(kMoveOffset + 2 == kRecordSize);
static_assert(kPlaneBytes * 8 >= kCells);

// 64-byte file header. All fields are explicit so a Python reader can reject
// files produced by a different engine build / format version.
struct Header {
  uint32_t version = kVersion;
  uint32_t headerSize = kHeaderSize;
  uint32_t recordSize = kRecordSize;
  uint32_t boardSize = kBoardSize;
  uint32_t searchDepth = 6;
  uint32_t shardId = 0;
  uint64_t recordCount = 0;
  uint64_t gameCount = 0;  // games with at least one recorded position
  uint64_t baseSeed = 0;
  uint8_t ruleMode = 2;       // 1 = freestyle, 2 = Renju
  uint8_t useVcf = 1;         // 0/1
  uint16_t ttMb = 16;         // transposition table size used by the worker
  uint16_t vcfNodeLimit = 5000;
  uint16_t openingTempX100 = 100;
};

struct Record {
  std::array<uint8_t, kPlaneBytes> black{};
  std::array<uint8_t, kPlaneBytes> white{};
  int16_t score = 0;
  uint8_t wdl = kWdlUnknown;
  Color stm = Color::Black;
  bool gameStart = false;
  bool gameEnd = false;
  uint8_t ply = 0;
  uint16_t move = kNoMove;
};

using RawHeader = std::array<uint8_t, kHeaderSize>;
using RawRecord = std::array<uint8_t, kRecordSize>;

RawHeader encodeHeader(const Header& header);
bool decodeHeader(const RawHeader& raw, Header& out, std::string* error);

int16_t encodeScore(int engineScore);
uint8_t wdlFor(GameResult result, Color sideToMove);

// Encodes a record with WDL = kWdlUnknown; endGame() patches it in place.
RawRecord encodeRecord(const Board& board, int engineScore, int playedMove);
bool decodeRecord(const RawRecord& raw, Record& out, std::string* error);
bool boardFromRecord(const Record& record, RuleMode mode, Board& out,
                     std::string* error);
bool bitOf(const std::array<uint8_t, kPlaneBytes>& plane, int cell);

// Streaming shard writer. Records are written immediately with an "unknown"
// result; when a game ends their flags byte is patched in place, which is why
// every record must be exactly 64 bytes. Header counts are rewritten on finish.
class ShardWriter {
 public:
  bool open(const std::string& path, const Header& header, std::string* error);
  void beginGame();
  void appendRecord(const Board& board, int engineScore, int playedMove);
  void endGame(GameResult result);
  bool finish(std::string* error);

  bool ok() const { return !failed_; }
  uint64_t recordCount() const { return header_.recordCount; }
  uint64_t gameCount() const { return header_.gameCount; }

 private:
  struct Pending {
    std::streamoff offset;
    uint8_t flags;
  };

  void writeAt(std::streamoff offset, const uint8_t* data, size_t size);

  std::fstream file_;
  Header header_{};
  std::streamoff endPos_ = static_cast<std::streamoff>(kHeaderSize);
  std::vector<Pending> pending_;
  bool failed_ = false;
};

// Read-only shard access for offline score replay.
class ShardReader {
 public:
  bool open(const std::string& path, std::string* error);
  bool read(uint64_t index, Record& out, std::string* error);

  const Header& header() const { return header_; }
  uint64_t availableRecords() const { return available_; }

 private:
  std::ifstream file_;
  Header header_{};
  uint64_t available_ = 0;
};

}  // namespace binpack
}  // namespace gomoku
