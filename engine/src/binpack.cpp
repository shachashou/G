#include "binpack.h"

#include <algorithm>
#include <filesystem>
#include <system_error>

namespace gomoku {
namespace binpack {

namespace {

void putU16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>(v >> 8);
}

void putU32(uint8_t* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

void putU64(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
}

uint16_t getU16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint32_t getU32(const uint8_t* p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(p[i]) << (8 * i);
  return v;
}

uint64_t getU64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

bool fail(std::string* error, const std::string& message) {
  if (error) *error = message;
  return false;
}

}  // namespace

RawHeader encodeHeader(const Header& h) {
  RawHeader raw{};
  std::copy(kMagic.begin(), kMagic.end(), raw.begin());
  putU32(&raw[8], h.version);
  putU32(&raw[12], h.headerSize);
  putU32(&raw[16], h.recordSize);
  putU32(&raw[20], h.boardSize);
  putU32(&raw[24], h.searchDepth);
  putU32(&raw[28], h.shardId);
  putU64(&raw[32], h.recordCount);
  putU64(&raw[40], h.gameCount);
  putU64(&raw[48], h.baseSeed);
  raw[56] = h.ruleMode;
  raw[57] = h.useVcf;
  putU16(&raw[58], h.ttMb);
  putU16(&raw[60], h.vcfNodeLimit);
  putU16(&raw[62], h.openingTempX100);
  return raw;
}

bool decodeHeader(const RawHeader& raw, Header& out, std::string* error) {
  if (!std::equal(kMagic.begin(), kMagic.end(), raw.begin())) {
    return fail(error, "bad binpack magic");
  }
  Header h;
  h.version = getU32(&raw[8]);
  h.headerSize = getU32(&raw[12]);
  h.recordSize = getU32(&raw[16]);
  h.boardSize = getU32(&raw[20]);
  h.searchDepth = getU32(&raw[24]);
  h.shardId = getU32(&raw[28]);
  h.recordCount = getU64(&raw[32]);
  h.gameCount = getU64(&raw[40]);
  h.baseSeed = getU64(&raw[48]);
  h.ruleMode = raw[56];
  h.useVcf = raw[57];
  h.ttMb = getU16(&raw[58]);
  h.vcfNodeLimit = getU16(&raw[60]);
  h.openingTempX100 = getU16(&raw[62]);

  if (h.version != kVersion) return fail(error, "unsupported binpack version");
  if (h.headerSize != kHeaderSize) return fail(error, "bad header size");
  if (h.recordSize != kRecordSize) return fail(error, "bad record size");
  if (h.boardSize != kBoardSize) return fail(error, "unsupported board size");
  if (h.searchDepth == 0 || h.searchDepth > 32) {
    return fail(error, "implausible search depth in header");
  }
  if (h.ruleMode != 1 && h.ruleMode != 2) {
    return fail(error, "bad rule mode in header");
  }
  if (h.useVcf > 1 || h.ttMb == 0 || h.ttMb > 4096) {
    return fail(error, "bad search config in header");
  }
  out = h;
  return true;
}

int16_t encodeScore(int engineScore) {
  return static_cast<int16_t>(std::clamp(engineScore, -kScoreMax, kScoreMax));
}

uint8_t wdlFor(GameResult result, Color sideToMove) {
  if (result == GameResult::Draw) return kWdlDraw;
  const Color winner = result == GameResult::BlackWin ? Color::Black : Color::White;
  return winner == sideToMove ? kWdlWin : kWdlLoss;
}

bool bitOf(const std::array<uint8_t, kPlaneBytes>& plane, int cell) {
  return (plane[cell >> 3] >> (cell & 7)) & 1u;
}

RawRecord encodeRecord(const Board& board, int engineScore, int playedMove) {
  RawRecord raw{};
  const auto& cells = board.rawCells();
  for (int i = 0; i < kCells; ++i) {
    if (cells[i] == Cell::Black) {
      raw[kBlackPlaneOffset + (i >> 3)] |= static_cast<uint8_t>(1u << (i & 7));
    } else if (cells[i] == Cell::White) {
      raw[kWhitePlaneOffset + (i >> 3)] |= static_cast<uint8_t>(1u << (i & 7));
    }
  }
  putU16(&raw[kScoreOffset], static_cast<uint16_t>(encodeScore(engineScore)));
  uint8_t flags = kWdlUnknown;
  if (board.sideToMove() == Color::White) flags |= kStmBit;
  raw[kFlagsOffset] = flags;
  raw[kPlyOffset] = static_cast<uint8_t>(board.moveCount());
  putU16(&raw[kMoveOffset],
         (playedMove >= 0 && playedMove < kCells)
             ? static_cast<uint16_t>(playedMove)
             : kNoMove);
  return raw;
}

bool decodeRecord(const RawRecord& raw, Record& out, std::string* error) {
  // Bits 225..231 of each plane are padding and must stay zero.
  if ((raw[kBlackPlaneOffset + kPlaneBytes - 1] & 0xFEu) != 0 ||
      (raw[kWhitePlaneOffset + kPlaneBytes - 1] & 0xFEu) != 0) {
    return fail(error, "non-zero padding bits in occupancy planes");
  }
  const uint8_t flags = raw[kFlagsOffset];
  if ((flags & kReservedMask) != 0) {
    return fail(error, "non-zero reserved flag bits");
  }
  const uint16_t rawScore = getU16(&raw[kScoreOffset]);
  if (rawScore == 0x8000u) return fail(error, "reserved score -32768");
  const uint8_t ply = raw[kPlyOffset];
  if (ply > kCells) return fail(error, "ply out of range");
  const uint16_t move = getU16(&raw[kMoveOffset]);
  if (move != kNoMove && move >= kCells) return fail(error, "move out of range");

  Record r;
  std::copy_n(raw.begin() + kBlackPlaneOffset, kPlaneBytes, r.black.begin());
  std::copy_n(raw.begin() + kWhitePlaneOffset, kPlaneBytes, r.white.begin());
  r.score = static_cast<int16_t>(rawScore);
  r.wdl = flags & kWdlMask;
  r.stm = (flags & kStmBit) ? Color::White : Color::Black;
  r.gameStart = (flags & kGameStartBit) != 0;
  r.gameEnd = (flags & kGameEndBit) != 0;
  r.ply = ply;
  r.move = move;
  out = r;
  return true;
}

bool boardFromRecord(const Record& record, RuleMode mode, Board& out,
                     std::string* error) {
  out.reset(mode);
  for (int i = 0; i < kCells; ++i) {
    const bool black = bitOf(record.black, i);
    const bool white = bitOf(record.white, i);
    if (black && white) {
      return fail(error, "black and white planes overlap");
    }
    if (black) {
      out.setCell(i, Cell::Black);
    } else if (white) {
      out.setCell(i, Cell::White);
    }
  }
  out.finalizeSync();
  if (out.sideToMove() != record.stm) {
    return fail(error, "side to move disagrees with stone counts");
  }
  if (out.moveCount() != record.ply) {
    return fail(error, "ply disagrees with occupancy planes");
  }
  if (record.move != kNoMove && !out.isPlayable(record.move)) {
    return fail(error, "recorded move is not playable");
  }
  return true;
}

void ShardWriter::writeAt(std::streamoff offset, const uint8_t* data,
                          size_t size) {
  if (failed_) return;
  file_.seekp(offset, std::ios::beg);
  file_.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(size));
  if (!file_.good()) failed_ = true;
}

bool ShardWriter::open(const std::string& path, const Header& header,
                       std::string* error) {
  std::error_code ec;
  const std::filesystem::path p(path);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path(), ec);
  if (ec) return fail(error, "cannot create output directory: " + ec.message());

  file_.open(p, std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file_.is_open()) return fail(error, "cannot open shard for writing: " + path);

  header_ = header;
  header_.recordCount = 0;
  header_.gameCount = 0;
  const RawHeader raw = encodeHeader(header_);
  writeAt(0, raw.data(), raw.size());
  if (failed_) return fail(error, "cannot write shard header");
  file_.flush();
  return true;
}

void ShardWriter::beginGame() { pending_.clear(); }

void ShardWriter::appendRecord(const Board& board, int engineScore,
                               int playedMove) {
  if (failed_) return;
  const RawRecord raw = encodeRecord(board, engineScore, playedMove);
  const std::streamoff offset = endPos_;
  writeAt(offset, raw.data(), raw.size());
  endPos_ += static_cast<std::streamoff>(raw.size());
  ++header_.recordCount;
  pending_.push_back({offset, raw[kFlagsOffset]});
}

void ShardWriter::endGame(GameResult result) {
  if (failed_) return;
  for (size_t i = 0; i < pending_.size(); ++i) {
    const Pending& p = pending_[i];
    const Color stm = (p.flags & kStmBit) ? Color::White : Color::Black;
    uint8_t flags = static_cast<uint8_t>((p.flags & ~kWdlMask) | wdlFor(result, stm));
    if (i == 0) flags |= kGameStartBit;
    if (i + 1 == pending_.size()) flags |= kGameEndBit;
    writeAt(p.offset + static_cast<std::streamoff>(kFlagsOffset), &flags, 1);
  }
  if (!pending_.empty()) ++header_.gameCount;
  pending_.clear();
}

bool ShardWriter::finish(std::string* error) {
  if (failed_) return fail(error, "shard write failed");
  const RawHeader raw = encodeHeader(header_);
  writeAt(0, raw.data(), raw.size());
  file_.flush();
  if (failed_ || !file_.good()) return fail(error, "cannot finalise shard header");
  file_.close();
  return true;
}

bool ShardReader::open(const std::string& path, std::string* error) {
  file_.open(path, std::ios::binary);
  if (!file_.is_open()) return fail(error, "cannot open shard for reading: " + path);
  RawHeader raw{};
  file_.read(reinterpret_cast<char*>(raw.data()),
             static_cast<std::streamsize>(raw.size()));
  if (file_.gcount() != static_cast<std::streamsize>(raw.size())) {
    return fail(error, "truncated binpack header");
  }
  if (!decodeHeader(raw, header_, error)) return false;

  file_.seekg(0, std::ios::end);
  const std::streamoff size = file_.tellg();
  if (size < static_cast<std::streamoff>(kHeaderSize) ||
      (size - static_cast<std::streamoff>(kHeaderSize)) %
              static_cast<std::streamoff>(kRecordSize) !=
          0) {
    return fail(error, "file size is not header + whole 64-byte records");
  }
  available_ = static_cast<uint64_t>(
      (size - static_cast<std::streamoff>(kHeaderSize)) /
      static_cast<std::streamoff>(kRecordSize));
  return true;
}

bool ShardReader::read(uint64_t index, Record& out, std::string* error) {
  if (index >= available_) return fail(error, "record index out of range");
  const std::streamoff offset =
      static_cast<std::streamoff>(kHeaderSize) +
      static_cast<std::streamoff>(index) *
          static_cast<std::streamoff>(kRecordSize);
  file_.seekg(offset, std::ios::beg);
  RawRecord raw{};
  file_.read(reinterpret_cast<char*>(raw.data()),
             static_cast<std::streamsize>(raw.size()));
  if (file_.gcount() != static_cast<std::streamsize>(raw.size())) {
    return fail(error, "truncated binpack record");
  }
  return decodeRecord(raw, out, error);
}

}  // namespace binpack
}  // namespace gomoku
