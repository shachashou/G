// Ticket 07: 64-byte binpack record / header / shard round-trip tests.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "binpack.h"
#include "board.h"

static int g_failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::cerr << "FAIL line " << __LINE__ << ": " << #cond << "\n"; \
      ++g_failures;                                                    \
    }                                                                  \
  } while (0)

using namespace gomoku;
using binpack::Record;

static_assert(binpack::kHeaderSize == 64);
static_assert(binpack::kRecordSize == 64);
static_assert(binpack::kPlaneBytes == 29);

static Board makeBoard(int stones) {
  Board b(RuleMode::Renju);
  for (int i = 0; i < stones; ++i) {
    const int x = (i * 5 + 1) % kBoardSize;
    b.setCell(Board::indexOf(x, i), i % 2 == 0 ? Cell::Black : Cell::White);
  }
  b.finalizeSync();  // infers side to move from stone counts
  return b;
}

static std::string tempShardPath() {
  const auto stamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  return (std::filesystem::temp_directory_path() /
          ("gomoku_binpack_test_" + std::to_string(stamp) + ".binpack"))
      .string();
}

static void testHeader() {
  binpack::Header h;
  h.searchDepth = 6;
  h.shardId = 7;
  h.recordCount = 12345;
  h.gameCount = 67;
  h.baseSeed = 0x123456789ABCDEull;
  h.ruleMode = 1;
  h.useVcf = 0;
  h.ttMb = 8;
  h.vcfNodeLimit = 4321;
  h.openingTempX100 = 150;

  const binpack::RawHeader raw = binpack::encodeHeader(h);
  CHECK(raw.size() == 64);
  CHECK(std::equal(binpack::kMagic.begin(), binpack::kMagic.end(), raw.begin()));

  std::string error;
  binpack::Header back;
  CHECK(binpack::decodeHeader(raw, back, &error));
  CHECK(back.version == binpack::kVersion);
  CHECK(back.headerSize == 64);
  CHECK(back.recordSize == 64);
  CHECK(back.searchDepth == 6);
  CHECK(back.shardId == 7);
  CHECK(back.recordCount == 12345);
  CHECK(back.gameCount == 67);
  CHECK(back.baseSeed == 0x123456789ABCDEull);
  CHECK(back.ruleMode == 1);
  CHECK(back.useVcf == 0);
  CHECK(back.ttMb == 8);
  CHECK(back.vcfNodeLimit == 4321);
  CHECK(back.openingTempX100 == 150);

  binpack::RawHeader bad = raw;
  bad[0] = 'X';
  CHECK(!binpack::decodeHeader(bad, back, &error));
  CHECK(error.find("magic") != std::string::npos);
  bad = raw;
  bad[8] = 9;  // unsupported version
  CHECK(!binpack::decodeHeader(bad, back, &error));
  bad = raw;
  bad[16] = 63;  // record size
  CHECK(!binpack::decodeHeader(bad, back, &error));
}

static void testScoreAndWdl() {
  CHECK(binpack::encodeScore(600) == 600);
  CHECK(binpack::encodeScore(-1234) == -1234);
  CHECK(binpack::encodeScore(999999) == binpack::kScoreMax);
  CHECK(binpack::encodeScore(-999999) == -binpack::kScoreMax);
  CHECK(binpack::encodeScore(500000) == binpack::kScoreMax);
  CHECK(binpack::encodeScore(0) == 0);

  CHECK(binpack::wdlFor(GameResult::BlackWin, Color::Black) == binpack::kWdlWin);
  CHECK(binpack::wdlFor(GameResult::BlackWin, Color::White) == binpack::kWdlLoss);
  CHECK(binpack::wdlFor(GameResult::WhiteWin, Color::Black) == binpack::kWdlLoss);
  CHECK(binpack::wdlFor(GameResult::Draw, Color::White) == binpack::kWdlDraw);
}

static void testRecord() {
  Board b = makeBoard(3);
  const int move = Board::indexOf(7, 10);
  const binpack::RawRecord raw = binpack::encodeRecord(b, 600, move);
  CHECK(raw.size() == 64);
  CHECK(b.sideToMove() == Color::White);
  CHECK(b.moveCount() == 3);

  // black cells (1,0) and (11,2); white cell (6,1).
  CHECK((raw[0 + (1 >> 3)] & (1u << (1 & 7))) != 0);
  const int black2 = Board::indexOf(11, 2);
  CHECK((raw[0 + (black2 >> 3)] & (1u << (black2 & 7))) != 0);
  const int white1 = Board::indexOf(6, 1);
  CHECK((raw[29 + (white1 >> 3)] & (1u << (white1 & 7))) != 0);
  CHECK((raw[28] & 0xFEu) == 0);  // padding bits
  CHECK((raw[57] & 0xFEu) == 0);
  CHECK(raw[58] == 0x58 && raw[59] == 0x02);  // int16 LE 600
  CHECK(raw[60] == (binpack::kWdlUnknown | binpack::kStmBit));
  CHECK(raw[61] == 3);
  CHECK(raw[62] == static_cast<uint8_t>(move & 0xFF));
  CHECK(raw[63] == static_cast<uint8_t>(move >> 8));

  std::string error;
  Record r;
  CHECK(binpack::decodeRecord(raw, r, &error));
  CHECK(r.score == 600);
  CHECK(r.wdl == binpack::kWdlUnknown);
  CHECK(r.stm == Color::White);
  CHECK(!r.gameStart && !r.gameEnd);
  CHECK(r.ply == 3);
  CHECK(r.move == move);
  CHECK(binpack::bitOf(r.black, 1));
  CHECK(binpack::bitOf(r.black, black2));
  CHECK(binpack::bitOf(r.white, white1));
  CHECK(!binpack::bitOf(r.black, white1));

  Board rebuilt(RuleMode::Renju);
  CHECK(binpack::boardFromRecord(r, RuleMode::Renju, rebuilt, &error));
  CHECK(rebuilt.cell(1) == Cell::Black);
  CHECK(rebuilt.cell(white1) == Cell::White);
  CHECK(rebuilt.moveCount() == 3);
  CHECK(rebuilt.sideToMove() == Color::White);

  binpack::RawRecord bad = raw;
  bad[28] |= 0x80;  // padding bit
  CHECK(!binpack::decodeRecord(bad, r, &error));
  bad = raw;
  bad[60] |= 0x20;  // reserved flag bit
  CHECK(!binpack::decodeRecord(bad, r, &error));
  bad = raw;
  bad[58] = 0x00;
  bad[59] = 0x80;  // -32768 is reserved
  CHECK(!binpack::decodeRecord(bad, r, &error));
  bad = raw;
  bad[61] = 226;  // ply above 225
  CHECK(!binpack::decodeRecord(bad, r, &error));
}

static void testLastPositionBeforeFullBoard() {
  // Colour pattern with no five-in-a-row in any of the four directions.
  auto cellAt = [](int x, int y) -> Cell {
    return ((x + 2 * y) % 5) < 2 ? Cell::Black : Cell::White;
  };
  Board almost(RuleMode::Freestyle);
  for (int i = 0; i < kCells - 1; ++i) {
    almost.setCell(i, cellAt(Board::xOf(i), Board::yOf(i)));
  }
  almost.finalizeSync();
  CHECK(almost.moveCount() == kCells - 1);
  CHECK(almost.result() == GameResult::Ongoing);

  const int last = kCells - 1;
  const binpack::RawRecord raw = binpack::encodeRecord(almost, -32767, last);
  CHECK(raw.size() == 64);
  CHECK(raw[61] == 224);
  std::string error;
  Record r;
  CHECK(binpack::decodeRecord(raw, r, &error));
  CHECK(r.ply == 224);
  CHECK(r.move == last);
  CHECK(r.score == -32767);
  Board rebuilt(RuleMode::Freestyle);
  CHECK(binpack::boardFromRecord(r, RuleMode::Freestyle, rebuilt, &error));
  CHECK(rebuilt.moveCount() == 224);

  Board full(RuleMode::Freestyle);
  for (int i = 0; i < kCells; ++i) {
    full.setCell(i, cellAt(Board::xOf(i), Board::yOf(i)));
  }
  full.finalizeSync();
  CHECK(full.result() == GameResult::Draw);
  CHECK(binpack::wdlFor(full.result(), full.sideToMove()) == binpack::kWdlDraw);
}

static void testShardBackfill() {
  const std::string path = tempShardPath();
  binpack::Header h;
  h.searchDepth = 2;
  h.shardId = 3;
  h.baseSeed = 42;
  std::string error;

  const Board b0 = makeBoard(3);
  const Board b1 = makeBoard(4);
  const Board b2 = makeBoard(5);
  const Board b3 = makeBoard(6);
  const int move = Board::indexOf(7, 10);

  binpack::ShardWriter writer;
  CHECK(writer.open(path, h, &error));
  writer.beginGame();
  writer.appendRecord(b0, 600, move);
  writer.appendRecord(b1, -1234, move);
  writer.appendRecord(b2, 999999, move);  // saturated to +32767
  writer.endGame(GameResult::BlackWin);
  writer.beginGame();
  writer.appendRecord(b3, 0, move);
  writer.endGame(GameResult::Draw);
  CHECK(writer.ok());
  CHECK(writer.finish(&error));
  CHECK(writer.recordCount() == 4);
  CHECK(writer.gameCount() == 2);

  std::ifstream sizeCheck(path, std::ios::binary | std::ios::ate);
  CHECK(static_cast<std::streamoff>(sizeCheck.tellg()) ==
        std::streamoff(64 + 4 * 64));
  sizeCheck.close();

  binpack::ShardReader reader;
  CHECK(reader.open(path, &error));
  CHECK(reader.availableRecords() == 4);
  CHECK(reader.header().recordCount == 4);
  CHECK(reader.header().gameCount == 2);
  CHECK(reader.header().shardId == 3);
  CHECK(reader.header().baseSeed == 42);

  Record r0, r1, r2, r3;
  CHECK(reader.read(0, r0, &error));
  CHECK(reader.read(1, r1, &error));
  CHECK(reader.read(2, r2, &error));
  CHECK(reader.read(3, r3, &error));
  CHECK(!reader.read(4, r0, &error));

  CHECK(r0.gameStart && !r0.gameEnd);
  CHECK(!r1.gameStart && !r1.gameEnd);
  CHECK(!r2.gameStart && r2.gameEnd);
  CHECK(r3.gameStart && r3.gameEnd);
  CHECK(r3.wdl == binpack::kWdlDraw);
  CHECK(r0.wdl == binpack::wdlFor(GameResult::BlackWin, b0.sideToMove()));
  CHECK(r1.wdl == binpack::wdlFor(GameResult::BlackWin, b1.sideToMove()));
  CHECK(r2.wdl == binpack::wdlFor(GameResult::BlackWin, b2.sideToMove()));
  CHECK(r0.score == 600);
  CHECK(r1.score == -1234);
  CHECK(r2.score == binpack::kScoreMax);
  CHECK(r0.ply == 3 && r1.ply == 4 && r2.ply == 5 && r3.ply == 6);

  Board rebuilt(RuleMode::Renju);
  CHECK(binpack::boardFromRecord(r0, RuleMode::Renju, rebuilt, &error));

  // A corrupted magic must be rejected by the reader.
  const std::string badPath = path + ".bad";
  {
    std::ifstream src(path, std::ios::binary);
    std::ofstream dst(badPath, std::ios::binary);
    dst << src.rdbuf();
  }
  {
    std::fstream f(badPath, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(0);
    f.put('X');
  }
  binpack::ShardReader badReader;
  CHECK(!badReader.open(badPath, &error));
  CHECK(error.find("magic") != std::string::npos);

  std::error_code ec;
  std::filesystem::remove(path, ec);
  std::filesystem::remove(badPath, ec);
}

int main() {
  testHeader();
  testScoreAndWdl();
  testRecord();
  testLastPositionBeforeFullBoard();
  testShardBackfill();
  if (g_failures) {
    std::cerr << g_failures << " binpack test(s) failed\n";
    return 1;
  }
  std::cout << "binpack_tests: all passed\n";
  return 0;
}
