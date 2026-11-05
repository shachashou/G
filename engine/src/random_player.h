#pragma once

#include <random>
#include <vector>
#include <cstdint>

#include "board.h"

namespace gomoku {

// Uniformly random legal mover (ticket 02 tracer; replaced by search in 04).
class RandomPlayer {
 public:
  explicit RandomPlayer(uint32_t seed = std::random_device{}());

  // Returns a playable move, or -1 if the board has none.
  int pickMove(const Board& board);

 private:
  std::mt19937 rng_;
  std::vector<int> playable_;
};

}  // namespace gomoku
