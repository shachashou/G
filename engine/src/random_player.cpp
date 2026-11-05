#include "random_player.h"

#include <vector>

namespace gomoku {

RandomPlayer::RandomPlayer(uint32_t seed) : rng_(seed) {}

int RandomPlayer::pickMove(const Board& board) {
  playable_.clear();
  for (int m = 0; m < kCells; ++m) {
    if (board.isPlayable(m)) playable_.push_back(m);
  }
  if (playable_.empty()) return -1;
  std::uniform_int_distribution<int> dist(0, static_cast<int>(playable_.size()) - 1);
  return playable_[dist(rng_)];
}

}  // namespace gomoku
