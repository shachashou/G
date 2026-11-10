#pragma once

#include <iosfwd>
#include <memory>
#include <string>

#include "executor.h"
#include "json.h"
#include "nnue.h"
#include "protocol.h"  // EngineConfig
#include "search.h"
#include "value_net.h"

namespace gomoku {

// JSON-lines front end for the authoritative executor (ticket 02). One
// request object per line, one response object per line. The Gomocup
// Protocol is left untouched for engine-vs-engine play.
class ExecutorProtocol {
 public:
  ExecutorProtocol(const EngineConfig& cfg, std::ostream& out);
  // Reads requests until "quit" or EOF; returns the process exit code.
  int run(std::istream& in);

 private:
  json::Value dispatch(const json::Value& request, bool& quit);
  json::Value errorValue(const std::string& message) const;
  void addState(json::Value& target, const PositionState& state) const;
  json::Value think(const json::Value& request) const;
  // Ticket 07: identity of the actually loaded GMVALUE1 model.
  json::Value valueModelValue() const;
  // Ticket 03: one standalone VCF query on the current position.
  json::Value vcfCommand(const json::Value& request) const;
  json::Value vcfValue(const VcfResult& result) const;

  EngineConfig cfg_;
  std::ostream& out_;
  Executor executor_;
  std::unique_ptr<NnueNetwork> nnue_;
  bool nnueActive_ = false;
  // Ticket 07: loaded GMVALUE1 model; takes precedence over nnue_.
  std::unique_ptr<ValueNetwork> valueNet_;
  std::unique_ptr<Search> search_;
};

}  // namespace gomoku
