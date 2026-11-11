#include <cstdlib>
#include <iostream>
#include <string_view>

#include "executor_protocol.h"
#include "protocol.h"
#include "version.h"

namespace {

void printUsage() {
  std::cout << "Usage: gomoku_nnue [options]\n"
            << "  --version            print engine banner and exit\n"
            << "  --random             random legal mover (strength baseline)\n"
            << "  --depth N            fixed search depth N (no clock)\n"
            << "  --movetime N         per-move time budget in ms (default 1000)\n"
            << "  --no-vcf             disable the VCF solver (off; A/B baseline)\n"
            << "  --vcf-policy NAME    VCF call sites: off | root | root_and_children\n"
            << "                       (default root_and_children)\n"
            << "  --vcf-nodes N        root VCF node budget (default 50000)\n"
            << "  --nnue PATH          legacy NNUE evaluation from a GMNNUE01 file\n"
            << "  --nnue-backend NAME  legacy NNUE backend: avx2 (default) | scalar\n"
            << "  --value-model PATH   deployed value network (GMVALUE1); a bad\n"
            << "                       file aborts instead of falling back\n"
            << "  --value-backend NAME value inference: avx2 (default, incremental\n"
            << "                       first layer) | scalar | reference (full\n"
            << "                       rebuild + scalar, numeric reference)\n"
            << "  --executor           JSON-lines authoritative executor (ticket 02)\n"
            << "  --help               show this help\n"
            << "Default: Alpha-Beta search over the Gomocup stdin/stdout protocol.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  gomoku::EngineConfig cfg;
  bool executorMode = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a(argv[i]);
    if (a == "--version") {
      std::cout << gomoku::kEngineName << ' ' << gomoku::kEngineVersion
                << " (C++20, AVX2) - Renju NNUE engine\n";
      return 0;
    }
    if (a == "--help") {
      printUsage();
      return 0;
    }
    if (a == "--random") {
      cfg.random = true;
    } else if (a == "--executor") {
      executorMode = true;
    } else if (a == "--no-vcf") {
      cfg.vcfPolicy = gomoku::VcfPolicy::Off;
    } else if (a == "--vcf-policy" && i + 1 < argc) {
      if (!gomoku::vcfPolicyFromName(argv[++i], cfg.vcfPolicy)) {
        std::cerr << "--vcf-policy must be off, root or root_and_children\n";
        return 2;
      }
    } else if (a == "--vcf-nodes" && i + 1 < argc) {
      cfg.vcfNodeLimit = static_cast<uint64_t>(std::atoll(argv[++i]));
    } else if (a == "--depth" && i + 1 < argc) {
      cfg.fixedDepth = std::atoi(argv[++i]);
    } else if (a == "--movetime" && i + 1 < argc) {
      cfg.moveTimeMs = std::atoi(argv[++i]);
    } else if (a == "--nnue" && i + 1 < argc) {
      cfg.nnueWeights = argv[++i];
    } else if (a == "--nnue-backend" && i + 1 < argc) {
      cfg.nnueBackend = argv[++i];
    } else if (a == "--value-model" && i + 1 < argc) {
      cfg.valueModel = argv[++i];
    } else if (a == "--value-backend" && i + 1 < argc) {
      cfg.valueBackend = argv[++i];
    }
  }
  if (!cfg.valueModel.empty() && !cfg.nnueWeights.empty()) {
    std::cerr << "--value-model and --nnue are mutually exclusive; choose the "
                 "deployed GMVALUE1 model or the legacy weights\n";
    return 2;
  }
  {
    gomoku::ValueBackend backend = gomoku::ValueBackend::Avx2;
    gomoku::ValueAccumulatorMode accumulator =
        gomoku::ValueAccumulatorMode::Incremental;
    if (!gomoku::valueEvalModeFromName(cfg.valueBackend, backend,
                                       accumulator)) {
      std::cerr << "--value-backend must be avx2, scalar or reference\n";
      return 2;
    }
  }
  try {
    if (executorMode) {
      // One JSON request per line, one JSON response per line. stdout is the
      // protocol stream; diagnostics go to stderr.
      gomoku::ExecutorProtocol protocol(cfg, std::cout);
      return protocol.run(std::cin);
    }
    // Protocol mode: speak Gomocup on stdin/stdout. No banner — extra output
    // would corrupt the protocol stream.
    gomoku::Protocol protocol(cfg);
    return protocol.run(std::cin, std::cout);
  } catch (const gomoku::ValueNetError& error) {
    // Ticket 07: an explicitly requested value model must either load exactly
    // or fail the process. There is no silent handcrafted continuation.
    std::cerr << "value model failed to load: " << error.what() << "\n";
    return 2;
  }
}

