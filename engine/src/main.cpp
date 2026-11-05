#include <iostream>
#include <string_view>

#include "protocol.h"
#include "version.h"

int main(int argc, char* argv[]) {
  if (argc > 1 && std::string_view(argv[1]) == "--version") {
    std::cout << gomoku::kEngineName << ' ' << gomoku::kEngineVersion
              << " (C++20, MSVC, AVX2) - Renju NNUE engine\n";
    return 0;
  }
  // Protocol mode: speak Gomocup on stdin/stdout. No banner — extra output
  // would corrupt the protocol stream.
  gomoku::Protocol protocol;
  return protocol.run(std::cin, std::cout);
}
