set(ENGINE_CORE_SRC
  src/board.cpp
  src/renju.cpp
  src/eval.cpp
  src/nnue.cpp
  src/movegen.cpp
  src/search.cpp
  src/vcf.cpp
  src/random_player.cpp
  src/binpack.cpp
  src/executor.cpp)

# Ticket 03: the VCF solver is the vendored upstream quintet crate (MIT,
# pinned commit in vendor/quintet/UPSTREAM.md), reached through the small C ABI
# in engine/vcf-bridge. Cargo builds it as a static library; every target that
# compiles vcf.cpp must link it.
if(MSVC)
  set(VCF_BRIDGE_LIB
    "${CMAKE_CURRENT_SOURCE_DIR}/vcf-bridge/target/release/vcf_bridge.lib")
else()
  set(VCF_BRIDGE_LIB
    "${CMAKE_CURRENT_SOURCE_DIR}/vcf-bridge/target/release/libvcf_bridge.a")
endif()
find_program(GOMOKU_CARGO cargo
  HINTS "$ENV{USERPROFILE}/.cargo/bin" "$ENV{HOME}/.cargo/bin")
if(NOT GOMOKU_CARGO)
  message(FATAL_ERROR
    "cargo was not found. Ticket 03 builds the vendored quintet VCF solver "
    "with the stable Rust toolchain (https://rustup.rs). Install it, or put "
    "cargo on PATH, then re-run cmake.")
endif()
add_test(NAME vcf_bridge_tests
  COMMAND ${GOMOKU_CARGO} test --release
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/vcf-bridge/Cargo.toml")
set_tests_properties(vcf_bridge_tests PROPERTIES TIMEOUT 300)

add_custom_target(vcf_bridge ALL
  COMMAND ${GOMOKU_CARGO} build --release
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/vcf-bridge/Cargo.toml"
  BYPRODUCTS "${VCF_BRIDGE_LIB}"
  COMMENT "Building vendored quintet VCF bridge (Rust)"
  VERBATIM)

function(gomoku_link_vcf_bridge target)
  add_dependencies(${target} vcf_bridge)
  target_link_libraries(${target} PRIVATE "${VCF_BRIDGE_LIB}")
  if(MSVC)
    # Native libraries a Rust static library needs on Windows
    # (`cargo rustc -- --print native-static-libs`).
    target_link_libraries(${target} PRIVATE kernel32 ntdll userenv ws2_32 dbghelp)
  else()
    target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS} pthread m)
  endif()
endfunction()

# GomokuNNUE engine target
add_executable(gomoku_nnue
  src/main.cpp
  src/protocol.cpp
  src/executor_protocol.cpp
  src/json.cpp
  ${ENGINE_CORE_SRC})
target_compile_features(gomoku_nnue PRIVATE cxx_std_20)
target_include_directories(gomoku_nnue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(gomoku_nnue)

# Cross-language parity probe (ticket 09): Python drives it over stdin/stdout.
add_executable(nnue_probe src/nnue_probe.cpp ${ENGINE_CORE_SRC})
target_compile_features(nnue_probe PRIVATE cxx_std_20)
target_include_directories(nnue_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(nnue_probe)

# Self-play datagen worker (ticket 07). One process owns one shard file;
# tools/datagen.py launches many of them in parallel.
add_executable(datagen src/datagen.cpp ${ENGINE_CORE_SRC})
target_compile_features(datagen PRIVATE cxx_std_20)
target_include_directories(datagen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(datagen)

# Board / random-player unit tests (ticket 02)
add_executable(board_tests tests/board_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(board_tests PRIVATE cxx_std_20)
target_include_directories(board_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(board_tests)
add_test(NAME board_tests COMMAND board_tests)

# Authoritative executor unit tests (ticket 02): illegal requests leave the
# position untouched, terminal rejection, undo, forbidden-move reasons,
# replay consistency and the legal-set / candidate-pruning distinction.
add_executable(executor_tests tests/executor_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(executor_tests PRIVATE cxx_std_20)
target_include_directories(executor_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(executor_tests)
add_test(NAME executor_tests COMMAND executor_tests)

# Renju forbidden-move unit tests (ticket 03)
add_executable(renju_tests tests/renju_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(renju_tests PRIVATE cxx_std_20)
target_include_directories(renju_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(renju_tests)
add_test(NAME renju_tests COMMAND renju_tests)

# Incremental evaluation consistency tests (ticket 04)
add_executable(eval_tests tests/eval_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(eval_tests PRIVATE cxx_std_20)
target_include_directories(eval_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(eval_tests)
add_test(NAME eval_tests COMMAND eval_tests)

# Search tactical behaviour tests (ticket 04)
add_executable(search_tests tests/search_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(search_tests PRIVATE cxx_std_20)
target_include_directories(search_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(search_tests)
add_test(NAME search_tests COMMAND search_tests)

# VCF (continuous-fours) tactic tests (ticket 05)
add_executable(vcf_tests tests/vcf_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(vcf_tests PRIVATE cxx_std_20)
target_include_directories(vcf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(vcf_tests)
add_test(NAME vcf_tests COMMAND vcf_tests)

# 64-byte binpack record round-trip tests (ticket 07)
add_executable(binpack_tests tests/binpack_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(binpack_tests PRIVATE cxx_std_20)
target_include_directories(binpack_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(binpack_tests)
add_test(NAME binpack_tests COMMAND binpack_tests)

# Ticket 09: scalar NNUE weight loader, integer forward and incremental
# accumulator consistency.
add_executable(nnue_tests tests/nnue_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(nnue_tests PRIVATE cxx_std_20)
target_include_directories(nnue_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(nnue_tests)
add_test(NAME nnue_tests COMMAND nnue_tests)

# Protocol black-box test: drives the exe via stdin/stdout.
# Prefer the project venv (py -3.12 -m venv .venv) on Windows (Scripts/) or
# POSIX (bin/), never the system Python.
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
  set(GOMOKU_PYTHON "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  set(GOMOKU_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python")
else()
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_FOUND)
    set(GOMOKU_PYTHON "${Python3_EXECUTABLE}")
  endif()
endif()
if(GOMOKU_PYTHON)
  add_test(NAME protocol_blackbox
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/protocol_blackbox.py"
            "$<TARGET_FILE:gomoku_nnue>")

  # Ticket 02: JSON-lines authoritative executor boundary, driven through
  # the shared Python client (GUI / collection / referee all use it).
  add_test(NAME executor_blackbox
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/executor_blackbox.py"
            "$<TARGET_FILE:gomoku_nnue>"
            "${CMAKE_SOURCE_DIR}/tools/executor.py")
  set_tests_properties(executor_blackbox PROPERTIES TIMEOUT 180)

  # Match runner self-test (ticket 06): real engine vs random baseline plus
  # illegal-move / timeout / crash anomaly paths driven by a fake engine.
  add_test(NAME match_runner_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/match_runner_tests.py"
            "$<TARGET_FILE:gomoku_nnue>"
            "${CMAKE_SOURCE_DIR}/tools/match_runner.py")
  set_tests_properties(match_runner_tests PROPERTIES TIMEOUT 120)

  # Datagen end-to-end test (ticket 07): real C++ worker writes shards,
  # Python read end validates / merges, C++ worker replays sampled scores.
  add_test(NAME datagen_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/datagen_tests.py"
            "$<TARGET_FILE:datagen>"
            "${CMAKE_SOURCE_DIR}/trainer/binpack.py"
            "${CMAKE_SOURCE_DIR}/tools/datagen.py")
  set_tests_properties(datagen_tests PROPERTIES TIMEOUT 300)

  # Ticket 08 feature/D4 spec, weight format and game-level split tests.
  # Torch-dependent model/train/export checks run automatically once the
  # training requirements are installed (otherwise reported as skipped).
  add_test(NAME training_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/training_tests.py")
  set_tests_properties(training_tests PROPERTIES TIMEOUT 300)

  # Ticket 13: frozen, D4-leak-free validation split (freeze/check/manifest
  # and, when torch is available, train.py integration).
  add_test(NAME validation_freeze_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/validation_freeze_tests.py")
  set_tests_properties(validation_freeze_tests PROPERTIES TIMEOUT 300)

  # Ticket 09: C++ scalar inference / feature indices vs the Python spec and
  # (when torch is available) the PyTorch float forward.
  add_test(NAME nnue_parity_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/nnue_parity_tests.py"
            "$<TARGET_FILE:nnue_probe>")
  set_tests_properties(nnue_parity_tests PROPERTIES TIMEOUT 300)

  # Ticket 10: scalar/AVX2 bit-exact parity and single-evaluation timing.
  add_test(NAME nnue_backend_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/nnue_backend_tests.py"
            "$<TARGET_FILE:nnue_probe>")
  set_tests_properties(nnue_backend_tests PROPERTIES TIMEOUT 300)

  # Ticket 03: reusable VCF tactical boundary for data collection (no GUI)
  # and the offscreen GUI panel that shows the proven status and variation.
  add_test(NAME vcf_tactics_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/vcf_tactics_tests.py"
            "$<TARGET_FILE:gomoku_nnue>"
            "${CMAKE_SOURCE_DIR}/tools")
  set_tests_properties(vcf_tactics_tests PROPERTIES TIMEOUT 120)

  add_test(NAME gui_vcf_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_vcf_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_vcf_smoke PROPERTIES
    TIMEOUT 180 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  # Ticket 04: real Rapfi teacher + executor + versioned sample store, and the
  # offscreen GUI panel. Exit code 77 skips when Rapfi is not installed.
  add_test(NAME teacher_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/teacher_tests.py"
            "$<TARGET_FILE:gomoku_nnue>"
            "${CMAKE_SOURCE_DIR}/tools")
  set_tests_properties(teacher_tests PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)

  add_test(NAME gui_teacher_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_teacher_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_teacher_smoke PROPERTIES
    TIMEOUT 300 SKIP_RETURN_CODE 77 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  # Ticket 09: full self-play game with NNUE evaluation + bad-weight fallback.
  add_test(NAME nnue_blackbox
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/nnue_blackbox.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(nnue_blackbox PROPERTIES TIMEOUT 300)
endif()

