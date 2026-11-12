set(ENGINE_CORE_SRC
  src/board.cpp
  src/renju.cpp
  src/eval.cpp
  src/nnue.cpp
  src/value_net.cpp
  src/movegen.cpp
  src/search.cpp
  src/vcf.cpp
  src/random_player.cpp
  src/binpack.cpp
  src/json.cpp
  src/executor.cpp)

# Ticket 03: the VCF solver is the vendored upstream quintet crate (MIT,
# pinned commit in vendor/quintet/UPSTREAM.md), reached through the small C ABI
# in engine/vcf-bridge. Cargo builds it as a static library; every target that
# compiles vcf.cpp must link it.
# Rust static libraries are named libvcf_bridge.a on every supported platform.
set(VCF_BRIDGE_LIB
  "${CMAKE_CURRENT_SOURCE_DIR}/vcf-bridge/target/release/libvcf_bridge.a")
# Ticket 07: the build toolchain is contained in .venv on Windows.
#   * C/C++: pip-installed ziglang (clang + libc++ + mingw-w64 ABI) through the
#     wrappers in tools/build. No Visual Studio / Windows SDK files are read.
#   * Rust: rustup toolchain installed under .venv (stable-x86_64-pc-windows-gnu),
#     linking with the venv rust-lld and self-contained mingw import libraries.
# Linux / macOS keep using the system toolchain.
set(GOMOKU_VENV "${CMAKE_SOURCE_DIR}/.venv")
set(GOMOKU_VENV_CARGO "${GOMOKU_VENV}/cargo/bin/cargo.exe")
if(WIN32)
  if(NOT EXISTS "${GOMOKU_VENV_CARGO}")
    message(FATAL_ERROR
      "Windows builds use only the toolchain inside .venv. Run: "
      ".venv\\Scripts\\python.exe tools\\build\\bootstrap_toolchain.py")
  endif()
  set(GOMOKU_CARGO "${GOMOKU_VENV_CARGO}")
  set(GOMOKU_RUST_LLD
    "${GOMOKU_VENV}/rustup/toolchains/stable-x86_64-pc-windows-gnu/lib/rustlib/x86_64-pc-windows-gnu/bin/rust-lld.exe")
  if(NOT EXISTS "${GOMOKU_RUST_LLD}")
    message(FATAL_ERROR
      "venv Rust toolchain is missing. Run: "
      ".venv\\Scripts\\python.exe tools\\build\\bootstrap_toolchain.py")
  endif()
  set(GOMOKU_CARGO_ENV
    "RUSTUP_HOME=${GOMOKU_VENV}/rustup"
    "CARGO_HOME=${GOMOKU_VENV}/cargo"
    "RUSTUP_TOOLCHAIN=stable-x86_64-pc-windows-gnu"
    "RUSTC=${GOMOKU_VENV}/cargo/bin/rustc.exe"
    "CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=${GOMOKU_RUST_LLD}"
    "CC_x86_64_pc_windows_gnu=${CMAKE_SOURCE_DIR}/tools/build/zig-cc.cmd"
    "CXX_x86_64_pc_windows_gnu=${CMAKE_SOURCE_DIR}/tools/build/zig-cxx.cmd"
    "RUSTFLAGS=-Clink-self-contained=yes"
    "ZIG_GLOBAL_CACHE_DIR=${GOMOKU_VENV}/zig-cache"
    "ZIG_LOCAL_CACHE_DIR=${GOMOKU_VENV}/zig-cache/local")
else()
  find_program(GOMOKU_CARGO cargo
    HINTS "$ENV{USERPROFILE}/.cargo/bin" "$ENV{HOME}/.cargo/bin")
  set(GOMOKU_CARGO_ENV "")
endif()
if(NOT GOMOKU_CARGO)
  message(FATAL_ERROR
    "cargo was not found. Ticket 03 builds the vendored quintet VCF solver "
    "with the stable Rust toolchain. On Windows run "
    "(venv\\Scripts\\python.exe tools/build/bootstrap_toolchain.py); on other "
    "platforms install Rust (https://rustup.rs).")
endif()
if(GOMOKU_CARGO_ENV)
  set(GOMOKU_CARGO_RUN ${CMAKE_COMMAND} -E env ${GOMOKU_CARGO_ENV}
      "${GOMOKU_CARGO}")
else()
  set(GOMOKU_CARGO_RUN "${GOMOKU_CARGO}")
endif()

# Ticket 07: AVX2 is part of the deployment contract (added by the top-level
# CMakeLists for non-MSVC toolchains).

# Zig's linker is memory-hungry (each link loads LLVM); run link steps one at
# a time while compilation stays parallel. This keeps the venv toolchain from
# hitting LLVM's out-of-memory during multi-target builds.
if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS link_pool=1)
  set(CMAKE_JOB_POOL_LINK link_pool)
endif()
add_test(NAME vcf_bridge_tests
  COMMAND ${GOMOKU_CARGO_RUN} test --release
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/vcf-bridge/Cargo.toml")
set_tests_properties(vcf_bridge_tests PROPERTIES TIMEOUT 300)

add_custom_target(vcf_bridge ALL
  COMMAND ${GOMOKU_CARGO_RUN} build --release
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/vcf-bridge/Cargo.toml"
  BYPRODUCTS "${VCF_BRIDGE_LIB}"
  COMMENT "Building vendored quintet VCF bridge (Rust, venv toolchain)"
  VERBATIM)

function(gomoku_link_vcf_bridge target)
  add_dependencies(${target} vcf_bridge)
  target_link_libraries(${target} PRIVATE "${VCF_BRIDGE_LIB}")
  if(WIN32)
    # Native libraries a Rust static library needs on Windows. Zig resolves
    # the mingw-w64 import libraries it bundles itself.
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
  ${ENGINE_CORE_SRC})
target_compile_features(gomoku_nnue PRIVATE cxx_std_20)
target_include_directories(gomoku_nnue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(gomoku_nnue)

# Cross-language parity probe (ticket 09): Python drives it over stdin/stdout.
add_executable(nnue_probe src/nnue_probe.cpp ${ENGINE_CORE_SRC})
target_compile_features(nnue_probe PRIVATE cxx_std_20)
target_include_directories(nnue_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(nnue_probe)

# Ticket 07: reference-path probe for the deployed GMVALUE1 value network.
add_executable(value_probe src/value_probe.cpp ${ENGINE_CORE_SRC})
target_compile_features(value_probe PRIVATE cxx_std_20)
target_include_directories(value_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(value_probe)

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

# Ticket 07: GMVALUE1 loader, scalar reference forward, full first-layer
# rebuild, side-to-move conversion, rejection paths and Alpha-Beta integration.
add_executable(value_net_tests tests/value_net_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(value_net_tests PRIVATE cxx_std_20)
target_include_directories(value_net_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
gomoku_link_vcf_bridge(value_net_tests)
add_test(NAME value_net_tests COMMAND value_net_tests)

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

  # Ticket 07: C++ GMVALUE1 reference forward / full rebuild vs the Python
  # integer reference and the exported training model. Ticket 11 adds the
  # incremental place/undo walk (scalar vs AVX2) and single-inference timing.
  add_test(NAME value_parity_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/value_parity_tests.py"
            "$<TARGET_FILE:value_probe>")
  set_tests_properties(value_parity_tests PROPERTIES TIMEOUT 600)

  # Ticket 11: real Alpha-Beta search and paired-opening play across the
  # avx2 / scalar / full-rebuild backends, plus measured search throughput.
  add_test(NAME value_backend_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/value_backend_tests.py"
            "$<TARGET_FILE:gomoku_nnue>"
            "${CMAKE_SOURCE_DIR}/tools")
  set_tests_properties(value_backend_tests PROPERTIES TIMEOUT 300)

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

  # Ticket 05: rolling sample pool boundary (no engine), collection with a
  # real executor and a deterministic stub teacher, and the offscreen GUI
  # collection panel against the real teacher (skipped when Rapfi is absent).
  add_test(NAME pool_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/pool_tests.py"
            "${CMAKE_SOURCE_DIR}/tools")
  set_tests_properties(pool_tests PROPERTIES TIMEOUT 180)

  add_test(NAME collect_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/collect_tests.py"
            "$<TARGET_FILE:gomoku_nnue>"
            "${CMAKE_SOURCE_DIR}/tools")
  set_tests_properties(collect_tests PROPERTIES TIMEOUT 300)

  add_test(NAME gui_collect_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_collect_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_collect_smoke PROPERTIES
    TIMEOUT 300 SKIP_RETURN_CODE 77 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  # Ticket 06: new value network training/export (architecture, contract, D4,
  # real small training, checkpoint resume, GMVALUE1 fidelity) and the
  # offscreen GUI training panel.
  add_test(NAME value_train_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/value_train_tests.py")
  set_tests_properties(value_train_tests PROPERTIES TIMEOUT 300)

  add_test(NAME gui_train_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_train_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_train_smoke PROPERTIES
    TIMEOUT 300 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  # Ticket 09: full self-play game with NNUE evaluation + bad-weight fallback.
  add_test(NAME nnue_blackbox
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/nnue_blackbox.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(nnue_blackbox PROPERTIES TIMEOUT 300)

  # Ticket 07: real games with a GMVALUE1 model through the executor and the
  # Gomocup protocol, side-to-move signs, terminal/VCF precedence, strict
  # rejection of damaged or incompatible models, and replay consistency.
  add_test(NAME value_engine_blackbox
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/value_engine_blackbox.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(value_engine_blackbox PROPERTIES TIMEOUT 600)

  # Ticket 07: offscreen GUI model picker / identity / candidate play / failure.
  add_test(NAME gui_model_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_model_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_model_smoke PROPERTIES
    TIMEOUT 300 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  add_test(NAME history_evaluation_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/history_evaluation_tests.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(history_evaluation_tests PROPERTIES TIMEOUT 300)

  add_test(NAME gui_history_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_history_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_history_smoke PROPERTIES
    TIMEOUT 300 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  add_test(NAME training_task_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/training_task_tests.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(training_task_tests PROPERTIES TIMEOUT 300)

  # Ticket 12: pause at data/training/evaluation boundaries, restart recovery
  # with frozen config/data/validation, stale lease/staging cleanup and
  # evaluation-run reuse.
  add_test(NAME recovery_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/recovery_tests.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(recovery_tests PROPERTIES TIMEOUT 600)

  add_test(NAME gui_recovery_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_recovery_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_recovery_smoke PROPERTIES
    TIMEOUT 600 SKIP_RETURN_CODE 77 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  # Ticket 13: simplified visual hierarchy, primary actions and disclosure.
  add_test(NAME gui_ui_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_ui_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_ui_smoke PROPERTIES
    TIMEOUT 120 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  # Ticket 14: read-only live mirror of a background game.
  add_test(NAME gui_live_board_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_live_board_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_live_board_smoke PROPERTIES
    TIMEOUT 120 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  add_test(NAME run_support_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_support_tests.py")
  set_tests_properties(run_support_tests PROPERTIES TIMEOUT 30)

  # Candidate models and historical data: planners, guards, CLI and dialog.
  add_test(NAME model_cleanup_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/model_cleanup_tests.py")
  set_tests_properties(model_cleanup_tests PROPERTIES TIMEOUT 60)

  add_test(NAME storage_cleanup_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/storage_cleanup_tests.py")
  set_tests_properties(storage_cleanup_tests PROPERTIES TIMEOUT 60)

  add_test(NAME gui_cleanup_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_cleanup_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_cleanup_smoke PROPERTIES
    TIMEOUT 120 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  add_test(NAME gui_training_task_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_training_task_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_training_task_smoke PROPERTIES
    TIMEOUT 300 SKIP_RETURN_CODE 77 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")

  add_test(NAME selfplay_task_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/selfplay_task_tests.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(selfplay_task_tests PROPERTIES TIMEOUT 300)
  add_test(NAME gui_selfplay_smoke
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gui_selfplay_smoke.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(gui_selfplay_smoke PROPERTIES
    TIMEOUT 300 ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
endif()

