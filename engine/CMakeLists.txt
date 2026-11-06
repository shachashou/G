set(ENGINE_CORE_SRC
  src/board.cpp
  src/renju.cpp
  src/eval.cpp
  src/nnue.cpp
  src/movegen.cpp
  src/search.cpp
  src/vcf.cpp
  src/random_player.cpp
  src/binpack.cpp)

# GomokuNNUE engine target
add_executable(gomoku_nnue
  src/main.cpp
  src/protocol.cpp
  ${ENGINE_CORE_SRC})
target_compile_features(gomoku_nnue PRIVATE cxx_std_20)
target_include_directories(gomoku_nnue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Cross-language parity probe (ticket 09): Python drives it over stdin/stdout.
add_executable(nnue_probe src/nnue_probe.cpp ${ENGINE_CORE_SRC})
target_compile_features(nnue_probe PRIVATE cxx_std_20)
target_include_directories(nnue_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Self-play datagen worker (ticket 07). One process owns one shard file;
# tools/datagen.py launches many of them in parallel.
add_executable(datagen src/datagen.cpp ${ENGINE_CORE_SRC})
target_compile_features(datagen PRIVATE cxx_std_20)
target_include_directories(datagen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Board / random-player unit tests (ticket 02)
add_executable(board_tests tests/board_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(board_tests PRIVATE cxx_std_20)
target_include_directories(board_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME board_tests COMMAND board_tests)

# Renju forbidden-move unit tests (ticket 03)
add_executable(renju_tests tests/renju_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(renju_tests PRIVATE cxx_std_20)
target_include_directories(renju_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME renju_tests COMMAND renju_tests)

# Incremental evaluation consistency tests (ticket 04)
add_executable(eval_tests tests/eval_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(eval_tests PRIVATE cxx_std_20)
target_include_directories(eval_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME eval_tests COMMAND eval_tests)

# Search tactical behaviour tests (ticket 04)
add_executable(search_tests tests/search_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(search_tests PRIVATE cxx_std_20)
target_include_directories(search_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME search_tests COMMAND search_tests)

# VCF (continuous-fours) tactic tests (ticket 05)
add_executable(vcf_tests tests/vcf_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(vcf_tests PRIVATE cxx_std_20)
target_include_directories(vcf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME vcf_tests COMMAND vcf_tests)

# 64-byte binpack record round-trip tests (ticket 07)
add_executable(binpack_tests tests/binpack_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(binpack_tests PRIVATE cxx_std_20)
target_include_directories(binpack_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME binpack_tests COMMAND binpack_tests)

# Ticket 09: scalar NNUE weight loader, integer forward and incremental
# accumulator consistency.
add_executable(nnue_tests tests/nnue_tests.cpp ${ENGINE_CORE_SRC})
target_compile_features(nnue_tests PRIVATE cxx_std_20)
target_include_directories(nnue_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME nnue_tests COMMAND nnue_tests)

# Protocol black-box test: drives the exe via stdin/stdout.
# Prefer the project venv (py -3.12 -m venv .venv), never the system Python.
set(GOMOKU_VENV_PYTHON "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
if(EXISTS "${GOMOKU_VENV_PYTHON}")
  set(GOMOKU_PYTHON "${GOMOKU_VENV_PYTHON}")
else()
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_FOUND)
    set(GOMOKU_PYTHON "${Python3_EXECUTABLE}")
  endif()
endif()
if(GOMOKU_PYTHON)
  # Ticket 12: bounded diagnostic/paired-gate/budget CLI contracts.
  add_test(NAME m5_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/m5_tests.py"
            --engine "$<TARGET_FILE:gomoku_nnue>" --probe "$<TARGET_FILE:nnue_probe>")
  set_tests_properties(m5_tests PROPERTIES TIMEOUT 120)

  add_test(NAME protocol_blackbox
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/protocol_blackbox.py"
            "$<TARGET_FILE:gomoku_nnue>")

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

  # Ticket 11: bootstrap loop pure helpers and command constructors.
  add_test(NAME bootstrap_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/bootstrap_tests.py")
  set_tests_properties(bootstrap_tests PROPERTIES TIMEOUT 60)

  # Ticket 15/17 follow-up: move-correlation diagnostic pure helpers.
  add_test(NAME move_correlation_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/move_correlation_tests.py")
  set_tests_properties(move_correlation_tests PROPERTIES TIMEOUT 60)

  add_test(NAME clean_search_ranking_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/clean_search_ranking_tests.py")
  set_tests_properties(clean_search_ranking_tests PROPERTIES TIMEOUT 60)

  add_test(NAME distillation_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/distillation_tests.py"
            "$<TARGET_FILE:nnue_probe>")
  set_tests_properties(distillation_tests PROPERTIES TIMEOUT 120)

  add_test(NAME local_ranking_diagnosis_tests
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/local_ranking_diagnosis_tests.py"
            "$<TARGET_FILE:nnue_probe>")
  set_tests_properties(local_ranking_diagnosis_tests PROPERTIES TIMEOUT 120)

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

  # Ticket 09: full self-play game with NNUE evaluation + bad-weight fallback.
  add_test(NAME nnue_blackbox
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/nnue_blackbox.py"
            "$<TARGET_FILE:gomoku_nnue>")
  set_tests_properties(nnue_blackbox PROPERTIES TIMEOUT 300)
endif()
