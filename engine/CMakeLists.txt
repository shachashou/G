set(ENGINE_CORE_SRC
  src/board.cpp
  src/renju.cpp
  src/eval.cpp
  src/movegen.cpp
  src/search.cpp
  src/random_player.cpp)

# GomokuNNUE engine target
add_executable(gomoku_nnue
  src/main.cpp
  src/protocol.cpp
  ${ENGINE_CORE_SRC})
target_compile_features(gomoku_nnue PRIVATE cxx_std_20)
target_include_directories(gomoku_nnue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

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
  add_test(NAME protocol_blackbox
    COMMAND "${GOMOKU_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/protocol_blackbox.py"
            "$<TARGET_FILE:gomoku_nnue>")
endif()
