# GomokuNNUE engine target
add_executable(gomoku_nnue
  src/main.cpp
  src/board.cpp
  src/renju.cpp
  src/protocol.cpp
  src/random_player.cpp)
target_compile_features(gomoku_nnue PRIVATE cxx_std_20)
target_include_directories(gomoku_nnue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Board / random-player unit tests (ticket 02)
add_executable(board_tests tests/board_tests.cpp src/board.cpp src/renju.cpp src/random_player.cpp)
target_compile_features(board_tests PRIVATE cxx_std_20)
target_include_directories(board_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME board_tests COMMAND board_tests)

# Renju forbidden-move unit tests (ticket 03)
add_executable(renju_tests tests/renju_tests.cpp src/board.cpp src/renju.cpp)
target_compile_features(renju_tests PRIVATE cxx_std_20)
target_include_directories(renju_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME renju_tests COMMAND renju_tests)

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
