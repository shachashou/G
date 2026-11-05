# GomokuNNUE engine target
add_executable(gomoku_nnue
  src/main.cpp
  src/board.cpp
  src/protocol.cpp
  src/random_player.cpp)
target_compile_features(gomoku_nnue PRIVATE cxx_std_20)
target_include_directories(gomoku_nnue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Board / random-player unit tests (ticket 02)
add_executable(board_tests tests/board_tests.cpp src/board.cpp src/random_player.cpp)
target_compile_features(board_tests PRIVATE cxx_std_20)
target_include_directories(board_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME board_tests COMMAND board_tests)

# Protocol black-box test: drives the exe via stdin/stdout
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  add_test(NAME protocol_blackbox
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/protocol_blackbox.py"
            "$<TARGET_FILE:gomoku_nnue>")
endif()
