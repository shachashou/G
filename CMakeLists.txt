cmake_minimum_required(VERSION 3.20)
project(gomoku_nnue LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # Windows builds are venv-contained (ziglang clang/libc++/mingw-w64). MSVC
  # would need Visual Studio files outside the venv and a different Rust ABI.
  message(FATAL_ERROR
    "MSVC is not supported: build with "
    ".venv\\Scripts\\python.exe tools\\build\\build.py")
else()
  # Cross-platform toolchains (clang/gcc/zig) enable the same AVX2 intrinsics
  # nnue.cpp uses.
  add_compile_options(-mavx2)
endif()

include(CTest)

add_subdirectory(engine)
