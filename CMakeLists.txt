cmake_minimum_required(VERSION 3.20)
project(gomoku_nnue LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  # AVX2 only, no runtime SIMD dispatch (single target machine).
  add_compile_options(/arch:AVX2 /W4 /utf-8 /permissive-)
else()
  # Cross-platform toolchains (clang/gcc) enable the same AVX2 intrinsics
  # nnue.cpp uses. The build no longer hard-requires MSVC.
  add_compile_options(-mavx2)
endif()

include(CTest)

add_subdirectory(engine)
