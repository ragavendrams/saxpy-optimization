cmake_minimum_required(VERSION 3.10)

# Set the project name
project(saxpy-optimization)

# Default to an optimized build. Without a build type CMake passes no -O flag,
# so the scalar loops would be measured at -O0.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Local installation of google-benchmark
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)

# Add the executable
add_executable(saxpy-optimization main.cpp)
target_link_libraries(saxpy-optimization PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)
target_compile_options(saxpy-optimization PRIVATE -march=native -mavx2 -mfma -fopt-info-vec-optimized)
#set(CMAKE_VERBOSE_MAKEFILE ON)