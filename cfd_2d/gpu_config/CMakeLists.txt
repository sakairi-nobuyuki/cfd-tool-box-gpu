cmake_minimum_required(VERSION 3.1)
project(gpu_config)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -Wall")

# Add sub directories
add_subdirectory(src)
add_subdirectory(tests)
