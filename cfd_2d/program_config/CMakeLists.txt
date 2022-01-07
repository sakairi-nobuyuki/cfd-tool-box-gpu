# CMakeのバージョンを設定
cmake_minimum_required(VERSION 3.10)

# プロジェクト名と使用する言語を設定
project(program_config CXX)

# add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
