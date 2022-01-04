# CMakeのバージョンを設定
cmake_minimum_required(VERSION 3.10)

# プロジェクト名と使用する言語を設定
project(magical_banana CXX)

# add subdirectories
add_subdirectory(src)



# a.outという実行ファイルをmain.cppとhello.cppから作成
#add_executable(main main.cpp magical_banana.cpp)