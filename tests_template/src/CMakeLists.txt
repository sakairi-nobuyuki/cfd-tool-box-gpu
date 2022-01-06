add_library(magical_banana
  SHARED
    magical_banana.cpp
)
# greetingライブラリのインクルードディレクトリを教えてあげる
# PROJECT_SOURCE_DIRはこのプロジェクトのルートディレクトリの絶対パス
target_include_directories(magical_banana
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)

#add_executable(main test_magical_banana.cpp)2
# a.outをコンパイルする際にgreetingsをリンクする
#target_link_libraries(main magical_banana test_magical_banana)
