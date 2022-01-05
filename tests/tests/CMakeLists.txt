#add_library(test_magical_banana
#  SHARED
#    test_magical_banana.cpp
#  )
#target_link_libraries(magical_banana)
#target_include_directories(magical_banana
#  PUBLIC ${PROJECT_SOURCE_DIR}/include
#)
target_include_directories(cpputest
  PUBLIC ${CPPUTEST_HOME}/include
)
target_link_directories(cpputest
  PUBLIC ${CPPUTEST_HOME}/lib
)
add_executable(main test_magical_banana.cpp)
target_link_libraries(main magical_banana CppUTest CppUTestExt)
