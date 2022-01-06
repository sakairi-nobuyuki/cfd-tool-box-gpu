# include CppUTest headers
include_directories($ENV{CPPUTEST_HOME}/include)

# add cpputest library
add_library(imp_cpputest STATIC IMPORTED)
set_property(TARGET imp_cpputest PROPERTY
             IMPORTED_LOCATION $ENV{CPPUTEST_HOME}/lib/libCppUTest.a)

add_executable(main test_magical_banana.cpp)
target_link_libraries(main magical_banana imp_cpputest)
             