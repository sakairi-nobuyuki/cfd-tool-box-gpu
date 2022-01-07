# include CppUTest headers
include_directories($ENV{CPPUTEST_HOME}/include)

# add cpputest library
add_library(imp_cpputest STATIC IMPORTED)
set_property(TARGET imp_cpputest PROPERTY
    IMPORTED_LOCATION $ENV{CPPUTEST_HOME}/lib/libCppUTest.a
)

add_executable(test_program_config test_memory_config.cpp)             
target_link_libraries(test_program_config memory_config imp_cpputest)