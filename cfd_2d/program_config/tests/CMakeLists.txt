# add cpputest library
add_library(imp_cpputest STATIC IMPORTED)
set_property(TARGET imp_cpputest PROPERTY
    IMPORTED_LOCATION $ENV{CPPUTEST_HOME}/lib/libCppUTest.a
)

# include directory of program config
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(test_program_config test_program_config.cpp)             
target_link_libraries(test_program_config program_config imp_cpputest)