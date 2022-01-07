add_library(program_config
    SHARED
        memory_config.cpp
)

target_include_directories(program_config
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)