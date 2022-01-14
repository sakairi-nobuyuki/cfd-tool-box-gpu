# add cpputest library
add_library(imp_cpputest STATIC IMPORTED)
set_property(TARGET imp_cpputest PROPERTY
    IMPORTED_LOCATION $ENV{CPPUTEST_HOME}/lib/libCppUTest.a
)

# add CUDA
find_package(CUDA REQUIRED)
#add_library(imp_cuda STATIC IMPORTED)
#set_property(TARGET imp_cuda PROPERTY
#    IMPORTED_LOCATION /usr/local/${CUDA_VERSION}/lib64
#)

# include directory of program config
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(test_gpu_config test_gpu_config.cu)             
target_link_libraries(test_gpu_config gpu_config imp_cpputest)

