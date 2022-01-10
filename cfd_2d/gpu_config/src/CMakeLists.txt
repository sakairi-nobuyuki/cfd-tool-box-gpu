find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    CUDA_ADD_LIBRARY(gpu_config STATIC
        gpu_config.cu
        
    )
    target_include_directories(gpu_config
        PUBLIC ${PROJECT_SOURCE_DIR}/include
    )
endif()