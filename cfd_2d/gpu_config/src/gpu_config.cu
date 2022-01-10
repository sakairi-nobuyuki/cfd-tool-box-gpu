#include <cuda_runtime.h>
#include <gpu_config.h>


GpuConfig::GpuConfig() {
    cudaGetDeviceProperties(&_deviceProp, _dev);


}
