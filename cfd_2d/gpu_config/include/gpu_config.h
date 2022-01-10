#ifndef __GPU_CONFIG_H__
#define __GPU_CONFIG_H__

#include <cuda_runtime.h>

class GpuConfig {
    private:
        int _n_device, _dev;
        cudaError_t _error_id;
        cudaDeviceProp _deviceProp;
    public:
        GpuConfig();

};


#endif