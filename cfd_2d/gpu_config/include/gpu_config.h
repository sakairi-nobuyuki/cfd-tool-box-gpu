#ifndef __GPU_CONFIG_H__
#define __GPU_CONFIG_H__

#include <cuda_runtime.h>
#include <comput_unit_config.h>

class GpuConfig {
    private:
        int _n_device, _n_dev;
        int _n_max_thread;
        ComputUnitConfig _blockConf, _gridConf, _threadConf;
        cudaError_t _error_id;
        cudaDeviceProp _deviceProp;
    public:
        GpuConfig(int n_dev = 0);
        cudaDeviceProp getGpuDeviceProp(int i_dev);
        void setThreadConfig(cudaDeviceProp _deviceProp);
        void setBlcokConfig(cudaDeviceProp _deviceProp);
        void setGridConfig(cudaDeviceProp _deviceProp);
        ComputUnitConfig getThreadConfig();
};


#endif