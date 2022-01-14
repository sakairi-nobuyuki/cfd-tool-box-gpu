#include <stdio.h>
#include <cuda_runtime.h>
#include <gpu_config.h>



GpuConfig::GpuConfig(int i_dev) {

    _error_id = cudaGetDeviceCount(&_n_dev);

    if (_error_id != cudaSuccess) {
        printf("cudaDeviceCount returned %d -> %s\n", (int)_error_id, cudaGetErrorString(_error_id));
        printf("Failed to initialize GPU\n");
        exit(1);
    }

    if (i_dev > _n_dev - 1) {
        printf("Assinged GPU is not exists\n");
        printf("  The number of GPU devices: %d\n", _n_dev);
        printf("  Assigned GPU: %d\n", i_dev);
        exit(1);

    }

    _deviceProp =  getGpuDeviceProp(i_dev);
    
    _threadConf.set1DimConfig(_deviceProp.maxThreadsPerBlock);
    _blockConf.set3DimConfig(_deviceProp.maxThreadsDim[0], _deviceProp.maxThreadsDim[1], _deviceProp.maxThreadsDim[2]);
    _gridConf.set3DimConfig (_deviceProp.maxGridSize[0],   _deviceProp.maxGridSize[1],   _deviceProp.maxGridSize[2]);

    printf("GPU device %d th: name: %s\n", i_dev, _deviceProp.name);
    printf("  Max number of threads per block: %d\n", _threadConf.getDimX());
    printf("  Max size of each dimension of a blcok: (%d, %d, %d)\n", 
        _blockConf.getDimX(), _blockConf.getDimY(), _blockConf.getDimZ());
    printf("  Max size of each dimension of a grid: (%d, %d, %d)\n", 
        _gridConf.getDimX(), _gridConf.getDimY(), _gridConf.getDimZ()); 
    printf("  Initializing GPU config finished\n");
    //exit(EXIT_SUCCESS);
}

cudaDeviceProp GpuConfig::getGpuDeviceProp(int i_dev) {
    //cudaDeviceProp getGpuDeviceProp::GpuConfig(int n_dev) {
    // Get GPU configuration with CUDA function.
    // Args:
    //   int i_dev: Indicating i_dev th GPU device out of _n_dev GPU devices.
    // Returns:
    //   cudaDeviceProp _deviceProp: Device configuration get by CUDA.
    cudaSetDevice(i_dev);
    cudaGetDeviceProperties(&_deviceProp, i_dev);

    return _deviceProp;
}

ComputUnitConfig GpuConfig::getThreadConfig() {
    return _threadConf;
}