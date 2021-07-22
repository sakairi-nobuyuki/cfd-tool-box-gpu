#ifndef __CUDA_FRONT_END__
#define __CUDA_FRONT_END__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cuda_cfd_kernel_funcs.h"




__host__ void allocate_cuda_memory(double *U, int n_bytes) {
//__global__ void allocate_cuda_memory(double *U, int n_bytes) {
    //printf('hoge');
    cudaMalloc ((void**) &U,    n_bytes);
    //cout << 'U: ' << U << ', &U: ' << &U << endl;
    //cudaMalloc ((void**) &U,    n_bytes);
}

__host__ void copy_memory_host_to_device(double *gU, double *U, int n_bytes) {
    cudaMemcpy (gU, U, n_bytes, cudaMemcpyHostToDevice);
}

__host__ void copy_memory_device_to_host(double *U, double *gU, int n_bytes) {
    cudaMemcpy (U, gU, n_bytes, cudaMemcpyDeviceToHost);

}

void cuda_device_synchronize() {
    cudaDeviceSynchronize();
}


__global__ void obtain_delta_plus_device(double *gU, double *gDelta, int n_len) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < n_len - 2) gDelta[i] = gU[i+1] - gU[i];
    if (i == n_len - 1) gDelta[i] = gDelta[i-1];
}

__global__ void obtain_delta_minus_device(double *gU, double *gDelta, int n_len) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (1 < i && i < n_len - 1) gDelta[i] = gU[i] - gU[i - 1];
    if (i == n_len - 1) gDelta[0] = gDelta[1];
}

void obtain_deltas_device(double *gU, double *gDeltaPlus, double *gDeltaMinus, int n_len) {
    obtain_delta_plus_device<<<1, 1>>>(gU, gDeltaPlus, n_len);
    obtain_delta_minus_device<<<1, 1>>>(gU, gDeltaMinus, n_len);
}



__host__ void free_cuda_memory(double *U) {
    cudaFree(U);
}



#endif