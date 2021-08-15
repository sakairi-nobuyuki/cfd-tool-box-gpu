#ifndef __CUDA_FRONT_END__
#define __CUDA_FRONT_END__

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#include "cuda_cfd_kernel_funcs.h"

using namespace std;


__host__ void allocate_cuda_memory(void **U, int n_bytes) {
//    printf("  In cuda_cfd_kernel_functions, allocate_cuda_memory, before allocating memory address of U: %p\n", &U);
    //printf("  In cuda_cfd_kernel_functions, allocate_cuda_memory, before allocating memory address of V:  %p\n", V);
    cudaMalloc (U, n_bytes);
//    printf("  In cuda_cfd_kernel_functions, allocate_cuda_memory, after allocating memory address of U:  %p\n", &U);
}

__host__ void copy_memory_host_to_device(double *gU, double *U, int n_bytes) {
    cudaMemcpy (gU, U, n_bytes, cudaMemcpyHostToDevice);
}

__host__ void copy_memory_device_to_host(double *U, double *gU, int n_bytes) {
    
    cudaMemcpy (U, gU, n_bytes, cudaMemcpyDeviceToHost);

}

__host__ void copy_memory_device_to_device(double *gV, double *gU, int n_bytes) {

    cudaMemcpy (gV, gU, n_bytes, cudaMemcpyDeviceToDevice);
}

__host__ void copy_memory_mock() {
    double *gU, *U, *V;
    int i, n;

    printf("In memory test mock, before memory allocation, address U: %p, V: %p, gU: %p\n", U, V, gU);

    n = 10;
    U = (double *) malloc(sizeof(double) * n);
    V = (double *) malloc(sizeof(double) * n);
    cudaMalloc((double **) &gU, sizeof(double) * n);

    printf("In memory test mock, after memory allocation, address U: %p, V: %p, gU: %p\n", U, V, gU);

    for (i = 0; i < n; i++) U[i] = (double) i;
    printf("In mock test, U\n  ");
    for (i = 0; i < n; i++) printf("%lf  ", U[i]);
    printf("\n");

    cudaMemcpy(gU, U, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(V, gU, sizeof(double) * n, cudaMemcpyDeviceToHost);
    printf("In mock test, V\n  ");
    for (i = 0; i < n; i++) printf("%lf  ", V[i]);
    printf("\n");
    printf("In mock test, assert U == V\n  ");
    for (i = 0; i < n; i++) assert(U[i] == V[i]);
    printf("\n");

    printf("In memory test mock, after test memory allocation, address U: %p, V: %p, gU: %p\n", U, V, gU);
}

void cuda_device_synchronize() {
    cudaDeviceSynchronize();
}


__global__ void obtain_delta_plus_device(double *gDelta, double *gU, int n_len) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < n_len) gDelta[i] = gU[i + 1] - gU[i];
    if (i == 0)             gDelta[i] = gU[i + 1] - gU[i];
    if (i == n_len - 1)     gDelta[i] = gDelta[i - 1];
}

__global__ void obtain_delta_minus_device(double *gDelta, double *gU, int n_len) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < n_len) gDelta[i] = gU[i] - gU[i - 1];
    if (i == 0)             gDelta[i] = gDelta[i + 1];
    if (i == n_len - 1)     gDelta[i] = gU[i] - gU[i - 1];
}

//void obtain_deltas_device(double *gU, double *gDeltaPlus, double *gDeltaMinus, int n_len) {
void obtain_deltas_device(double *gDeltaPlus, double *gDeltaMinus, double *gU, GridDim *dimGrid, BlockDim *dimBlock, int n_len) {
    dim3 grid(dimGrid->x, dimGrid->y), block(dimBlock->x, dimBlock->y, dimBlock->z);
    obtain_delta_plus_device<<<grid, block>>>(gDeltaPlus, gU, n_len);
    obtain_delta_minus_device<<<grid, block>>>(gDeltaMinus, gU, n_len);

}


__global__ void obtain_minmod(double *gBarDeltaPlus, double *gBarDeltaMinus, double *gDeltaPlus, double *gDeltaMinus, double b, int n_len) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 <= i && i < n_len) 
        gBarDeltaPlus[i] 
            = copysignf(1.0, gDeltaPlus[i]) 
            * fmaxf(0.0, fminf(fabs(gDeltaPlus[i]), copysignf(1.0, gDeltaPlus[i]) * b * gDeltaMinus[i]));
    if (0 <= i && i < n_len) 
        gBarDeltaMinus[i] 
            = copysignf(1.0, gDeltaMinus[i]) 
            * fmaxf(0.0, fminf(fabs(gDeltaMinus[i]), copysignf(1.0, gDeltaMinus[i]) * b * gDeltaPlus[i]));

            

}

void obtain_minmod_device(double *gBarDeltaPlus, double *gBarDeltaMinus, double *gDeltaPlus, double *gDeltaMinus, double b, GridDim *dimGrid, BlockDim *dimBlock, int n_len) {
    dim3 grid(dimGrid->x, dimGrid->y), block(dimBlock->x, dimBlock->y, dimBlock->z);
    obtain_minmod<<<grid, block>>>(gBarDeltaPlus, gBarDeltaMinus, gDeltaPlus, gDeltaMinus, b, n_len);

}


__global__ void obtain_slope(double *gSlope, double *gDeltaPlus, double *gDeltaMinus, double epsilon, int n_len) {
    
    
}


__host__ void free_cuda_memory(double *U) {
    cudaFree(U);
}



#endif