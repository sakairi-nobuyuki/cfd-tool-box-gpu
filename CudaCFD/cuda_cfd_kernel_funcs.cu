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


__global__ void obtain_slope(double *Slope, double *DeltaPlus, double *DeltaMinus, double epsilon, int n_len) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (0 <= i && i < n_len) Slope[i] = (2.0 * DeltaPlus[i] * DeltaMinus[i] + epsilon) 
        / (pow(DeltaPlus[i], 2.0) + pow(DeltaMinus[i], 2.0) + epsilon);
}

void obtain_slope_device(double *Slope, double *DeltaPlus, double *DeltaMinus, double epsilon, GridDim *dimGrid, BlockDim *dimBlock, int n_len) {
    dim3 grid(dimGrid->x, dimGrid->y), block(dimBlock->x, dimBlock->y, dimBlock->z);
    obtain_slope<<<grid, block>>>(Slope, DeltaPlus, DeltaMinus, epsilon, n_len);
}


__global__ void obtain_cell_intface_values(double *R, double *L, double *Q, double *DeltaPlus, double *DeltaMinus, double *s, double kappa, int n_len) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (0 <= i && i < n_len - 1) L[i] = Q[i] + 0.25 * s[i] * ((1.0 - kappa * s[i]) * DeltaMinus[i] + (1.0 + kappa) * s[i] * DeltaPlus[i]);
    if (i == n_len - 1) L[n_len - 1] = L[n_len - 2];

    if (0 < i && i < n_len) R[i] = Q[i+1] - 0.25 * s[i+1] * ((1.0 - kappa * s[i+1]) * DeltaPlus[i+1] + (1.0 + kappa * s[i+1]) * DeltaMinus[i+1]);
    if (i == 0) R[0] = R[1];
}

void obtain_cell_intface_value_device(double *R, double *L, double *Q, double *DeltaPlus, double *DeltaMinus, double *s, double kappa, GridDim *dimGrid, BlockDim *dimBlock, int n_len) {
    dim3 grid(dimGrid->x, dimGrid->y), block(dimBlock->x, dimBlock->y, dimBlock->z);
    obtain_cell_intface_values<<<grid, block>>>(R, L, Q, DeltaPlus, DeltaMinus, s, kappa, n_len);
}

void obtain_cell_intface_value_from_Q_device(double *R, double *L, double *s, double *BarDeltaPlus, double *BarDeltaMinus, double *DeltaPlus, double *DeltaMinus, double *Q, double kappa, double epsilon, double b, GridDim *dimGrid, BlockDim *dimBlock, int n_len) {
    dim3 grid(dimGrid->x, dimGrid->y), block(dimBlock->x, dimBlock->y, dimBlock->z);
    cudaDeviceSynchronize();
    obtain_delta_plus_device<<<grid, block>>>(DeltaPlus, Q, n_len);
    obtain_delta_minus_device<<<grid, block>>>(DeltaMinus, Q, n_len);
    cudaDeviceSynchronize();
    obtain_minmod<<<grid, block>>>(BarDeltaPlus, BarDeltaMinus, DeltaPlus, DeltaMinus, b, n_len);
    cudaDeviceSynchronize();
    obtain_slope<<<grid, block>>>(s, BarDeltaPlus, BarDeltaMinus, epsilon, n_len);
    cudaDeviceSynchronize();
    obtain_cell_intface_values<<<grid, block>>>(R, L, Q, DeltaPlus, DeltaMinus, s, kappa, n_len);    
    cudaDeviceSynchronize();
}

__host__ void free_cuda_memory(double *U) {
    cudaFree(U);
}



#endif