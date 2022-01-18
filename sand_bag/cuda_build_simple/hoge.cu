#include <stdio.h>
#include "hoge.h"

Hoge::Hoge(int n_size){
    _n_thread_x = 128;
    _n_block_x  = n_size / 128 + 1;
    _n_size = _n_block_x * _n_thread_x;
    _n_size_orig = n_size;

    printf("n_block_x: %d, n_thread_x: %d, n_size: %d, n_size_orig: %d\n", _n_block_x, _n_thread_x, _n_size, _n_size_orig);

    cudaMalloc(&_u_gpu, sizeof(int) * _n_size);
    _u_cpu = (int *) malloc(sizeof(int) * _n_size);
}

void Hoge::print(){
    int i;
    cudaMemcpy(_u_cpu, _u_gpu, sizeof(int) * _n_size, cudaMemcpyDeviceToHost);

    printf("res:");
    for (i = 0; i < _n_size_orig; i++) printf("%d ", _u_cpu[i]);
    printf("\n");
}

__global__ void sum_global(int *n){
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    n[i] = i;
}

void Hoge::sum() {
    sum_global<<<_n_block_x, _n_thread_x>>>(_u_gpu);
}
