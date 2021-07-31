#ifndef __CUDA_MEMORY_CONFIG_CPP__
#define __CUDA_MEMORY_CONFIG_CPP__

#include "cuda_memory_config.h"

BlockDim::BlockDim() {
    x = 0;
    y = 0;
    z = 0;
}

GridDim::GridDim() {
    x = 0;
    y = 0;
}

void setCudaGridBlockConfig1D(int n_len, GridDim *dimGrid, BlockDim *dimBlock) {
    dimBlock->x = 2;
    dimBlock->y = 1;
    dimBlock->z = 1;
    dimGrid->x = n_len / dimBlock->x;
    dimGrid->y = 1;
}


#endif