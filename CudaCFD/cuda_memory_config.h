#ifndef __CUDA_MEMORY_CONFIG_H__
#define __CUDA_MEMORY_CONFIG_H__

struct BlockDim {
    public:
        BlockDim();
        int x, y, z;
};

struct GridDim {
    public:
        GridDim();
        int x, y;
};

void setCudaGridBlockConfig1D(int n_len, GridDim *dimGrid, BlockDim *dimBlock);
//setCudaBlockGridConfig1D(int n_len, BlockDim dimBlock, GridDim dimGrid);



#endif
