#ifndef __MEMORY_CONFIG_H__
#define __MEMORY_CONFIG_H__

class MemoryConfig{
    // Args:
    //   int n_len_x, n_len_y:
    //     X and Y directional calculation mesh size.
    //   int n_pad_x, n_pad_y;
    //     X and Y directiobal padding mesh size to optimize memory access.
    // Returns:
    //   int n_bytes:
    //     Total memory allocation size. Product of X and Y directional memory allocation size.
    //     Fully optimized memory access, it must be a common multiple of n_optimal_memory_granylarity.
    // Attributes: 
    //   int _n_optimal_memory_granularity:
    //     CUDA optimized memory size to make aligned and coalesced memory access. Usually the value is 128 bytes.
    //
    // Definition:
    //   n_bytes is an integer such that n_bytes % n_optimal_memory_granylarity = 0.
    //   n_bytes = n_bytes_x * n_bytes_y.
    //   n_bytes_x = (n_len_x + n_pad_x) * sizeof(double)
    //   n_bytes_y = (n_len_y + n_pad_y) * sizeof(double)
    //   
    //   Where n_pad_x and n_pad_y are any integer that fills memory space 
    private:
        int _n_optimal_memory_granularity;
    public:
        MemoryConfig(int n_optimal_granularity_input = 128);
        
        int obtainBytes(int n_len_x, int n_len_y);
        int getGpuMemoryGranularity();

};


#endif