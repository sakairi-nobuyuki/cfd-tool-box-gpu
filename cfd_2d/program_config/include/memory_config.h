#ifndef __MEMORY_CONFIG_H__
#define __MEMORY_CONFIG_H__

class MemoryConfig{
    private:
    public:
        // int n_len_x, n_len_y:
        //   X and Y directional calculation mesh size.
        // int n_pad_x, n_pad_y;
        //   X and Y directiobal padding mesh size to optimize memory access.
        // int n_bytes_x, n_bytes_y:
        //   X and Y directional memory allocation size.
        // int n_bytes:
        //   Total memory allocation size. Product of X and Y directional memory allocation size.
        //   Fully optimized memory access, it must be a common multiple of n_optimal_memory_granylarity.
        // int n_optimal_memory_granularity:
        //   CUDA optimized memory size to make aligned and coalesced memory access. Usually the value is 128 bytes.
        //
        // Definition:
        //   n_bytes is an integer such that n_bytes % n_optimal_memory_granylarity = 0.
        //   n_bytes = n_bytes_x * n_bytes_y.
        //   n_bytes_x = (n_len_x + n_pad_x) * sizeof(double)
        //   n_bytes_y = (n_len_y + n_pad_y) * sizeof(double)
        MemoryConfig();
        MemoryConfig(int n_len_x_inp, int n_len_y_inp, int n_optimal_granularity_input = 128);
        int n_bytes;
        int n_len_x, n_len_y;
        int n_pad_x, n_pad_y;  
        int n_optimal_memory_granularity;
      

};


#endif