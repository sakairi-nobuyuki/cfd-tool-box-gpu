#include <memory_config.h>
          


MemoryConfig::MemoryConfig(int n_optimal_granularity_input){
    _n_optimal_memory_granularity = n_optimal_granularity_input;
}


int MemoryConfig::obtainBytes(int n_len_x, int n_len_y){
    
    return (1 + sizeof(double) * n_len_x * n_len_y / _n_optimal_memory_granularity) * _n_optimal_memory_granularity;

}

int MemoryConfig::getGpuMemoryGranularity(){
    return _n_optimal_memory_granularity;
}