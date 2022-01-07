#ifndef __MEMORY_CONFIG_CPP__
#define __MEMORY_CONFIG_CPP__

#include "memory_config.h"
          

MemoryConfig::MemoryConfig(){


}

MemoryConfig::MemoryConfig(int n_len_x_inp, int n_len_y_inp){
    n_len_x = n_len_x_inp;
    n_len_y = n_len_y_inp;

    n_optimal_memory_granularity = 128;
}

MemoryConfig::MemoryConfig(int n_len_x_inp, int n_len_y_inp, int n_optimal_granularity_input){
    n_len_x = n_len_x_inp;
    n_len_y = n_len_y_inp;

    n_optimal_memory_granularity = 128;
}


#endif