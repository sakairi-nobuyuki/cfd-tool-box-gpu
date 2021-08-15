#include "cuda_memory_config.h"


//void allocate_cuda_memory(double *U, int n_bytes);
void allocate_cuda_memory(void **U, int n_bytes);
void copy_memory_host_to_device(double *gU, double *U, int n_bytes);
void copy_memory_device_to_host(double *U, double *gU, int n_bytes);
void copy_memory_device_to_device(double *gV, double *gU, int n_bytes);
void cuda_device_synchronize();

//void obtain_deltas_device(double *gU, double *gDeltaPlus, double *gDeltaMinus, int n_len);
void obtain_deltas_device(double *gDeltaPlus, double *gDeltaMinus, double *gU, GridDim *dimGrid, BlockDim *dimBlock, int n_len);

// MUSCL limiters
void obtain_minmod_device(double *gBarDeltaPlus, double *gBarDeltaMinus, double *gDeltaPlus, double *gDeltaMinus, double b, GridDim *dimGrid, BlockDim *dimBlock, int n_len);


void free_cuda_memory(double *gU);


void copy_memory_mock();