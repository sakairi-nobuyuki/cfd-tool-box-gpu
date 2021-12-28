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
void obtain_slope_device(double *Slope, double *DeltaPlus, double *DeltaMinus, double epsilon, GridDim *dimGrid, BlockDim *dimBlock, int n_len);
void obtain_cell_intface_value_device(double *R, double *L, double *Q, double *DeltaPlus, double *DeltaMinus, double *s, double kappa, GridDim *dimGrid, BlockDim *dimBlock, int n_len);
void obtain_cell_intface_value_from_Q_device(double *R, double *L, double *s, double *BarDeltaPlus, double *BarDeltaMinus, double *DeltaPlus, double *DeltaMinus, double *Q, double kappa, double epsilon, double b, GridDim *dimGrid, BlockDim *dimBlock, int n_len);
void test_solve_1d_conv_eq_device(double *Qtemp, double *Q, double *R, double *L, double dt, GridDim *dimGrid, BlockDim *dimBlock, int n_len);


// Shallow water eq. utilities
void create_h_flux_device(double *Flux, double *Q, GridDim *dimGrid, BlockDim *dimBlock, int n_len);
void create_q_flux_device(double *Flux, double *Q, double *H, double g, GridDim *dimGrid, BlockDim *dimBlock, int n_len);


// utilities
void print_var(double *U, int n_len);
void print_two_vars(double *U, double *V, int n_len);

void free_cuda_memory(double *gU);


