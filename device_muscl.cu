#include "device_muscl.h"

__device__
void obtain_delta_bar(double *U, double *DeltaAnonBar, int i_max) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < i_max - 2) {
        DeltaAnonBar[i] = U[i + 1] - U[i];
    }
}