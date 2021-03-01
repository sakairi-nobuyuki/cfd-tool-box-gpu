#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NN 20


__global__
void sum_array (double *array_1, double *array_2, double *array_3, int n_array) {
    int i, j, n;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    n = 10;

    for (j = 0; j < n; j++) {
        //if (i < NN)  array_3[i] = array_1[i] + array_3[i];        
        array_3[i] = 2.0 * array_3[i];
    }
    
}

__global__
void derivertive_array (double *array_in, double *array_out, int n_array) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < NN - 1) array_out[i] = array_in[i] - array_in[i-1];
    if (i == 0) array_out[i] = array_in[i + 1];
    if (i == NN - 1) array_out[i] = array_in[i - 1];
}

__global__ void solve_diffusion_eq (double *array_in, double *array_out, double rdx2, double dt, int n_array) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (0 < i && i < NN - 1) array_out[i] = array_out[i] = array_in[i] + 0.5 * (array_in[i + 1] - 2.0 * array_in[i] + array_in[i - 1]) * rdx2 * dt;
    if (i == 0) array_out[i] = array_in[i + 1];
    if (i == NN - 1) array_out[i] = array_in[i - 1];
}

__global__ void renew_vers (double *array_in, double *array_out) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;
    array_out[i] = array_in[i];
}


void initialize_array (double *array, int size) {
    int i;

//    for (i = 0; i < NN; i++)  array[i] = (double) rand ();
    //for (i = 0; i < NN; i++)  array[i] = 1.0;
    //for (i = 0; i < NN; i++)  array[i] = (double) i;

    for (i = 0; i < NN; i++) {
        if (i < NN / 3 || 2 * NN/ 3 < i) array[i] = 0.0;
        else array[i] = 1.0;
    }

}

void print_result (double *array, int n) {
    int i;


    //for (i = 0; i < n; i++)  printf ("%.0lf ", array[i]);
    for (i = 0; i < n; i++)  printf ("%.3lf ", array[i]);
    printf ("\n");

}

int main () {
    int i, n;
    double *array_1, *array_2, *array_3;
    double *d_array_1, *d_array_2, *d_array_3;
    size_t n_bytes = NN * sizeof (double);
    time_t start_time, end_time;
    dim3 Grid, Block;

    Grid.x = NN / 196 + 1;
    Block.x = 196;

    printf ("start calc\n");
    start_time = time (NULL);

    array_1 = (double *) malloc (n_bytes);
    array_2 = (double *) malloc (n_bytes);
    array_3 = (double *) malloc (n_bytes);

    printf ("memory allocation finished\n");

    initialize_array (array_1, n_bytes);
    initialize_array (array_2, n_bytes);
    initialize_array (array_3, n_bytes);

    printf ("initialize memory\n");


    printf ("cuda memory allocation\n");

    cudaMalloc ((void**)&d_array_1, n_bytes);
    cudaMalloc ((void**)&d_array_2, n_bytes);
    cudaMalloc ((void**)&d_array_3, n_bytes);

    printf ("cuda memory allocation finished\n");


    printf ("cuda memory copy\n");

    cudaMemcpy (d_array_1, array_1, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy (d_array_2, array_2, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy (d_array_3, array_3, n_bytes, cudaMemcpyHostToDevice);

    printf ("cuda memory copy finished\n");

    printf ("inp array1\n");
    print_result (array_1, NN);
    //printf ("inp array2\n");
    //print_result (array_2, NN);
    //printf ("inp array3\n");
    //print_result (array_3, NN);

    printf ("start kernel function\n");


    //sum_array<<<Grid, Block>>> (d_array_1, d_array_2, d_array_3, n_bytes);
    //derivertive_array<<<Grid, Block>>> (d_array_1, d_array_3, n_bytes);

    for (n = 0; n < 1000000; n++) {
        if (n % 10000 == 0) {
            cudaMemcpy (array_1, d_array_1, n_bytes, cudaMemcpyDeviceToHost);
            print_result (array_1, NN);
        }
        solve_diffusion_eq <<<Grid, Block>>> (d_array_1, d_array_3, 0.0001, 0.0001, n_bytes);
        cudaDeviceSynchronize();
        renew_vers <<<Grid, Block>>> (d_array_3, d_array_1);
        cudaDeviceSynchronize();
        
    }


    cudaDeviceSynchronize();
    printf ("end kernel function\n");

    cudaMemcpy (array_3, d_array_3, n_bytes, cudaMemcpyDeviceToHost);

    printf ("res array3\n");
    print_result (array_3, NN);

    end_time = time (NULL);
    
    printf ("calc time: %ld\n", end_time - start_time);

    return 0;
}