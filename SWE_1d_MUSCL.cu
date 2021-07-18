#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NN 20

void output_result (double *U, int n_array, int n_iter) {
    int i;
    FILE *fp_out;
    char file_name[64];

    sprintf (file_name, "%05d.dat", n_iter);
    printf ("output result to: %s\n", file_name);

    if ((fp_out = fopen (file_name, "w")) == NULL) {
        printf ("cannot open %s\n", file_name);
        exit (1);
    }

    for (i = 0; i < n_array; i++) fprintf (fp_out, "%d\t%lf\n", i, U[i]);

    fclose (fp_out);

}


__global__ void sum_array (double *array_1, double *array_2, double *array_3, int n_array) {
    int i, j, n;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    n = 10;

    for (j = 0; j < n; j++) {
        //if (i < NN)  array_3[i] = array_1[i] + array_3[i];        
        array_3[i] = 2.0 * array_3[i];
    }
    
}


__global__ void ones (double *U) {
    int i;
    i = blockIdx.x * blockDim.x + threadIdx.x;

    U[i] = i;
}


__global__ void derivertive_array (double *array_in, double *array_out, int n_array) {
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
    double *H, *Htmp, *HU, *HUtmp;   // SWE concerning vars in CPU
    double *U, *Utmp;                // supplementary vars in CPU
    double *gU, *gUtmp;              // SWE concerning vars in GPU
    size_t n_bytes = NN * sizeof (double);   // size of memory allocation for calculation area
    time_t start_time, end_time;
    dim3 Grid, Block;               //  number of grid and block in CUDA

    Grid.x = NN / 196 + 1;
    Block.x = 196;

    printf ("start calc\n");
    start_time = time (NULL);

    H    = (double *) malloc (n_bytes);
    Htmp = (double *) malloc (n_bytes);
    HU    = (double *) malloc (n_bytes);
    HUtmp = (double *) malloc (n_bytes);
    U    = (double *) malloc (n_bytes);
    Utmp = (double *) malloc (n_bytes);

    printf ("memory allocation finished\n");

    initialize_array (U, n_bytes);
    initialize_array (Utmp, n_bytes);

    printf ("initialize memory\n");
    printf ("cuda memory allocation\n");

    cudaMalloc ((void**) &gU,    n_bytes);
    cudaMalloc ((void**) &gUtmp, n_bytes);

    printf ("cuda memory allocation finished\n");
    printf ("cuda memory copy\n");

    cudaMemcpy (gU,    U,    n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy (gUtmp, Utmp, n_bytes, cudaMemcpyHostToDevice);

    printf ("cuda memory copy finished\n");

    printf ("inp array1\n");
    print_result (U, NN);

    printf ("start kernel function\n");
    //sum_array<<<Grid, Block>>> (d_array_1, d_array_2, d_array_3, n_bytes);
    derivertive_array<<<Grid, Block>>> (gU, gUtmp, n_bytes);
    cudaMemcpy (U, gU, n_bytes, cudaMemcpyDeviceToHost);
    //cudaMemcpy (Utmp, gUtmp, n_bytes, cudaMemcpyDeviceToHost);
    printf ("derivertice test\n");
    //print_result (Utmp, NN);
    print_result (U, NN);

    for (n = 0; n < 100000; n++) {
        if (n % 1000 == 0) {
            cudaMemcpy (U, gU, n_bytes, cudaMemcpyDeviceToHost);
            print_result (U, NN);
            output_result (U, NN, n);
        }
        solve_diffusion_eq <<<Grid, Block>>> (gU, gUtmp, 1.0, 0.1, n_bytes);
        cudaDeviceSynchronize();
        renew_vers <<<Grid, Block>>> (gUtmp, gU);
        cudaDeviceSynchronize();
     }

    cudaDeviceSynchronize();
    printf ("end kernel function\n");
    cudaMemcpy (U, gU, n_bytes, cudaMemcpyDeviceToHost);
    printf ("res array3\n");
    print_result (U, NN);

    end_time = time (NULL);
    
    printf ("calc time: %ld\n", end_time - start_time);

    return 0;
}