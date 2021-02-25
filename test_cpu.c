#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NN 2000000


void sum_array (double *array_1, double *array_2, double *array_3, int n_array) {
    int i, j, n;
    n = 10000;

    for (j = 0; j < n; j++) {
        for (i = 0; i < NN; i++)  array_3[i] = array_1[i] + array_3[i];        
    }
    
}

void initialize_array (double *array, int size) {
    int i;

//    for (i = 0; i < NN; i++)  array[i] = (double) rand ();
    for (i = 0; i < NN; i++)  array[i] = 1.0;

}

void print_result (double *array, int n) {
    int i;


    for (i = 0; i < n; i++)  printf ("%.0lf ", array[i]);
    printf ("\n");

}

int main () {
    double *array_1, *array_2, *array_3;
    double *d_array_1, *d_array_2, *d_array_3;
    size_t n_bytes = NN * sizeof (double);
    time_t start_time, end_time;
    
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

    printf ("start calc\n");

    sum_array (array_1, array_2, array_3, n_bytes);

    //printf ("res array3\n");
    //print_result (array_3, NN);

    end_time = time (NULL);
    
    printf ("calc time: %ld\n", end_time - start_time);

    return 0;
}