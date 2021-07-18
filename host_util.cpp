#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "host_util.h"


void output_result (double *U, int n_array, int n_iter) {
    int i;
    FILE *fp_out;
    char file_name[64];

    sprintf (file_name, "gpu_%05d.dat", n_iter);
    printf ("output result to: %s\n", file_name);

    if ((fp_out = fopen (file_name, "w")) == NULL) {
        printf ("cannot open %s\n", file_name);
        exit (1);
    }

    for (i = 0; i < n_array; i++) fprintf (fp_out, "%d\t%lf\n", i, U[i]);

    fclose (fp_out);

}


void initialize_array_with_zero (double *array, int size) {
    int i;


    for (i = 0; i < size; i++) {
        array[i] = 0.0;
    }

}
