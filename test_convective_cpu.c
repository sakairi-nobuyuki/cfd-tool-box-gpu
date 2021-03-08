#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define NN 200


void output_result (double *U, int n_array, int n_iter) {
    int i;
    FILE *fp_out;
    char file_name[64];

    sprintf (file_name, "cpu_%05d.dat", n_iter);
    printf ("output result to: %s\n", file_name);

    if ((fp_out = fopen (file_name, "w")) == NULL) {
        printf ("cannot open %s\n", file_name);
        exit (1);
    }

    for (i = 0; i < n_array; i++) fprintf (fp_out, "%d\t%lf\n", i, U[i]);

    fclose (fp_out);

}




void renew_vers (double *array_in, double *array_out) {
    int i;
    for (i = 0; i < NN; i++)
        array_out[i] = array_in[i];
}

void solve_convective_1o_up_wind (double *U, double *U_tmp, int n_len, double C, double Co) {
    int i;
    
    ///  boundary condition
    U_tmp[0] = U[1];
    U_tmp[NN - 1] = U[NN - 2];

    for (i = 0; i < NN; i++) 
        U_tmp[i] = U[i] - 0.5 * Co * (C * (U[i + 1] - U[i-1]) - fabs (C) * (U[i + 1] - 2.0 * U[i] + U[i - 1]));
    //if (0 < i && i < NN - 1) U_tmp[i] = U[i] + 0.5 * Co * (C * (U[i + 1] - U[i - 1]));
    //if (0 < i && i < NN - 1) U_tmp[i] = U[i] - 0.5 * Co * (C * (U[i + 1] - 2.0 *  U[i - 1] + U[i - 1]));
    //if (0 < i && i < NN - 1) U_tmp[i] = U[i] - U[i - 1];
    //if (i == 0)      U_tmp[i] = U[i + 1];
    //if (i == NN - 1) U_tmp[i] = U[i - 1];
}


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
    //for (i = 0; i < NN; i++)  array[i] = 1.0;
    for (i = 0; i < NN; i++) {
        if (i < NN / 3 || 2 * NN/ 3 < i) array[i] = 0.0;
        else array[i] = 1.0;
    }

}

void print_result (double *array, int n) {
    int i;


    for (i = 0; i < n; i++)  printf ("%.0lf ", array[i]);
    printf ("\n");

}

int main () {
    int n;
    double *array_1, *array_2, *array_3;
    double *d_array_1, *d_array_2, *d_array_3;
    double *U, *Utmp;
    double C, Co;
    size_t n_bytes = NN * sizeof (double);
    time_t start_time, end_time;
    
    printf ("start calc\n");
    start_time = time (NULL);

    U = (double *) malloc (n_bytes);
    Utmp = (double *) malloc (n_bytes);
    //array_1 = (double *) malloc (n_bytes);
    //array_2 = (double *) malloc (n_bytes);
    //array_3 = (double *) malloc (n_bytes);

    printf ("memory allocation finished\n");

    //initialize_array (array_1, n_bytes);
    initialize_array (U, n_bytes);
    initialize_array (Utmp, n_bytes);

    printf ("initialize memory\n");

    printf ("start calc\n");

    Co = 0.01; 
    C  = 0.1;
    for (n = 0; n < 100000; n++) {
        if (n % 1000 == 0) {
            print_result (U, NN);
            output_result (U, NN, n);
        }
        solve_convective_1o_up_wind (U, Utmp, NN, C, Co);
        renew_vers (Utmp, U);
        //solve_diffusion_eq <<<Grid, Block>>> (gU, gUtmp, 0.1, 0.1, n_bytes);

     }
    //sum_array (array_1, array_2, array_3, n_bytes);

    //printf ("res array3\n");
    //print_result (array_3, NN);

    end_time = time (NULL);
    
    printf ("calc time: %ld\n", end_time - start_time);

    return 0;
}