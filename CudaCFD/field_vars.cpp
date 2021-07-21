#ifndef __FIELD_VARS__
#define __FIELD_VARS__

#include <stdio.h>
#include <stdlib.h>

#include "field_vars.h"
#include "cuda_front_end.h"


FieldVars1D::FieldVars1D(int array_length, char var_name[64]) {
    // setting variable name
    sprintf(name, "%s", var_name);
    printf("initializing %s: \n", name);

    // obtain length of array
    n_len   = array_length;
    n_bytes = sizeof(double) * n_len;
    printf("  length of the array: %d\n", n_len);
    
    // allocate memories
    allocate_cuda_memory(gArray, n_bytes);
    cArray = (double *) malloc(n_bytes);
    printf("  allocated CPU and GPU memory\n");

    // init by substituting zeros
    initVarsWithZero();
    printf("  initialize CPU and GPU memory\n");
}

void FieldVars1D::initVarsWithZero() {
    int i;

    for(i = 0; i < n_len; i++) cArray[i] = 0.0;
    copy_memory_host_to_device(gArray, cArray, n_bytes);
}



#endif