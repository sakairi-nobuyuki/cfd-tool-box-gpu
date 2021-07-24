#ifndef __FIELD_VARS_CPP__
#define __FIELD_VARS_CPP__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "field_vars.h"
#include "cuda_cfd_kernel_funcs.h"

FieldVars1D::FieldVars1D() {}
FieldVars1D::FieldVars1D(int array_length, char var_name[64]) {
    // setting variable name
    sprintf(name, "%s", var_name);
    printf("initializing %s: \n", name);

    // obtain length of array
    n_len   = array_length;
    n_bytes = sizeof(double) * n_len;
    printf("  length of the array: %d\n", n_len);
    
    // allocate memories
    //allocate_cuda_memory(gArray, n_bytes);
    cArray = (double *) malloc(n_bytes);
    printf("  allocated CPU and GPU memory\n");

    // init by substituting zeros
    //initVarsWithZero();
    //printf("  initialize CPU and GPU memory\n");
}

double FieldVars1D::testObtainCorrelFactor(double *U, double *V, int n_len) {
    double Uave = 0.0, Vave = 0.0, UU = 0.0, UV = 0.0, VV = 0.0;
    int i;

    for (i = 0; i < n_len; i++) Uave += U[i];
    Uave /= (double) n_len;
    for (i = 0; i < n_len; i++) Vave += V[i];
    Vave /= (double) n_len;

    for (i = 0; i < n_len; i++) UU += pow(U[i] - Uave, 2);
    for (i = 0; i < n_len; i++) VV += pow(V[i] - Vave, 2);
    for (i = 0; i < n_len; i++) UV += (V[i] - Vave) * (U[i] - Uave);

    return UV / sqrt(UU * VV);

}

void FieldVars1D::testMemory() {
    int n_failure;
    double Cor;

    printf("  Memory test of %s:\n", name);
    cArrayMemoryTest = (double *) malloc(n_bytes);
    allocate_cuda_memory((void **) &gArrayMemoryTest, n_bytes);
    initWithHeavisiteFunc(cArray, n_len);
    
    printf("    Simple memory copy test\n");
    copy_memory_host_to_device(gArray, cArray, n_bytes);
    copy_memory_device_to_device(gArrayMemoryTest, gArray, n_bytes);
    copy_memory_device_to_host(cArrayMemoryTest, gArrayMemoryTest, n_bytes);

    printf ("    validation of contents of 2 memory areas:\n");
    n_failure = compareArrays(cArray, cArrayMemoryTest, n_len);
    printf ("      %d data was different out of %d\n", n_failure, n_len);
    Cor = testObtainCorrelFactor(cArray, cArrayMemoryTest, n_len);
    printf ("      Correlation factor between 2 memories: %lf\n", Cor);



//cuda_device_synchronize();
    //obtain_deltas_device(gArray, gDeltaPlus, gDeltaMinus, n_len);
    //cuda_device_synchronize();
    //copy_memory_device_to_host(cDeltaPlusTest, gDeltaPlus, n_bytes);
    //copy_memory_device_to_host(cDeltaMinusTest, gDeltaMinus, n_bytes);


}

void FieldVars1D::initFieldVars(int array_length, char var_name[64]) {
    // setting variable name
    sprintf(name, "%s", var_name);
    printf("initializing %s: \n", name);

    // obtain length of array
    n_len   = array_length;
    n_bytes = sizeof(double) * n_len;
    printf("  length of the array: %d\n", n_len);
    printf("  size of the array:   %d\n", n_bytes);
    
    // allocate memories
    printf("  In field_vars, allocating GPU memory\n");
    printf("    before allocate_cuda_memory address of gArray: %p\n", gArray);
    //allocate_cuda_memory(gArray, n_bytes);
    allocate_cuda_memory((void **) &gArray, n_bytes);
    printf("    after allocate_cuda_memory address of gArray:  %p\n", gArray);
    cArray = (double *) malloc(n_bytes);
    
    // memory test
    testMemory();

}


FieldVars1D::~FieldVars1D() {
    free(cArray);
    free_cuda_memory(gArray);
    

}


void FieldVars1D::initVarsWithZero() {
    int i;

    for(i = 0; i < n_len; i++) cArray[i] = 0.0;
    copy_memory_host_to_device(gArray, cArray, n_bytes);
}

void FieldVars1D::initVarsWithHeavisiteFunc() {
    int i;

    for(i = 0; i < n_len; i++) {
        if (i < n_len/2) {
            cArray[i] = 1.0;
        } else {
            cArray[i] = 0.0;
        }
    }
    copy_memory_host_to_device(gArray, cArray, n_bytes);
}

void FieldVars1D::initWithHeavisiteFunc(double *Array, int n_len) {
    int i;

    for(i = 0; i < n_len; i++) {
        if (i < n_len/2) {
            Array[i] = 1.0;
        } else {
            Array[i] = 0.0;
        }
    }
}

int FieldVars1D::compareArrays(double *U, double *V, int n_len) {
    int i, n_failure = 0;
    
    for (i = 0; i < n_len - 1; i++) {
        if (U[i] != V[i]) n_failure++;
    }

    return n_failure;
}
    //cDeltaPlusTest  = (double *) malloc(n_bytes);
    //cDeltaMinusTest = (double *) malloc(n_bytes);
    //obtainDeltas();

    //cuda_device_synchronize();
    //obtain_deltas_device(gArray, gDeltaPlus, gDeltaMinus, n_len);
    //cuda_device_synchronize();
    //copy_memory_device_to_host(cDeltaPlusTest, gDeltaPlus, n_bytes);
    //copy_memory_device_to_host(cDeltaMinusTest, gDeltaMinus, n_bytes);

    //compareResultCPUandGPU(cDeltaPlus, cDeltaPlusTest, n_len);
    //compareResultCPUandGPU(cDeltaMinus, cDeltaMinusTest, n_len);

void FieldVars1D::obtainDeltas() {
    int i;

    for (i = 0; i < n_len - 2; i++) cDeltaPlus[i] = cArray[i+1] - cArray[i];
    cDeltaPlus[n_len - 1] = cDeltaPlus[n_len - 2];
    for (i = 1; i < n_len - 1; i++) cDeltaMinus[i] = cArray[i] - cArray[i-1];
    cDeltaMinus[0] = cDeltaMinus[1];

}

#endif