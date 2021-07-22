#ifndef __FIELD_VARS_CPP__
#define __FIELD_VARS_CPP__

#include <stdio.h>
#include <stdlib.h>

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
    allocate_cuda_memory(gArray, n_bytes);
    cArray = (double *) malloc(n_bytes);
    printf("  allocated CPU and GPU memory\n");

    // init by substituting zeros
    initVarsWithZero();
    printf("  initialize CPU and GPU memory\n");


}

void FieldVars1D::init_field_vars(int array_length, char var_name[64]) {
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
    
    // allocate memories Delta
    allocate_cuda_memory(gDeltaPlus, n_bytes);
    cDeltaPlus = (double *) malloc(n_bytes);
    
    allocate_cuda_memory(gDeltaMinus, n_bytes);
    cDeltaMinus = (double *) malloc(n_bytes);
    
    printf("  allocated CPU and GPU memory\n");

    testDeviceVarsAllocation();

    // init by substituting zeros
    initVarsWithZero();
    printf("  initialize CPU and GPU memory\n");

    
    
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

void FieldVars1D::compareResultCPUandGPU(double *ResCPU, double *ResGPU, int n_len) {
    int i;
    for (i = 0; i < n_len - 1; i++) {
        printf("res cpu: %lf, res gpu: %lf, diff: %lf\n", ResCPU[i], ResGPU[i], ResCPU[i] - ResGPU[i]);
    }
}

void FieldVars1D::testDeviceVarsAllocation() {
    initVarsWithHeavisiteFunc();
    copy_memory_host_to_device(gArray, cArray, n_bytes);

    cDeltaPlusTest  = (double *) malloc(n_bytes);
    cDeltaMinusTest = (double *) malloc(n_bytes);
    obtainDeltas();

    cuda_device_synchronize();
    obtain_deltas_device(gArray, gDeltaPlus, gDeltaMinus, n_len);
    cuda_device_synchronize();
    copy_memory_device_to_host(cDeltaPlusTest, gDeltaPlus, n_bytes);
    copy_memory_device_to_host(cDeltaMinusTest, gDeltaMinus, n_bytes);

    compareResultCPUandGPU(cDeltaPlus, cDeltaPlusTest, n_len);
    compareResultCPUandGPU(cDeltaMinus, cDeltaMinusTest, n_len);
}

void FieldVars1D::obtainDeltas() {
    int i;

    for (i = 0; i < n_len - 2; i++) cDeltaPlus[i] = cArray[i+1] - cArray[i];
    cDeltaPlus[n_len - 1] = cDeltaPlus[n_len - 2];
    for (i = 1; i < n_len - 1; i++) cDeltaMinus[i] = cArray[i] - cArray[i-1];
    cDeltaMinus[0] = cDeltaMinus[1];

}

#endif