#ifndef __FIELD_VARS_CPP__
#define __FIELD_VARS_CPP__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "field_vars.h"
#include "cuda_cfd_kernel_funcs.h"

FieldVars1D::FieldVars1D() {}
//void FieldVars1D::initFieldVars(int array_length, char var_name[64]) {
void FieldVars1D::initFieldVars(int array_length, char var_name[64], GridDim *dimGridInp, BlockDim *dimBlockInp) {
    // setting variable name
    sprintf(name, "%s", var_name);
    printf("initializing FieldVars1D of %s: \n", name);

    // obtain length of array
    n_len   = array_length;
    n_bytes = sizeof(double) * n_len;
    printf("  length of the array: %d\n", n_len);
    printf("  size of the array:   %d\n", n_bytes);
    
    // setting CUDA memory config
    dimBlock = dimBlockInp;
    dimGrid  = dimGridInp;
    printf("  configuration of CUDA memory: dim3 grid(%d, %d), block(%d, %d, %d)\n", dimGrid->x, dimGrid->y, dimBlock->x, dimBlock->y, dimBlock->z);

    // setting debug mode on/off
    debug_mode = -1; 

    // allocate memories
    printf("  In field_vars, allocating GPU memory\n");
    //allocate_cuda_memory(gArray, n_bytes);
    allocate_cuda_memory((void **) &gArray,      n_bytes);
    allocate_cuda_memory((void **) &gDeltaPlus,  n_bytes);
    allocate_cuda_memory((void **) &gDeltaMinus, n_bytes);
    cArray =      (double *) malloc(n_bytes);
    cDeltaPlus  = (double *) malloc(n_bytes);
    cDeltaMinus = (double *) malloc(n_bytes);

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


int FieldVars1D::compareArrays(double *U, double *V, int n_len) {
    // count number of elements of U and V that differs each other 
    int i, n_failure = 0;
    
    for (i = 0; i < n_len - 1; i++) {
        if (U[i] != V[i]) n_failure++;
    }

    return n_failure;
}
    
void FieldVars1D::obtainDeltas() {
    int i;

    for (i = 0; i < n_len - 1; i++) cDeltaPlus[i] = cArray[i+1] - cArray[i];
    cDeltaPlus[n_len - 1] = cDeltaPlus[n_len - 2];
    for (i = 1; i < n_len; i++) cDeltaMinus[i] = cArray[i] - cArray[i-1];
    cDeltaMinus[0] = cDeltaMinus[1];

}


void initArrayWithHeavisiteFunc(double *Array, int n_len) {
    //  initialize the Array by Heavisite step function
    int i;

    for(i = 0; i < n_len; i++) {
        if (i < n_len/2) {
            Array[i] = 1.0;
        } else {
            Array[i] = 0.0;
        }
    }

}


void initArrayWithRampFunc(double *Array, int n_len) {
    //  initialize the Array by Ramp function
    int i;

    for(i = 0; i < n_len; i++) {
        Array[i] = 0.3 * (double) i;
    }
}

void initArrayWithParaboraicFunc(double *Array, int n_len) {
    //  initialize the Array by Ramp function
    int i;

    for(i = 0; i < n_len; i++) {
        Array[i] = 0.3 * pow((double) i, 2);
    }
}

double FieldVars1D::testObtainCorrelFactor(double *U, double *V, int n_len) {
    //  obtain correlation factor of U and V in a test sequence
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
    printf("    in memory test, configuration of CUDA memory: dim3 grid(%d, %d), block(%d, %d, %d)\n", dimGrid->x, dimGrid->y, dimBlock->x, dimBlock->y, dimBlock->z);
    cArrayMemoryTest = (double *) malloc(n_bytes);
    allocate_cuda_memory((void **) &gArrayMemoryTest, n_bytes);
    initArrayWithHeavisiteFunc(cArray, n_len);
    
    //  Memory copy test 
    //  copy memory: host --> device, device --> device; device --> host
    //  validation by FN rate and correlation factor. I'm not sure if this validation is suitable or not, but I believe it works well.
    printf("    Simple memory copy test\n");
    copy_memory_host_to_device(gArray, cArray, n_bytes);
    copy_memory_device_to_device(gArrayMemoryTest, gArray, n_bytes);
    copy_memory_device_to_host(cArrayMemoryTest, gArrayMemoryTest, n_bytes);

    printf ("    validation of contents of 2 memory areas:\n");
    n_failure = compareArrays(cArray, cArrayMemoryTest, n_len);
    printf ("      %d data was different out of %d\n", n_failure, n_len);
    Cor = testObtainCorrelFactor(cArray, cArrayMemoryTest, n_len);
    printf ("      Correlation factor between 2 memories: %lf\n", Cor);
    copy_memory_device_to_host(cArrayMemoryTest, gArray, n_bytes);
    if (debug_mode == 0) printTwoVars(cArray, cArrayMemoryTest, n_len);

    //  Test of derivertive calculation
    //  calculating up-wind and down-wind derivertive by GPU and CPU
    printf("    Derivertive difference validation between CPU and GPU\n");
    cDeltaPlusTest  = (double *) malloc(n_bytes);  //  temporally memory allocation for test
    cDeltaMinusTest = (double *) malloc(n_bytes);  //  temporally memory allocation for test

    //  Tesing obtaining Delta+ and Delta- with a variation of initial conditions
    testObtainDeltasAbstract(initArrayWithHeavisiteFunc, gArray, gDeltaPlus, gDeltaMinus, 
        cArray, cDeltaPlusTest, cDeltaMinusTest, n_len, n_bytes, "Heavisite step function");
    testObtainDeltasAbstract(initArrayWithRampFunc, gArray, gDeltaPlus, gDeltaMinus, 
        cArray, cDeltaPlusTest, cDeltaMinusTest, n_len, n_bytes, "Ramp function");        
    testObtainDeltasAbstract(initArrayWithParaboraicFunc, gArray, gDeltaPlus, gDeltaMinus, 
        cArray, cDeltaPlusTest, cDeltaMinusTest, n_len, n_bytes, "paraboraic function");
    
    free(cDeltaPlusTest);
    free(cDeltaMinusTest);
    free(cArrayMemoryTest);
    free_cuda_memory(gArrayMemoryTest);
    printf("    test of %s is completed\n\n", name);

}

//void FieldVars1D::testObtainDeltasAbstract(void (FieldVars1D::*initArray) (double *cArray, int n_len), double *gArray, double *gDeltaPlus, double *gDeltaMinus, 
void FieldVars1D::testObtainDeltasAbstract(void (*initArray) (double *cArray, int n_len), double *gArray, double *gDeltaPlus, double *gDeltaMinus, 
        double *cArray, double *cDeltaPlusTest, double *cDeltaMinusTest, int n_len, int n_bytes, char test_name[64]) {
    double Cor;
    int n_failure;
    //initWithHeavisiteFunc(cArray, n_len);
    //(this->*initArray)(cArray, n_len);

    //debug_mode = 0;
    debug_mode = -1;

    initArray(cArray, n_len);
    copy_memory_host_to_device(gArray, cArray, n_bytes);

    cuda_device_synchronize();
    obtain_deltas_device(gArray, gDeltaPlus, gDeltaMinus, n_len);
    cuda_device_synchronize();
    copy_memory_device_to_host(cDeltaPlusTest, gDeltaPlus, n_bytes);
    copy_memory_device_to_host(cDeltaMinusTest, gDeltaMinus, n_bytes);

    obtainDeltas();

    printf ("  validation of calculation of Deltas+ with initial condition of %s:\n", test_name);
    printf ("    validation of contents of 2 memory areas of Delta+:\n");
    n_failure = compareArrays(cDeltaPlus, cDeltaPlusTest, n_len);
    printf ("      n_failures: %d / %d\n", n_failure, n_len);
    Cor = testObtainCorrelFactor(cDeltaPlus, cDeltaPlusTest, n_len);
    printf ("      Correlation factor: %lf\n", Cor);
    if (debug_mode == 0) printTwoVars(cDeltaPlus, cDeltaPlusTest, n_len);
    printf ("    validation of contents of 2 memory areas of Delta-:\n");
    n_failure = compareArrays(cDeltaMinus, cDeltaMinusTest, n_len);
    printf ("      n_failures: %d / %d\n", n_failure, n_len);
    Cor = testObtainCorrelFactor(cDeltaMinus, cDeltaMinusTest, n_len);
    printf ("      Correlation factor: %lf\n", Cor);
    
    if (debug_mode == 0) printTwoVars(cDeltaMinus, cDeltaMinusTest, n_len);
    printf ("    original values before derivertive\n");
 


}


void FieldVars1D::printTwoVars(double *U, double *V, int n_len) {
    //  print the elements of U and V to std output
    int i;

    printf("  confirming two vars:\n");
    for (i = 0; i < n_len; i++)  printf("    U: %lf, V: %lf\n", U[i], V[i]);
    printf("\n");

}


#endif