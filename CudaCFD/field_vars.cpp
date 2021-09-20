#ifndef __FIELD_VARS_CPP__
#define __FIELD_VARS_CPP__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

    // setting CFD something
    kappa = 1.0 / 3.0;
    b = (3.0 - kappa) / (1.0 - kappa);
    epsilon = 0.01;

    // setting debug mode on/off
    debug_mode = 0; 
    //debug_mode = -1; 

    // allocate memories
    printf("  In %s, allocating GPU memory\n", name);
    //allocate_cuda_memory(gArray, n_bytes);
    allocate_cuda_memory((void **) &gArray,      n_bytes);
    allocate_cuda_memory((void **) &gDeltaPlus,  n_bytes);
    allocate_cuda_memory((void **) &gDeltaMinus, n_bytes);
    allocate_cuda_memory((void **) &gBarDeltaPlus,  n_bytes);
    allocate_cuda_memory((void **) &gBarDeltaMinus, n_bytes);
    allocate_cuda_memory((void **) &gSlope, n_bytes);
    allocate_cuda_memory((void **) &gLeft, n_bytes);
    allocate_cuda_memory((void **) &gRight, n_bytes);
    allocate_cuda_memory((void **) &gArrayTemp, n_bytes);
    cArray =      (double *) malloc(n_bytes);
    cDeltaPlus  = (double *) malloc(n_bytes);
    cDeltaMinus = (double *) malloc(n_bytes);
    cBarDeltaPlus  = (double *) malloc(n_bytes);
    cBarDeltaMinus = (double *) malloc(n_bytes);
    cSlope = (double *) malloc(n_bytes);
    cLeft  = (double *) malloc(n_bytes);
    cRight = (double *) malloc(n_bytes);
    cArrayTemp = (double *) malloc (n_bytes);

    // memory and other calculation test
    testMemoryDerivertiveLimiterAndFlux();
    if (debug_mode == 0) testSolveConvectiveEq();
    printf("finish initializing FieldVars1D of %s: \n", name);
}


FieldVars1D::~FieldVars1D() {
    free(cArray);
    free(cDeltaPlus);
    free(cDeltaMinus);
    free(cBarDeltaPlus);
    free(cBarDeltaMinus);
    free(cSlope);
    free(cLeft);
    free(cRight);
    free_cuda_memory(gArray);
    free_cuda_memory(gDeltaPlus);
    free_cuda_memory(gDeltaMinus);
    free_cuda_memory(gBarDeltaPlus);
    free_cuda_memory(gBarDeltaMinus);
    free_cuda_memory(gSlope);
    free_cuda_memory(gLeft);
    free_cuda_memory(gRight);
}

void FieldVars1D::initAuxFieldVars(int array_length, char var_name[64], GridDim *dimGridInp, BlockDim *dimBlockInp) {
    // setting variable name
    sprintf(name, "%s", var_name);
    printf("initializing Aux. FieldVars1D of %s: \n", name);

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
    debug_mode = 0; 
    //debug_mode = -1; 

    printf("  In aux. field_vars, allocating GPU memory\n");
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


void FieldVars1D::obtainMinmod() {
    int i;

    for (i = 0; i < n_len; i++) {
        cBarDeltaPlus[i] 
            = copysignf(1.0, cDeltaPlus[i]) 
            * fmaxf(0.0, fminf(fabs(cDeltaPlus[i]), copysignf(1.0, cDeltaPlus[i]) * b * cDeltaMinus[i]));
    }

    for (i = 0; i < n_len; i++) {
        cBarDeltaMinus[i] 
            = copysignf(1.0, cDeltaMinus[i]) 
            * fmaxf(0.0, fminf(fabs(cDeltaMinus[i]), copysignf(1.0, cDeltaMinus[i]) * b * cDeltaPlus[i]));
    }

}


void FieldVars1D::obtainSlope() {
    int i;

    for (i = 0; i < n_len; i++) {
        cSlope[i] = (2.0 * cBarDeltaPlus[i] * cBarDeltaMinus[i] + epsilon) 
        / (pow(cBarDeltaPlus[i], 2.0) + pow(cBarDeltaMinus[i], 2.0) + epsilon);
    }
}


void FieldVars1D::obtainCellIntfaceValue() {
    int i;

    for (i = 0; i < n_len - 1; i++) {
        cLeft[i] = cArray[i] + 0.25 * cSlope[i] * ((1.0 - kappa * cSlope[i]) * cBarDeltaMinus[i] + (1.0 + kappa * cSlope[i]) * cBarDeltaPlus[i]);
    }
    cLeft[n_len - 1] = cLeft[n_len - 2];

    for (i = 0; i < n_len - 1; i++) {
        cRight[i] = cArray[i+1]  - 0.25 * cSlope[i+1] * ((1.0 - kappa * cSlope[i+1]) * cBarDeltaPlus[i+1] + (1.0 + kappa * cSlope[i+1]) * cBarDeltaMinus[i+1]);
    }
    cRight[0] = cRight[1];
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

    if (UU == 0.0 || VV == 0.0) {
        if (UV == 0.0) return 1.0;
        else return 0.0;
    } else {
        return UV / sqrt(UU * VV);
    }

}

int FieldVars1D::testSolveConvectiveEq() {
    // solving 1 dimensional convective equation of which phase speed is 1 and dt = 0.1 and dx = 1 for testing

    double dt = 0.1, Cor;
    int i, n, N = 10, n_failures;
    time_t start_time, end_time;

    cArrayMemoryTest = (double *) malloc(n_bytes);

    // set initial condition
    initArrayWithHeavisiteFunc(cArray, n_len);
    // memory copy to GPU
    copy_memory_host_to_device(gArray, cArray, n_bytes);

    
    // CPU calculation
    start_time = time (NULL);
    for (n = 0; n < N; n++) {
        obtainDeltas();
        obtainMinmod();
        obtainSlope();
        obtainCellIntfaceValue();
        for (i = 1; i < n_len - 1; i++) 
            cArrayTemp[i] = cArray[i] - dt * (cLeft[i] - cLeft[i - 1]);
        renewValues();
        setNeumannBoundaryCondition();
        
    }
    end_time = time (NULL);
    printf ("  CPU calc time: %ld\n", end_time - start_time);

    // GPU calculation
    start_time = time (NULL);
    for (n = 0; n < N; n++) {
        
        obtain_cell_intface_value_from_Q_device(gRight, gLeft, gSlope, gBarDeltaPlus, gBarDeltaMinus, gDeltaPlus, gDeltaMinus, gArray, 
            kappa, epsilon, b, dimGrid, dimBlock, n_len);
        test_solve_1d_conv_eq_device(gArrayTemp, gArray, gRight, gLeft, dt, dimGrid, dimBlock, n_len);
        copy_memory_device_to_host(cArrayMemoryTest, gArray, n_bytes);
        //printOneVar(cArrayMemoryTest, n_len);
        //printf("  %d th iteration end.", n);
    }
    end_time = time (NULL);

    printf ("  GPU calc time: %ld\n", end_time - start_time);
    copy_memory_device_to_host(cArrayMemoryTest, gArray, n_bytes);
    n_failures = compareArrays(cArray, cArrayMemoryTest, n_len);
    
    printf ("  Validation of 1-dimensional convetive eq of %s:\n", name);
    printf ("      Correlation factor between 2 memories: %lf\n", Cor);
    validateTwoVarsByCorrelationFactor(cArray, cArrayMemoryTest, "CPU", "GPU", 0.5, n_len);
    if (debug_mode == 0) printTwoVars(cArray, cArrayMemoryTest, n_len);

    free(cArrayMemoryTest);
}

void FieldVars1D::setNeumannBoundaryCondition() {
    cArray[2] = cArrayTemp[3];
    cArray[1] = cArrayTemp[3];
    cArray[0] = cArrayTemp[3];
    cArray[n_len - 1] = cArrayTemp[n_len - 4];
    cArray[n_len - 2] = cArrayTemp[n_len - 4];
    cArray[n_len - 3] = cArrayTemp[n_len - 4];
}

void FieldVars1D::renewValues() {
    int i;
    for (i = 0; i < n_len; i++) cArray[i] = cArrayTemp[i];
}

int FieldVars1D::testMemoryCopy(double *cArray, double *gArray, double *gArrayMemoryTest, double *cArrayMemoryTest, char var_name[64]) {
    int n_failures;
    double Cor;

    printf("  Testing %s memory copy:\n", var_name);
    copy_memory_host_to_device(gArray, cArray, n_bytes);
    copy_memory_device_to_device(gArrayMemoryTest, gArray, n_bytes);
    copy_memory_device_to_host(cArrayMemoryTest, gArrayMemoryTest, n_bytes);

    printf ("    validation of contents of 2 memory areas:\n");
    n_failures = compareArrays(cArray, cArrayMemoryTest, n_len);
    printf ("      %d data was different out of %d\n", n_failures, n_len);
    Cor = testObtainCorrelFactor(cArray, cArrayMemoryTest, n_len);
    printf ("      Correlation factor between 2 memories: %lf\n", Cor);
    copy_memory_device_to_host(cArrayMemoryTest, gArray, n_bytes);
    if (debug_mode == 0) printTwoVars(cArray, cArrayMemoryTest, n_len);

    if (n_failures / n_len > 0.5 || Cor < 0.5) {
        printf("    Too much inconsisntency at memory copy test of %s\n", var_name);
        return -1;
    } else {
        printf("    Memory test of %s succeeded\n\n", var_name);
        return 0;
    }
    
}

int FieldVars1D::testDerivertive() {
    int n_failure = 0;
    printf("    Testing %s delta plus and delta minus for minmod:\n", name);
    n_failure += testObtainDeltasMinmodAbstract(initArrayWithHeavisiteFunc, gArray, gDeltaPlus, gDeltaMinus, 
        cArray, cDeltaPlusTest, cDeltaMinusTest, n_len, n_bytes, "Heavisite step function");
    n_failure += testObtainDeltasMinmodAbstract(initArrayWithRampFunc, gArray, gDeltaPlus, gDeltaMinus, 
        cArray, cDeltaPlusTest, cDeltaMinusTest, n_len, n_bytes, "Ramp function");        
    n_failure += testObtainDeltasMinmodAbstract(initArrayWithParaboraicFunc, gArray, gDeltaPlus, gDeltaMinus, 
        cArray, cDeltaPlusTest, cDeltaMinusTest, n_len, n_bytes, "paraboraic function");

    return n_failure;
}

void FieldVars1D::testMemoryDerivertiveLimiterAndFlux() {
    int n_failure = 0;
    //double Cor;

    printf("  Memory test of %s:\n", name);
    printf("    in memory test, configuration of CUDA memory: dim3 grid(%d, %d), block(%d, %d, %d)\n", dimGrid->x, dimGrid->y, dimBlock->x, dimBlock->y, dimBlock->z);
    cArrayMemoryTest = (double *) malloc(n_bytes);
    allocate_cuda_memory((void **) &gArrayMemoryTest, n_bytes);
    initArrayWithHeavisiteFunc(cArray, n_len);
    
    //  Memory copy test 
    //  copy memory: host --> device, device --> device; device --> host
    //  validation by FN rate and correlation factor. I'm not sure if this validation is suitable or not, but I believe it works well.
    printf("    Simple memory copy test\n");
    n_failure += testMemoryCopy(cArray, gArray, gArrayMemoryTest, cArrayMemoryTest, "gArray");
    n_failure += testMemoryCopy(cArray, gDeltaPlus,  gArrayMemoryTest, cArrayMemoryTest, "gDeltaPlus");
    n_failure += testMemoryCopy(cArray, gDeltaMinus, gArrayMemoryTest, cArrayMemoryTest, "gDeltaMinus");
    n_failure += testMemoryCopy(cArray, gBarDeltaPlus,  gArrayMemoryTest, cArrayMemoryTest, "gDeltaPlusBar");
    n_failure += testMemoryCopy(cArray, gBarDeltaMinus, gArrayMemoryTest, cArrayMemoryTest, "gDeltaMinusBar");


    //  Test of derivertive calculation
    //  calculating up-wind and down-wind derivertive by GPU and CPU
    printf("    Derivertive difference validation between CPU and GPU\n");
    cDeltaPlusTest  = (double *) malloc(n_bytes);  //  temporally memory allocation for test
    cDeltaMinusTest = (double *) malloc(n_bytes);  //  temporally memory allocation for test

    //  Tesing obtaining Delta+ and Delta- with a variation of initial conditions
    printf("    Testing %s delta plus and delta minus for minmod:\n", name);
    n_failure += testDerivertive();
    
    free(cDeltaPlusTest);
    free(cDeltaMinusTest);
    free(cArrayMemoryTest);
    free_cuda_memory(gArrayMemoryTest);
    printf("    test of %s is completed\n", name);
    printf("    number of failures in %s test was %d\n", name, n_failure);
    if (n_failure > 1) {
        printf("TOO MUCH FAILURES AT MEMORY AND CALC TEST. GOING TO ABORT. CONFIRM CUDA SOMETHING OR CODE\n");
        exit(1);
    }

}

//void FieldVars1D::testObtainDeltasAbstract(void (FieldVars1D::*initArray) (double *cArray, int n_len), double *gArray, double *gDeltaPlus, double *gDeltaMinus, 
int FieldVars1D::testObtainDeltasMinmodAbstract(void (*initArray) (double *cArray, int n_len), double *gArray, double *gDeltaPlus, double *gDeltaMinus, 
        double *cArray, double *cDeltaPlusTest, double *cDeltaMinusTest, int n_len, int n_bytes, char test_name[64]) {
    
    int n_failure = 0;
    //initWithHeavisiteFunc(cArray, n_len);
    //(this->*initArray)(cArray, n_len);

    initArray(cArray, n_len);
    copy_memory_host_to_device(gArray, cArray, n_bytes);

    cuda_device_synchronize();
    obtain_deltas_device(gDeltaPlus, gDeltaMinus, gArray, dimGrid, dimBlock, n_len);
    cuda_device_synchronize();
    copy_memory_device_to_host(cDeltaPlusTest, gDeltaPlus, n_bytes);
    copy_memory_device_to_host(cDeltaMinusTest, gDeltaMinus, n_bytes);

    obtainDeltas();

    n_failure += testDeltasMinmodValidation(cDeltaPlus,  cDeltaPlusTest,  n_len, "delta+", test_name);
    n_failure += testDeltasMinmodValidation(cDeltaMinus, cDeltaMinusTest, n_len, "delta-", test_name);

    // Testing minmod
    cuda_device_synchronize();
    obtain_minmod_device(gBarDeltaPlus, gBarDeltaMinus, gDeltaPlus, gDeltaMinus, b, dimGrid, dimBlock, n_len);
    cuda_device_synchronize();
    obtainMinmod();
    copy_memory_device_to_host(cDeltaPlusTest, gBarDeltaPlus, n_bytes);
    copy_memory_device_to_host(cDeltaMinusTest, gBarDeltaMinus, n_bytes);

    n_failure += testDeltasMinmodValidation(cBarDeltaPlus,  cDeltaPlusTest,  n_len, "minmod+", test_name);
    n_failure += testDeltasMinmodValidation(cBarDeltaMinus, cDeltaMinusTest, n_len, "minmod-", test_name);
    
    // testing slope
    cuda_device_synchronize();
    obtain_slope_device(gSlope, gBarDeltaPlus, gBarDeltaMinus, epsilon, dimGrid, dimBlock, n_len);
    cuda_device_synchronize();
    obtainSlope();
    copy_memory_device_to_host(cDeltaPlusTest, gSlope, n_bytes);
    n_failure += testDeltasMinmodValidation(cSlope,  cDeltaPlusTest,  n_len, "slope", test_name);

    // testing Left and Right values
    cuda_device_synchronize();
    obtain_cell_intface_value_device(gRight, gLeft, gArray, gBarDeltaPlus, gBarDeltaMinus, gSlope, kappa, dimGrid, dimBlock, n_len);
    cuda_device_synchronize();
    obtainCellIntfaceValue();
    copy_memory_device_to_host(cDeltaPlusTest, gRight, n_bytes);
    n_failure += testDeltasMinmodValidation(cRight,  cDeltaPlusTest,  n_len, "Right", test_name);
    copy_memory_device_to_host(cDeltaPlusTest, gLeft, n_bytes);
    n_failure += testDeltasMinmodValidation(cLeft,  cDeltaPlusTest,  n_len, "Left", test_name);

    //  directly obtaining QR and QL and its test
    printf("  Directly get QR and QL test of %s:\n", name);
    obtain_cell_intface_value_from_Q_device(gRight, gLeft, gSlope, gBarDeltaPlus, gBarDeltaMinus, gDeltaPlus, gDeltaMinus, gArray, 
        kappa, epsilon, b, dimGrid, dimBlock, n_len);
    copy_memory_device_to_host(cDeltaPlusTest, gRight, n_bytes);
    n_failure += testDeltasMinmodValidation(cRight,  cDeltaPlusTest,  n_len, "Right", test_name);
    copy_memory_device_to_host(cDeltaPlusTest, gLeft, n_bytes);
    n_failure += testDeltasMinmodValidation(cLeft,  cDeltaPlusTest,  n_len, "Left", test_name);        

    return n_failure;
}

int FieldVars1D::testDeltasMinmodValidation(double *cArray1, double *cArray2, int n_len, char var_type[64], char test_name[64]) {
    int n_failure;
    double Cor;

    printf ("  validation of calculation of %s with initial condition of %s:\n", var_type, test_name);
    printf ("    validation of contents of 2 memory areas of %s:\n", var_type);
    n_failure = compareArrays(cArray1, cArray2, n_len);
    printf ("      n_failures: %d / %d\n", n_failure, n_len);
    Cor = testObtainCorrelFactor(cArray1, cArray2, n_len);
    printf ("      Correlation factor: %lf\n", Cor);
    if (debug_mode == 0) printTwoVars(cArray1, cArray2, n_len);
    printf ("    original values before derivertive\n");

    if (n_failure / n_len > 0.5 || Cor < 0.5) {
        printf("    Too much inconsisntency at test of %s at %s\n", var_type, test_name);
        return -1;
    } else {
        printf("    Memory test of %s at %s succeeded\n\n", var_type, test_name);
        return 0;
    }

}

void FieldVars1D::printOneVar(double *U, int n_len) {
//  print the elements of U and V to std output
    int i;

    printf("  confirming one var:\n");
    for (i = 0; i < n_len; i++)  printf("    U: %lf\n", U[i]);
    printf("\n");

}

int  FieldVars1D::validateTwoVarsByCorrelationFactor(double *U, double *V, char var_name_1[64], char var_name_2[64], double threshCor, int n_len) {
    double Cor;
    Cor = testObtainCorrelFactor(U, V, n_len);

    if (threshCor > Cor) {
        printf("  Correlation factors between %s and %s was %lf, while threshold is %lf\n  COR FACTOR IS TOO LOW\n", 
        var_name_1, var_name_2, Cor, threshCor);
        return -1;
    } else {
        printf("  Correlation factors between %s and %s was %lf, while threshold is %lf\n  VALIDATION OK!!\n", 
        var_name_1, var_name_2, Cor, threshCor);
        return 0;
    }
}

void FieldVars1D::printTwoVars(double *U, double *V, int n_len) {
    //  print the elements of U and V to std output
    int i;

    printf("  confirming two vars:\n");
    for (i = 0; i < n_len; i++)  printf("    U: %lf, V: %lf\n", U[i], V[i]);
    printf("\n");

}


#endif