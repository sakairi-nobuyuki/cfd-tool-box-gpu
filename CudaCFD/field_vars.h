#ifndef __FIELD_VARS_H__
#define __FIELD_VARS_H__

#include "cuda_memory_config.h"

class FieldVars1D {
    protected:
        double *cArray, *gArray, *cArrayMemoryTest, *gArrayMemoryTest;
        double *cDeltaPlus, *cDeltaMinus, *gDeltaPlus, *gDeltaMinus, *cDeltaPlusTest, *cDeltaMinusTest;
        double *cBarDeltaPlus, *cBarDeltaMinus, *gBarDeltaPlus, *gBarDeltaMinus, *cBarDeltaPlusTest, *cBarDeltaMinusTest;
        double *cSlope, gSlope;
        double b, epsilon;
        int n_bytes;
        BlockDim *dimBlock;
        GridDim *dimGrid;
        bool debug_mode;

        void initVarsWithZero();

        void testMemory();
        int testObtainDeltasMinmodAbstract(void (*initArray) (double *cArray, int n_len), double *gArray, double *gDeltaPlus, double *gDeltaMinus, 
            double *cArray, double *cDeltaPlusTest, double *cDeltaMinusTest, int n_len, int n_bytes, char test_name[64]);
        
        double testObtainCorrelFactor(double *U, double *V, int n_len);
        int testMemoryCopy(double *cArray, double *gArray, double *gArrayMemoryTest, double *cArrayMemoryTest, char var_name[64]);
        int testDerivertive();
        int testDeltasMinmodValidation(double *cArray1, double *cArray2, int n_len, char var_type[64], char test_name[64]);
        
        int compareArrays(double *ResCPU, double *ResGPU, int n_len);
        void printTwoVars(double *U, double *V, int n_len);

        void obtainDeltas();
        void obtainMinmod();
    public:
        int n_len;
        char name[64];
        FieldVars1D();
        ~FieldVars1D();
        
        void initFieldVars(int array_length, char var_name[64], GridDim *dimGridInp, BlockDim *dimBlockInp);
        //void initFieldVars(int array_length, char var_name[64]);
        void output(double time);
        
};


void initArrayWithHeavisiteFunc(double *Array, int n_len);
void initArrayWithRampFunc(double *Array, int n_len);
void initArrayWithParaboraicFunc(double *Array, int n_len);

#endif