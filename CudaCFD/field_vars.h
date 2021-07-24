#ifndef __FIELD_VARS_H__
#define __FIELD_VARS_H__

class FieldVars1D {
    protected:
        double *cArray, *gArray, *cArrayMemoryTest, *gArrayMemoryTest;
        double *cDeltaPlus, *cDeltaMinus, *gDeltaPlus, *gDeltaMinus, *cDeltaPlusTest, *cDeltaMinusTest;
        int n_bytes;

        void initVarsWithZero();
        void initVarsWithHeavisiteFunc();
        void initWithHeavisiteFunc(double *Array, int n_len);

        void testMemory();
        double testObtainCorrelFactor(double *U, double *V, int n_len);
        int compareArrays(double *ResCPU, double *ResGPU, int n_len);

        void obtainDeltas();
    public:
        int n_len;
        char name[64];
        FieldVars1D();
        FieldVars1D(int array_length, char var_name[64]);
        ~FieldVars1D();
        
        void initFieldVars(int array_length, char var_name[64]);
        void output(double time);
        
};

#endif