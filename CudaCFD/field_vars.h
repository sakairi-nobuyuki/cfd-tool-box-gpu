#ifndef __FIELD_VARS_H__
#define __FIELD_VARS_H__

class FieldVars1D {
    protected:
        double *cArray, *gArray, *cArrayMemoryTest, *gArrayMemoryTest;
        double *cDeltaPlus, *cDeltaMinus, *gDeltaPlus, *gDeltaMinus, *cDeltaPlusTest, *cDeltaMinusTest;
        int n_bytes;
        bool debug_mode;

        void initVarsWithZero();

        void testMemory();
        void testObtainDeltasAbstract(void (*initArray) (double *cArray, int n_len), double *gArray, double *gDeltaPlus, double *gDeltaMinus, 
            double *cArray, double *cDeltaPlusTest, double *cDeltaMinusTest, int n_len, int n_bytes, char test_name[64]);
        double testObtainCorrelFactor(double *U, double *V, int n_len);
        
        int compareArrays(double *ResCPU, double *ResGPU, int n_len);
        void printTwoVars(double *U, double *V, int n_len);

        void obtainDeltas();
    public:
        int n_len;
        char name[64];
        FieldVars1D();
        ~FieldVars1D();
        
        void initFieldVars(int array_length, char var_name[64]);
        void output(double time);
        
};


void initArrayWithHeavisiteFunc(double *Array, int n_len);
void initArrayWithRampFunc(double *Array, int n_len);
void initArrayWithParaboraicFunc(double *Array, int n_len);

#endif