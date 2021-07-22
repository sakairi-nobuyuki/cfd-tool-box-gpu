#ifndef __FIELD_VARS_H__
#define __FIELD_VARS_H__

class FieldVars1D {
    protected:
        double *cArray, *gArray;
        double *cDeltaPlus, *cDeltaMinus, *gDeltaPlus, *gDeltaMinus;
        int n_bytes;

        void initVarsWithZero();
        void initVarsWithHeavisiteFunc();
        void testDeviceVarsAllocation();

        void obtainDeltas();
    public:
        int n_len;
        char name[64];
        FieldVars1D();
        FieldVars1D(int array_length, char var_name[64]);
        ~FieldVars1D();
        
        void init_field_vars(int array_length, char var_name[64]);
        void output(double time);
        
};

#endif