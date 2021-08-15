#ifndef __SHALLOW_WATER_EQ_H__
#define __SHALLOW_WATER_EQ_H__

#include <string>
#include "field_vars.h"
#include "cuda_memory_config.h"

//class FieldVars1d;
class ShallowWaterEq {
    protected:
        BlockDim dimBlock;
        GridDim dimGrid;

    public:
        int n_len;
        char name[64];


        FieldVars1D U;
        FieldVars1D HU;

        ShallowWaterEq();
        ShallowWaterEq(int n_len);
        void init(int n_len_inp, char name_inp[64]);



};

#endif