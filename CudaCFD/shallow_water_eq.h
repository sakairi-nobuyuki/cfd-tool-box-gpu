#ifndef __SHALLOW_WATER_EQ_H__
#define __SHALLOW_WATER_EQ_H__

#include <string>
#include "field_vars.h"

//class FieldVars1d;
class ShallowWaterEq {
    protected:


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