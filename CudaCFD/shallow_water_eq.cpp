#ifndef __SHALLOW_WATER_EQ_CPP__
#define __SHALLOW_WATER_EQ_CPP__

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "field_vars.h"
#include "cuda_cfd_kernel_funcs.h"
#include "shallow_water_eq.h"


using namespace std;

//class FieldVars1D;
ShallowWaterEq::ShallowWaterEq() {}
ShallowWaterEq::ShallowWaterEq(int n_len) {
    cout << "length " << n_len << endl;

    //U.init_field_vars(n_len, "U");
    //HU.init_field_vars(n_len, "HU");
}

void ShallowWaterEq::init(int n_len_inp, char name_inp[64]) {
    n_len = n_len_inp;
    sprintf(name, "%s", name_inp);
    
    cout << "initializing \"" << name << "\" of shallow water eq class" << endl;
    cout << "  length: " << n_len << endl;
    
    U.initFieldVars(n_len, "U");
    //HU.init_field_vars(n_len, "HU");

}



#endif