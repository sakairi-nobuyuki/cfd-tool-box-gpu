#ifndef __SHALLOW_WATER_EQ_CPP__
#define __SHALLOW_WATER_EQ_CPP__

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "field_vars.h"
#include "cuda_cfd_kernel_funcs.h"
#include "shallow_water_eq.h"
#include "cuda_memory_config.h"

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
    cout << "Initializing \"" << name_inp << "\" of shallow water eq" << endl;
    cout << "  configure CUDA memory" << endl;
    setCudaGridBlockConfig1D(n_len, &dimGrid, &dimBlock);
    printf("    grid (%d, %d), block (%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y, dimBlock.z);


    cout << "  initializing \"" << name << "\" of shallow water eq class" << endl;
    cout << "    length: " << n_len << endl;
    H.initUnknownFieldVars1D(n_len, "H", &dimGrid, &dimBlock);
    Q.initUnknownFieldVars1D(n_len, "Q", &dimGrid, &dimBlock);
    
    Hflux.initFluxFieldVars1D(n_len, "Hflux", &dimGrid, &dimBlock);
    Qflux.initFluxFieldVars1D(n_len, "Qflux", &dimGrid, &dimBlock);



}

void ShallowWaterEq::createFlux() {
    create_h_flux_device(Hflux.gArray, Q.gArray, &dimGrid, &dimBlock, n_len);
    create_q_flux_device(Qflux.gArray, Q.gArray, H.gArray, g, &dimGrid, &dimBlock, n_len);    
    cuda_device_synchronize();
}


void ShallowWaterEq::deinit() {

    H.deinitUnknownFieldVars1D();
    Q.deinitUnknownFieldVars1D();
    Hflux.deinitFluxFieldVars1D();
    Qflux.deinitFluxFieldVars1D();
}

#endif