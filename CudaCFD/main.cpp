#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "field_vars.h"
#include "cuda_cfd_kernel_funcs.h"
#include "shallow_water_eq.h"

using namespace std;

int main() {
    ShallowWaterEq SWE_test;

    //SWE_test.init(100);
    SWE_test.init(100, "test");

    printf("%s %d\n", SWE_test.HU.name, SWE_test.HU.n_len);
    

    

    

    


}