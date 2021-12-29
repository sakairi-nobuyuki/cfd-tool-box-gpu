#ifndef __CFD_CONFG_H__
#define __CFD_CONFG_H__


#include "cuda_memory_config.h"

class CfdConfig1D {
    public:
        int array_length;
        char var_name[64];
        GridDim *dimGridInp;
        BlockDim *dimBlockInp;

        double kappa;
        double b;
        double epsilon;

        CfdConfig1D();
};

#endif
