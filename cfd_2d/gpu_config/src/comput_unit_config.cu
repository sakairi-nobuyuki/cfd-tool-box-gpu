#include <comput_unit_config.h>


ComputUnitConfig::ComputUnitConfig(){}
ComputUnitConfig::ComputUnitConfig(int n_max){
    set1DimConfig(n_max);
}

ComputUnitConfig::ComputUnitConfig(int n_max_x, int n_max_y){
    set2DimConfig(n_max_x, n_max_y);
}

ComputUnitConfig::ComputUnitConfig(int n_max_x, int n_max_y, int n_max_z){
    set3DimConfig(n_max_x, n_max_y, n_max_z);

}


void ComputUnitConfig::set3DimConfig(int n_max_x, int n_max_y, int n_max_z){
    _max_x = n_max_x;
    _max_y = n_max_y;
    _max_z = n_max_z;
    _max = _max_x * _max_y * _max_z;
}

void ComputUnitConfig::set2DimConfig(int n_max_x, int n_max_y){
    _max_x = n_max_x;
    _max_y = n_max_y;
    _max_z = 0;
    _max = _max_x * _max_y;   
    
}


void ComputUnitConfig::set1DimConfig(int n_max_x){
    _max_x = n_max_x;
    _max_y = 0;
    _max_z = 0;
    _max = _max_x;   
    
}


int ComputUnitConfig::getDimX(){
    return _max_x;
}

int ComputUnitConfig::getDimY(){
    return _max_y;
}
int ComputUnitConfig::getDimZ(){
    return _max_z;
}

int ComputUnitConfig::getDim(){
    return _max;
}
