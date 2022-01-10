#include <comput_unit_config.h>


ComputUnitConfig::ComputUnitConfig(){}
ComputUnitConfig::ComputUnitConfig(int n_max){
    _max = n_max;
    _max_x = 0;
    _max_y = 0;
    _max_z = 0;
}

ComputUnitConfig::ComputUnitConfig(int n_max_x, int n_max_y){
    _max_x = n_max_x;
    _max_y = n_max_y;
    _max_z = 0;
    _max = _max_x * _max_y;
}

ComputUnitConfig::ComputUnitConfig(int n_max_x, int n_max_y, int n_max_z){
    _max_x = n_max_x;
    _max_y = n_max_y;
    _max_z = n_max_z;
    _max = _max_x * _max_y * _max_z;
}

ComputUnitConfig::ComputUnitConfig(int n_max_x, int n_max_y, int n_max_z, int n_max){
    _max_x = n_max_x;
    _max_y = n_max_y;
    _max_z = n_max_z;
    _max = _max_x * _max_y * _max_z;

}