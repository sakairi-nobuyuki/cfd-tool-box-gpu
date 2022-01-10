#ifndef __COMPUT_UNIT_CONFIG_H__
#define __COMPUT_UNIT_CONFIG_H__

class ComputUnitConfig{
    // Data class for calculation unit wuth GPU.
    // Used for information about number of block, grid, or thread. 
    private:
        // Attributes:
        //   int _max_x: max number of calculation unit in X-direction, if it can be apparently declared. This is an optional variable.
        //   int _max_y: max number of calculation unit in Y-direction, if it can be apparently declared. This is an optional variable.
        //   int _max_z: max number of calculation unit in Z-direction, if it can be apparently declared. This is an optional variable.
        //   int _max: max number of calculation unit. This is an optional variable.
        int _max_x, _max_y, _max_z, _max;
    public:
        int x, y, z;
        ComputUnitConfig();
        ComputUnitConfig(int n_max_x, int n_max_y, int n_max_z, int n_max);
        ComputUnitConfig(int n_max_x, int n_max_y, int n_max_z);
        ComputUnitConfig(int n_max_x, int n_max_y);
        ComputUnitConfig(int n_max);
};


#endif