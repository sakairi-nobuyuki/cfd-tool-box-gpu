# CFD for CUDA documkents


## Configuration of the program


```mermaid
classDiagram
    class ShallowWaterEq {
        -public: int n_len
        -public: char name[64]
        -public: FieldVars1D U
        -public: FieldVars1D HU
        -public: ShallowWaterEq()
        -public: ShallowWaterEq(int n_len)
        -public: void init(int n_len_inp,char name_inp[64])
    }
    class FieldVars1D {
        -protected: double *cArray
        -protected: double *gArray
        -protected: double *cArrayMemoryTest
        -protected: double *gArrayMemoryTest
        -protected: double *cDeltaPlus
        -protected: double *cDeltaMinus
        -protected: double *gDeltaPlus
        -protected: double *gDeltaMinus
        -protected: double *cDeltaPlusTest
        -protected: double *cDeltaMinusTest
        -protected: int n_bytes
        -protected: bool debug_mode
        -public: int n_len
        -public: int char name[64]
        -protected: void initVarsWithZero()
        -protected: void testMemory()
        -protected: void testObtainDeltasAbstract(void (*initArray) (double *cArray, int n_len), double *gArray, double *gDeltaPlus, double *gDeltaMinus, double *cArray, double *cDeltaPlusTest, double *cDeltaMinusTest, int n_len, int n_bytes, char test_name[64])
        -protected: double testObtainCorrelFactor(double *U, double *V, int n_len)
        -protected: int compareArrays(double *ResCPU, double *ResGPU, int n_len)
        -protected: void printTwoVars(double *U, double *V, int n_len)
        -protected: void obtainDeltas()
        -public: FieldVars1D()
        -public: ~FieldVars1D()
        -public: void initFieldVars(int array_length, char var_name[64])
        -public: void output(double time)
    }
    class cuda_cfd_kernel_funcs {
        -void allocate_cuda_memory(void **U, int n_bytes);
        -void copy_memory_host_to_device(double *gU, double *U, int n_bytes);
        -void copy_memory_device_to_host(double *U, double *gU, int n_bytes);
        -void copy_memory_device_to_device(double *gV, double *gU, int n_bytes);
        -void cuda_device_synchronize();
        -void obtain_deltas_device(double *gU, double *gDeltaPlus, double *gDeltaMinus, int n_len);
        -void free_cuda_memory(double *gU);
        -void copy_memory_mock();
    }
    class BlockDim {
        -public: int x
        -public: int y
        -public: int z
    }
    class GridDim {
        -public: int x
        -public: int y
    }
    class cuda_memory_config{
        -void setCudaGridBlockConfig1D(int n_len, GridDim *dimGrid, BlockDim *dimBlock)
    }
    ShallowWaterEq o-- BlockDim
    ShallowWaterEq o-- GridDim
    FieldVars1D o-- BlockDim
    FieldVars1D o-- GridDim
    ShallowWaterEq o-- FieldVars1D
    ShallowWaterEq <-- cuda_memory_config
    FieldVars1D <-- cuda_cfd_kernel_funcs
```


## Reference documts

https://http.download.nvidia.com/developer/cuda/jp/CUDA_Programming_Basics_PartI_jp.pdf
https://http.download.nvidia.com/developer/cuda/jp/CUDA_Programming_Basics_PartII_jp.pdf

https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html

https://tech-blog.optim.co.jp/entry/2019/08/15/163000

https://co-crea.jp/wp-content/uploads/2016/07/File_2.pdf
