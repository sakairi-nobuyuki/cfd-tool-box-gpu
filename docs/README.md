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


## Reference documents

### CUDA computation

https://http.download.nvidia.com/developer/cuda/jp/CUDA_Programming_Basics_PartI_jp.pdf
https://http.download.nvidia.com/developer/cuda/jp/CUDA_Programming_Basics_PartII_jp.pdf

https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html

https://tech-blog.optim.co.jp/entry/2019/08/15/163000

https://co-crea.jp/wp-content/uploads/2016/07/File_2.pdf

### shallow water eq. with MUSCL

https://reader.elsevier.com/reader/sd/pii/S0898122114004672?token=D737314E0310DFF74911D98D8EE6FC8FC7848BFE3D8DD1853121E87896A54B7FAAEB7CF75CB24C2AD96C41B3B87F42BA&originRegion=us-east-1&originCreation=20210816170543

- Journal of Applied Mathematics, Szu-Hsien Peng, Volume 2012, Article ID 489269, 14 pages, 
https://www.researchgate.net/publication/292617263_Numerical_methods_for_Shallow_water_equations

- Computers and Mathematics with Applications, 77(2), An improved multislope MUSCL scheme for solving shallow water equations on unstructured
grids
https://escholarship.org/content/qt0vs0w9mc/qt0vs0w9mc.pdf

- Victor Michel-Dansac, Christophe Berthon, Stéphane Clain, Françoise Foucher. A well-balanced
scheme for the shallow-water equations with topography. Computers & Mathematics with Applications, Elsevier, 2016, 72, pp.568 - 593. ff10.1016/j.camwa.2016.05.015ff. ffhal-01201825v2f
https://hal.archives-ouvertes.fr/hal-01201825/document

https://reader.elsevier.com/reader/sd/pii/S1877705816319749?token=CC759EEBB3E349F5DC36866CEE9491204C29E17D3D0A52146DC7DC4B634D6D892F292075D100BCC79AF454095E507F4F&originRegion=us-east-1&originCreation=20210816185317

https://www.emis.de/journals/HOA/JAM/Volume2012/489269.pdf

- Harten Lax, and, van Leer, SIAM rev. 25 (1) pp. 35-61, 1983, On upstream differencing and Godunov-type schemes for hyperbolic conservertion law.
https://www.jstor.org/stable/2030019
https://www.researchgate.net/publication/290998226_On_Upstream_Differencing_and_Godunov-Type_Schemes_for_Hyperbolic_Conservation_Laws

- Christophe Berthon, Victor Michel-Dansac. A simple fully well-balanced and entropy preserving
scheme for the shallow-water equations. Applied Mathematics Letters, Elsevier, 2018, 86, pp.284-290.
ff10.1016/j.aml.2018.07.013ff. ffhal-01708991v2f
https://hal.archives-ouvertes.fr/hal-01708991v2/document

- https://www.math.sciences.univ-nantes.fr/~berthon/publications/publis.htm

- SIAM J. SCI. STAT. COMPUT. Vol. 5, No. 1, March 1984, ON THE RELATION BETWEEN THE UPWIND-DIFFERENCING SCHEMES OF GODUNOV, ENGQUIST-OSHER AND ROE
https://www.researchgate.net/publication/265398564_On_the_Relation_Between_the_Upwind-Differencing_Schemes_of_Godunov_Engquist-Osher_and_Roe

- Roe, Journal of Computational Physics 43(2):357-372, Approximate Riemann Solvers, Parameter Vector, and Difference Schemes
https://www.researchgate.net/publication/222453065_Approximate_Riemann_Solvers_Parameter_Vector_and_Difference_Schemes