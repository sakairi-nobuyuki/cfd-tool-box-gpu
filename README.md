# Sand bag for numerical simulation by cuda

Nothing guaranteed.

Installation of cuda and its necessary dirvers shall be done before hand.


# Usage

Some files names as `test_hoge.cu` are source files can be compiled and executed according to their functions.

Compilation.
```
$ nvcc test_hoge.cu -o test_hoge
```

Preparing `Makefile` might be smart, however I didn't do it, since the system is truely simple. 

Execution.
```
$ ./test_hoge
```
Calcutaion results are saved into same directory. Each files are named such as `00000.dat`, `01000.dat` or `02000.dat`.
The position of the mesh and each values are contained in each `*.dat` file.
They can be visualized by gnuplot.

# File list

Calculation mesh can be changed by modifying `NN` in each files.

- `test.cu`: Solves diffusion equation.
- `test_convective.cu`: Solves convective equation by 1st order up-wind method. Courant number can be changed by `Co`. Phase speed is `C`.
- `test_conventive_cpu.c`: Solves convective equation by 1st order up-wind method with CPU. Courant number can be changed by `Co`. Phase speed is `C`.
