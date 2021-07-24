PROGRAM = main
OBJS    = cuda_cfd_kernel_funcs.o field_vars.o shallow_water_eq.o main.o
NVCC = nvcc
CXX  = g++
CPPFLAGS = -Wall -O3 -fPIC

$(PROGRAM):$(OBJS)
	$(NVCC) $(OBJS) -o $(PROGRAM)

clean:
	rm $(OBJS)	


.SUFFIXES: .o .cpp .cu

.cu.o:
	$(NVCC) -c $<
.cpp.o:
	$(CXX) $(CPPFLAGS) -c $<

