PROGRAM = main
OBJS    = field_vars.o cuda_front_end.o main.o
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

