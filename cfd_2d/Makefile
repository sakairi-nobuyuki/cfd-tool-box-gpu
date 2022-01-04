PROGRAM = cuda_cfd_2d
OBJS    = /src/program_architecture/memory_config.o
NVCC = nvcc
CXX  = g++
CPPFLAGS = -Wall -O3 -fPIC -Iinclude

$(PROGRAM):$(OBJS)
	$(CXX) $(OBJS) -o $(PROGRAM)
#	$(NVCC) $(OBJS) -o $(PROGRAM)

clean:
	rm $(OBJS)	


.SUFFIXES: .o .cpp .cu

.cu.o:
	$(NVCC) -c $<
.cpp.o:
	$(CXX) $(CPPFLAGS) -c $<

