PROGRAM = main
OBJS    = hoge.o main.o
NVCC    = nvcc
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

