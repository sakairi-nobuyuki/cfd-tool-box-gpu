MAKEFILE_DIR:=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PROGRAM = magical_banana
OBJS    = magical_banana.o
CXX     = g++
CPPFLAGS = -Wall -O3 -fPIC
CPPFLAGS += -I$(CPPUTEST_HOME)/include
CPPFLAGS += -I$(MAKEFILE_DIR)include
LD_LIBRARIES = -L$(CPPUTEST_HOME)/lib -lCppUTest -lCppUTestExt


$(PROGRAM):$(OBJS)
	$(CXX) $(OBJS) -o $(PROGRAM) $(LD_LIBRARIES) $(CPPFLAGS)

clean:
	rm $(OBJS)	

.SUFFIXES: .o .cpp .cu

.cu.o:
	$(NVCC) -c $<
.cpp.o:
	$(CXX) $(CPPFLAGS) -c $<

