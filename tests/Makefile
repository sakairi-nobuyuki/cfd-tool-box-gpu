PROGRAM = hoge
OBJS    = hoge.o
CXX  = g++
CPPFLAGS = -Wall -O3 -fPIC
CPPFLAGS += -I$(CPPUTEST_HOME)/include
#LD_LIBRARIES = -L$(CPPUTEST_HOME)/lib -lCppUTest -lCppUTestExt
LD_LIBRARIES = -L$(CPPUTEST_HOME)/lib -lCppUTest -lCppUTestExt


$(PROGRAM):$(OBJS)
	$(CXX) $(LD_LIBRARIES) $(CPPFLAGS) $(OBJS) -o $(PROGRAM)

clean:
	rm $(OBJS)	


.SUFFIXES: .o .cpp .cu

.cu.o:
	$(NVCC) -c $<
.cpp.o:
	$(CXX) $(CPPFLAGS) -c $<

