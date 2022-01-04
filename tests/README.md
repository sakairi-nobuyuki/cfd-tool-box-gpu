# Test by CPPUTest


## Installation

### Compile from source
```
$ git clone git://github.com/cpputest/cpputest.git
$ sudo mv cpputest/ /usr/share
$ cd /usr/share/cpputest
$ cd cpputest_build/
$ cmake ..
$ make
$ sudo make install
$ cd ../scripts
$ chmod +x NewProject.sh
```

### Setting path

In the case when,

- Header files are installed in: `/usr/local/include`
- Libraries are installed in: `/usr/local/lib`

add following to `~/.bashrc`

```
export CPPUTEST_HOME=/usr/local
export PATH=$PATH:${CPPUTEST_HOME}/lib
```

and, 

```
$ source ~/.bashrc
```

to reflect the setting.

## Starting-up

Sample test code. `MagicalBanana` method does not make sense.

```hoge.cpp
#include


class TestClass {
    private:
        int i, j, k;
    public:
        int MagicalBanana(int i) {
            return 2 * i;
        }
};



TEST_GROUP(TestGroup1){};
TEST(TestGroup1, MELON_PEACH){
    STRCMP_EQUAL("peach", "melon");

}

int main(int argc, char *argv[]) {
    return CommandLineTestRunner::RunAllTests(argc, argv);
}
```

### Compile manually

Compilation by,

```
$ g++ hoge.o -o hoge -L/usr/local/lib -lCppUTest -lCppUTestExt
$ ./hoge

hoge.cpp:18: error: Failure in TEST(TestGroup1, MELON_PEACH)
expected
but was
difference starts at position 0 at: < melon >
^

.
Errors (1 failures, 1 tests, 1 ran, 1 checks, 0 ignored, 0 filtered out, 0 ms)
```

### Compilation with a Makefile

```Makefile
PROGRAM = hoge
OBJS = hoge.o
CXX = g++
CPPFLAGS = -Wall -O3 -fPIC
CPPFLAGS += -I$(CPPUTEST_HOME)/include
LD_LIBRARIES = -L$(CPPUTEST_HOME)/lib -lCppUTest -lCppUTestExt


$(PROGRAM):$(OBJS)
    $(CXX) $(OBJS) -o $(PROGRAM) $(LD_LIBRARIES) $(CPPFLAGS)

clean:
    rm $(OBJS)


.SUFFIXES: .o .cpp .cu

.cu.o:
    $(NVCC) -c $<
.cpp.o:
    $(CXX) $(CPPFLAGS) -c $
```

and,

```
$ make
g++ -Wall -O3 -fPIC -I/usr/local/include -c hoge.cpp
g++ hoge.o -o hoge -L/usr/local/lib -lCppUTest -lCppUTestExt -Wall -O3 -fPIC -I/usr/local/include
$ ./hoge

hoge.cpp:18: error: Failure in TEST(TestGroup1, MELON_PEACH)
expected
but was
difference starts at position 0 at: < melon >
^

.
Errors (1 failures, 1 tests, 1 ran, 1 checks, 0 ignored, 0 filtered out, 0 ms)
```