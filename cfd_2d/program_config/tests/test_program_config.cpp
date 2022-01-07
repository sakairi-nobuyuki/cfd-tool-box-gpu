#include <CppUTest/CommandLineTestRunner.h>
#include <memory_config.h>


TEST_GROUP(TestConstructorDimInput){

    TEST_SETUP(){}
};
TEST(TestConstructorDimInput, DEFAULT){
    MemoryConfig memory_config(10, 11);
    
    CHECK_EQUAL(10, memory_config.n_len_x);
    CHECK_EQUAL(11, memory_config.n_len_y);
    CHECK_EQUAL(128, memory_config.n_optimal_memory_granularity);
}

TEST(TestConstructorDimInput, CUDA_MEMORY_SIZE){
    MemoryConfig memory_config(10, 11, 256);
    
    CHECK_EQUAL(10, memory_config.n_len_x);
    CHECK_EQUAL(11, memory_config.n_len_y);
    CHECK_EQUAL(256, memory_config.n_optimal_memory_granularity);
}


int main(int argc, char *argv[]) {
  return CommandLineTestRunner::RunAllTests(argc, argv);
}