#include <CppUTest/CommandLineTestRunner.h>
#include <memory_config.h>


TEST_GROUP(TestConstructorDimInput){

    TEST_SETUP(){}

};

TEST(TestConstructorDimInput, DEFAULT){
    MemoryConfig memory_config;
    
    CHECK_EQUAL(128, memory_config.getGpuMemoryGranularity());

}

TEST(TestConstructorDimInput, CUDA_MEMORY_SIZE){
    MemoryConfig memory_config(256);
    
    CHECK_EQUAL(256, memory_config.getGpuMemoryGranularity());

}
TEST(TestConstructorDimInput, TEST_BYTES){
    MemoryConfig memory_config;
    
    CHECK_EQUAL(896, memory_config.obtainBytes(10, 11));
}


int main(int argc, char *argv[]) {
  return CommandLineTestRunner::RunAllTests(argc, argv);
}