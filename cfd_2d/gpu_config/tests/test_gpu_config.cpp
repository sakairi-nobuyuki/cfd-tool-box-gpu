#include <CppUTest/CommandLineTestRunner.h>
#include <gpu_config.h>
#include <stdio.h>

TEST_GROUP(TestGpuProp){

    TEST_SETUP(){}

};

TEST(TestGpuProp, CONSTRUCTOR){
    GpuConfig g_conf(0);
    int n_thread = g_conf.getThreadConfig().getDim();
    
    CHECK_EQUAL(1024, n_thread);
    printf("getter %d\n", n_thread);
}






int main(int argc, char *argv[]) {
  return CommandLineTestRunner::RunAllTests(argc, argv);
}