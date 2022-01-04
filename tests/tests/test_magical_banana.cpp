
#include <CppUTest/CommandLineTestRunner.h>


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