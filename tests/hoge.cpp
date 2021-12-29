#include <CppUTest/CommandLineTestRunner.h>


class TestClass {
    private:
      int i, j, k;
    public:
        int MagicalBanana(int i) {
            return 2 * i;
        }
};



//int main() {
//    TestClass tc;
//    printf("%d\n", 2);
//    printf("%d\n", tc.MagicalBanana(2));


//    return 0;
//}


TEST_GROUP(TestGroup1){};
TEST(TestGroup1, MELON_PEACH){
    STRCMP_EQUAL("peach", "melon");


}

int main(int argc, char *argv[]) {
  return CommandLineTestRunner::RunAllTests(argc, argv);
}