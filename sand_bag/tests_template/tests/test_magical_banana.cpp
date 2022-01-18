
#include <CppUTest/CommandLineTestRunner.h>
#include <magical_banana.h>



TEST_GROUP(TestGroup1){};
TEST(TestGroup1, MELON_PEACH){
    STRCMP_EQUAL("peach", "melon");
}

TEST_GROUP(TestMagicalBanana){
    MagicalBanana mb;
    
    TEST_SETUP() {

    }
};
TEST(TestMagicalBanana, MAGICAL_TWO){
    MagicalBanana mb_arg(1);
    DOUBLES_EQUAL(mb.MagicalTwo(2), 4.0, 1.0E-06);
}


int main(int argc, char *argv[]) {
  return CommandLineTestRunner::RunAllTests(argc, argv);
}