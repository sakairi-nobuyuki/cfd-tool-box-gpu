#include <stdio.h>
#include <magical_banana.h>


MagicalBanana::MagicalBanana() {

}


MagicalBanana::MagicalBanana(int n) {
    printf("number of bananas: %d\n", n);
}


double MagicalBanana::MagicalTwo(int i){
    return 2.0 * i;
}