#include <stdlib.h>
#include <stdint.h>

#include "random.h"


void init_rand(){
    const unsigned int SEED = 522766;
    srand(SEED);
}

double random_double_in_range(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

void fill_with_random_numbers(Matrix matrix){
    const double MIN_VALUE = 0;
    const double MAX_VALUE = 10;
    for(uint64_t i=0;i<matrix.h;i++){
        for(uint64_t j=0;j<matrix.w;j++){
            matrix.d[i][j] = random_double_in_range(MIN_VALUE, MAX_VALUE);
        }
    }
}
