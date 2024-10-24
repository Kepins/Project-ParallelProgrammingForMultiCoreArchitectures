#include <stdint.h>

#include "matrix.h"
#include "random.h"


int main(){
    init_rand();

    const uint64_t MATRIX_A_HEIGHT = 1000;
    const uint64_t MATRIX_A_WIDTH = 1000;
    const uint64_t MATRIX_B_HEIGHT = MATRIX_A_WIDTH;
    const uint64_t MATRIX_B_WIDTH = 1000;
    const uint64_t MATRIX_C_HEIGHT = MATRIX_A_HEIGHT;
    const uint64_t MATRIX_C_WIDTH = MATRIX_B_WIDTH;

    Matrix a = allocate_matrix_data(MATRIX_A_WIDTH, MATRIX_A_HEIGHT);
    Matrix b = allocate_matrix_data(MATRIX_B_WIDTH, MATRIX_B_HEIGHT);
    Matrix c = allocate_matrix_data(MATRIX_C_WIDTH, MATRIX_C_HEIGHT);

    fill_with_random_numbers(a);
    fill_with_random_numbers(b);

    multiplicate(a, b, c);

    free_matrix_data(c);
    free_matrix_data(b);
    free_matrix_data(a);
    return 0;
}
