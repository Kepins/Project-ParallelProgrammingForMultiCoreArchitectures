#include <stdint.h>
#include <stdlib.h>

#include "../include/random.h"

void init_rand(void) {
  const unsigned int SEED = 522766;
  srand(SEED);
}

float random_float_in_range(float min, float max) {
  return min + (max - min) * ((float)rand() / RAND_MAX);
}

void fill_with_random_numbers(Matrix matrix) {
  const float MIN_VALUE = 0;
  const float MAX_VALUE = 10;
  for (uint64_t i = 0; i < matrix.h; i++) {
    for (uint64_t j = 0; j < matrix.w; j++) {
      *get_element(matrix, i, j) = random_float_in_range(MIN_VALUE, MAX_VALUE);
    }
  }
}
