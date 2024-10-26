#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "../include/matrix.h"
#include "../include/random.h"

int main(void) {
  init_rand();

  const uint64_t MATRIX_A_HEIGHT = 10000;
  const uint64_t MATRIX_A_WIDTH = 1000;
  const uint64_t MATRIX_B_HEIGHT = MATRIX_A_WIDTH;
  const uint64_t MATRIX_B_WIDTH = 1000;
  const uint64_t MATRIX_C_HEIGHT = MATRIX_A_HEIGHT;
  const uint64_t MATRIX_C_WIDTH = MATRIX_B_WIDTH;

  double start, end;
  double time_spent, time_spent_p;

  Matrix a = allocate_matrix_data(MATRIX_A_WIDTH, MATRIX_A_HEIGHT);
  Matrix b = allocate_matrix_data(MATRIX_B_WIDTH, MATRIX_B_HEIGHT);
  Matrix c = allocate_matrix_data(MATRIX_C_WIDTH, MATRIX_C_HEIGHT);

  fill_with_random_numbers(a);
  fill_with_random_numbers(b);

  start = omp_get_wtime();
  multiplicate(a, b, c);
  end = omp_get_wtime();

  time_spent = (double)(end - start);
  printf("Matrix multiplication took %lf seconds\n", time_spent);

  start = omp_get_wtime();
  openmp_multiplicate(a, b, c);
  end = omp_get_wtime();

  time_spent_p = (double)(end - start);
  printf("Openmp matrix multiplication took %lf seconds\n", time_spent_p);

  printf("Speed up: %lf\n", time_spent / time_spent_p);

  free_matrix_data(c);
  free_matrix_data(b);
  free_matrix_data(a);
  return 0;
}
