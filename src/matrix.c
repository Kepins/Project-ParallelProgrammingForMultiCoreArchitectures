#include <stdint.h>

#include "../include/matrix.h"
#include "../include/random.h"

Matrix allocate_matrix_data(uint64_t width, uint64_t height) {
  Matrix matrix;
  matrix.w = width;
  matrix.h = height;

  matrix.d = (double **)malloc(height * sizeof(double *));
  for (uint64_t i = 0; i < height; i++) {
    matrix.d[i] = (double *)malloc(width * sizeof(double));
  }
  return matrix;
}

void free_matrix_data(Matrix matrix) {
  for (uint64_t i = 0; i < matrix.h; i++) {
    free(matrix.d[i]);
  }
  free(matrix.d);
}

void multiplicate(Matrix a, Matrix b, Matrix c) {
  for (uint64_t i = 0; i < c.h; i++) {
    for (uint64_t j = 0; j < c.w; j++) {
      c.d[i][j] = 0.0;
      for (uint64_t k = 0; k < a.w; k++) {
        c.d[i][j] += a.d[i][k] * b.d[k][j];
      }
    }
  }
}

void openmp_multiplicate(Matrix a, Matrix b, Matrix c) {
  #pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < c.h; i++) {
    for (uint64_t j = 0; j < c.w; j++) {
      c.d[i][j] = 0.0;
      for (uint64_t k = 0; k < a.w; k++) {
        c.d[i][j] += a.d[i][k] * b.d[k][j];
      }
    }
  }
}
