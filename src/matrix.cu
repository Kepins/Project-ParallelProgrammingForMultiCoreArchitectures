#include <stdint.h>

#include "../include/matrix.h"
#include "../include/random.h"

float* get_element(Matrix m, int row, int col) { return &(m.d[row * m.w + col]); }

Matrix allocate_matrix_data(uint64_t width, uint64_t height) {
  Matrix matrix;
  matrix.w = width;
  matrix.h = height;

  matrix.d = new float[width * height];

  return matrix;
}

void fill_with(Matrix matrix, double val) {
  for (uint64_t i = 0; i < matrix.h; i++) {
    for (uint64_t j = 0; j < matrix.w; j++) {
      *get_element(matrix, i, j) = val;
    }
  }
}

void free_matrix_data(Matrix matrix) {
  free(matrix.d);
}

void multiplicate(Matrix a, Matrix b, Matrix c) {
  for (uint64_t i = 0; i < c.h; i++) {
    for (uint64_t j = 0; j < c.w; j++) {
      *get_element(c, i, j) = 0.0f;
      for (uint64_t k = 0; k < a.w; k++) {
        *get_element(c, i, j) += *get_element(a, i, k) * *get_element(b, k, j);
      }
    }
  }
}

void openmp_multiplicate(Matrix a, Matrix b, Matrix c) {
#pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < c.h; i++) {
    for (uint64_t j = 0; j < c.w; j++) {
      *get_element(c, i, j) = 0.0f;
      for (uint64_t k = 0; k < a.w; k++) {
        *get_element(c, i, j) += *get_element(a, i, k) * *get_element(b, k, j);
      }
    }
  }
}

void openmp_multiplicate2(Matrix a, Matrix b, Matrix c) {
  const uint64_t BLOCK_SIZE = 16;
  fill_with(c, 0.0);

  uint64_t block_num_i = c.h / BLOCK_SIZE;
  uint64_t block_num_j = c.w / BLOCK_SIZE;
  uint64_t block_num_k = a.w / BLOCK_SIZE; // a.w or b.h

#pragma omp parallel for collapse(2)
  for (uint64_t bi = 0; bi < block_num_i; bi++) {
    for (uint64_t bj = 0; bj < block_num_j; bj++) {
      for (uint64_t bk = 0; bk < block_num_k; bk++) {
        // C bi,bj - partial sum from: A bi,bk and  B bk,bj
        for (uint64_t i = 0; i < BLOCK_SIZE; i++) {
          for (uint64_t j = 0; j < BLOCK_SIZE; j++) {
            uint64_t global_c_row = bi * BLOCK_SIZE + i;
            uint64_t global_c_col = bj * BLOCK_SIZE + j;
            double partial_sum = 0.0;
            for (uint64_t k = 0; k < BLOCK_SIZE; k++) {
              // Calculate the indices in the global matrix
              uint64_t global_a_row = bi * BLOCK_SIZE + i;
              uint64_t global_a_col = bk * BLOCK_SIZE + k;
              uint64_t global_b_row = bk * BLOCK_SIZE + k;
              uint64_t global_b_col = bj * BLOCK_SIZE + j;
              // Accumulate the partial sum
              partial_sum += *get_element(a, global_a_row, global_a_col) *
                             *get_element(b, global_b_row, global_b_col);
            }
            // Update the result matrix C
            *get_element(c, global_c_row, global_c_col) += partial_sum;
          }
        }
      }
    }
  }
}

void cuda_multiplicate(Matrix a, Matrix b, Matrix c) {}

void cuda_multiplicate2(Matrix a, Matrix b, Matrix c) {}
