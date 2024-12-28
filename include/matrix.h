#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float *d;
  uint64_t w;
  uint64_t h;
} Matrix;

__host__ __device__ float* get_element(Matrix m, int row, int col);

Matrix allocate_matrix_data(uint64_t w, uint64_t h);
void free_matrix_data(Matrix);

void multiplicate(Matrix a, Matrix b, Matrix c);
void openmp_multiplicate(Matrix a, Matrix b, Matrix c);
void openmp_multiplicate2(Matrix a, Matrix b, Matrix c);
void cuda_multiplicate(Matrix a, Matrix b, Matrix c);
void cuda_multiplicate2(Matrix a, Matrix b, Matrix c);

#endif
