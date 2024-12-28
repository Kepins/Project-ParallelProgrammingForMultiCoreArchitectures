#include <stdint.h>
#include <cuda_runtime.h>

#include "../include/matrix.h"
#include "../include/random.h"

__host__ __device__ float* get_element(Matrix m, int row, int col) { return &(m.d[row * m.w + col]); }

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


__global__ void matrixMultiplyKernel(Matrix d_a, Matrix d_b, Matrix d_c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < d_c.h && col < d_c.w) {
        float sum = 0.0f;
        for (uint64_t k = 0; k < d_a.w; k++) {
          sum += *get_element(d_a, row, k) * *get_element(d_b, k, col);
        }
        *get_element(d_c, row, col) = sum;
    }
}


void cuda_multiplicate(Matrix h_a, Matrix h_b, Matrix h_c) {
  Matrix d_a = {NULL, h_a.w, h_a.h}, d_b = {NULL, h_b.w, h_b.h}, d_c = {NULL, h_c.w, h_c.h};

  cudaMallocManaged((void**)&d_a.d, sizeof(float) * d_a.w * d_a.h);
  cudaMallocManaged((void**)&d_b.d, sizeof(float) * d_b.w * d_b.h);
  cudaMallocManaged((void**)&d_c.d, sizeof(float) * d_c.w * d_c.h);

  memcpy(d_a.d, h_a.d, sizeof(float) * h_a.h * h_a.w);
  memcpy(d_b.d, h_b.d, sizeof(float) * h_b.h * h_b.w);

  const int TILE_SIZE = 16;
  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((d_c.w + TILE_SIZE - 1) / TILE_SIZE, (d_c.h + TILE_SIZE - 1) / TILE_SIZE);
  matrixMultiplyKernel<<<gridDim, blockDim>>>(d_a, d_b, d_c);

  cudaDeviceSynchronize();

  memcpy(h_c.d, d_c.d, sizeof(float) * h_c.h * h_c.w);

  cudaFree(d_c.d);
  cudaFree(d_b.d);
  cudaFree(d_a.d);
}

void cuda_multiplicate2(Matrix a, Matrix b, Matrix c) {}
