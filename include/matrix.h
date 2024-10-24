#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    double **d;
    uint64_t w;
    uint64_t h;
} Matrix;

Matrix allocate_matrix_data(uint64_t w, uint64_t h);
void free_matrix_data(Matrix);

void multiplicate(Matrix a, Matrix b, Matrix c);

#endif
