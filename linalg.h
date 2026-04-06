#ifndef LINALG_H
#define LINALG_H

#include <cstddef>

// Memory Management
double* allocate_aligned(size_t size, size_t alignment = 64);
void free_aligned(double* ptr);

// Part 1: Baseline Implementations
void multiply_mv_row_major(const double* matrix, int rows, int cols, const double* vector, double* result);
void multiply_mv_col_major(const double* matrix, int rows, int cols, const double* vector, double* result);
void multiply_mm_naive(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result);
void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA, const double* matrixB_transposed, int rowsB, int colsB, double* result);

// Part 2: Optimized Implementation (Cache Blocking / Tiling)
void multiply_mm_tiled(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result, int block_size);

#endif // LINALG_H