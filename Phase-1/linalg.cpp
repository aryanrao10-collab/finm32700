#include "linalg.h"
#include <iostream>
#include <cstdlib>
#include <algorithm>

double* allocate_aligned(size_t size, size_t alignment) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size * sizeof(double)) != 0) {
        return nullptr;
    }
    return static_cast<double*>(ptr);
}

void free_aligned(double* ptr) {
    free(ptr);
}

// Inline helper for basic error checking
inline bool check_dimensions(const double* a, const double* b, double* c) {
    return (a != nullptr && b != nullptr && c != nullptr);
}

void multiply_mv_row_major(const double* matrix, int rows, int cols, const double* vector, double* result) {
    if (!check_dimensions(matrix, vector, result)) return;
    
    for (int i = 0; i < rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

void multiply_mv_col_major(const double* matrix, int rows, int cols, const double* vector, double* result) {
    if (!check_dimensions(matrix, vector, result)) return;

    for (int i = 0; i < rows; ++i) result[i] = 0.0;
    
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            result[i] += matrix[j * rows + i] * vector[j];
        }
    }
}

void multiply_mm_naive(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result) {
    if (!check_dimensions(matrixA, matrixB, result) || colsA != rowsB) return;
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < colsA; ++k) {
                sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
            }
            result[i * colsB + j] = sum;
        }
    }
}

void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA, const double* matrixB_transposed, int rowsB, int colsB, double* result) {
    if (!check_dimensions(matrixA, matrixB_transposed, result) || colsA != rowsB) return;
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < colsA; ++k) {
                sum += matrixA[i * colsA + k] * matrixB_transposed[j * colsB + k];
            }
            result[i * colsB + j] = sum;
        }
    }
}

void multiply_mm_tiled(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result, int block_size) {
    if (!check_dimensions(matrixA, matrixB, result) || colsA != rowsB) return;
    
    for (int i = 0; i < rowsA * colsB; ++i) result[i] = 0.0;

    for (int ii = 0; ii < rowsA; ii += block_size) {
        for (int jj = 0; jj < colsB; jj += block_size) {
            for (int kk = 0; kk < colsA; kk += block_size) {
                for (int i = ii; i < std::min(ii + block_size, rowsA); ++i) {
                    for (int j = jj; j < std::min(jj + block_size, colsB); ++j) {
                        double sum = 0.0;
                        for (int k = kk; k < std::min(kk + block_size, colsA); ++k) {
                            sum += matrixA[i * colsA + k] * matrixB[k * colsB + j];
                        }
                        result[i * colsB + j] += sum;
                    }
                }
            }
        }
    }
}