#include "linalg.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>

const int SIZE = 1024; // Square matrix size for benchmarking
const int ITERATIONS = 5; // To get a stable average

void fill_random(double* ptr, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = dis(gen);
    }
}

// Helper to transpose matrix for testing
void transpose(const double* src, double* dst, int rows, int cols) {
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

int main() {
    std::cout << "Allocating 64-byte aligned memory for " << SIZE << "x" << SIZE << " matrices..." << std::endl;
    
    size_t num_elements = SIZE * SIZE;
    
    double* matA = allocate_aligned(num_elements);
    double* matB = allocate_aligned(num_elements);
    double* matB_T = allocate_aligned(num_elements);
    double* vec = allocate_aligned(SIZE);
    double* result = allocate_aligned(num_elements);

    fill_random(matA, num_elements);
    fill_random(matB, num_elements);
    fill_random(vec, SIZE);
    transpose(matB, matB_T, SIZE, SIZE);

    std::cout << "Running Benchmarks (" << ITERATIONS << " iterations)...\n" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITERATIONS; ++i) multiply_mv_row_major(matA, SIZE, SIZE, vec, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "MV Row-Major Avg Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / ITERATIONS << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITERATIONS; ++i) multiply_mv_col_major(matA, SIZE, SIZE, vec, result);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "MV Col-Major Avg Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / ITERATIONS << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITERATIONS; ++i) multiply_mm_naive(matA, SIZE, SIZE, matB, SIZE, SIZE, result);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "MM Naive Avg Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / ITERATIONS << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITERATIONS; ++i) multiply_mm_transposed_b(matA, SIZE, SIZE, matB_T, SIZE, SIZE, result);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "MM Transposed B Avg Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / ITERATIONS << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<ITERATIONS; ++i) multiply_mm_tiled(matA, SIZE, SIZE, matB, SIZE, SIZE, result, 64);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "MM Tiled (Block=64) Avg Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / ITERATIONS << " ms\n";

    free_aligned(matA); free_aligned(matB); free_aligned(matB_T); free_aligned(vec); free_aligned(result);
    return 0;
}