#include "linalg.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

const std::vector<int> SIZES = {256, 512, 1024}; // Small, Medium, Large
const int ITERATIONS = 5; 

void fill_random(double* ptr, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < size; ++i) ptr[i] = dis(gen);
}

void transpose(const double* src, double* dst, int rows, int cols) {
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            dst[j * rows + i] = src[i * cols + j];
}

bool verify_results(const double* baseline, const double* test, int size) {
    for(int i = 0; i < size; ++i) {
        if(std::abs(baseline[i] - test[i]) > 1e-5) return false;
    }
    return true;
}

void print_stats(const std::string& name, const std::vector<double>& times) {
    double sum = 0.0, mean = 0.0, variance = 0.0;
    for(double t : times) sum += t;
    mean = sum / times.size();
    for(double t : times) variance += (t - mean) * (t - mean);
    double stddev = std::sqrt(variance / times.size());
    std::cout << std::left << std::setw(30) << name 
              << "Avg: " << std::setw(8) << mean << " ms | StdDev: " << std::setw(8) << stddev << " ms\n";
}

int main() {

    for (int SIZE : SIZES) {
        std::cout << "\n--- Testing Matrix Size: " << SIZE << "x" << SIZE << " ---\n";
        size_t num_elements = SIZE * SIZE;
        
        double* matA = allocate_aligned(num_elements);
        double* matA_col = allocate_aligned(num_elements); 
        double* matB = allocate_aligned(num_elements);
        double* matB_T = allocate_aligned(num_elements);
        double* vec = allocate_aligned(SIZE);
        
        // Verification
        double* res_baseline_mv = allocate_aligned(SIZE);
        double* res_test_mv = allocate_aligned(SIZE);
        double* res_baseline_mm = allocate_aligned(num_elements);
        double* res_test_mm = allocate_aligned(num_elements);

        fill_random(matA, num_elements); fill_random(matB, num_elements); fill_random(vec, SIZE);
        transpose(matB, matB_T, SIZE, SIZE);
        transpose(matA, matA_col, SIZE, SIZE);

        multiply_mv_row_major(matA, SIZE, SIZE, vec, res_baseline_mv);
        multiply_mv_col_major(matA_col, SIZE, SIZE, vec, res_test_mv); 
        if(!verify_results(res_baseline_mv, res_test_mv, SIZE)) std::cout << "ERROR: MV calculations mismatch!\n";

        multiply_mm_naive(matA, SIZE, SIZE, matB, SIZE, SIZE, res_baseline_mm);
        multiply_mm_transposed_b(matA, SIZE, SIZE, matB_T, SIZE, SIZE, res_test_mm);
        if(!verify_results(res_baseline_mm, res_test_mm, num_elements)) std::cout << "ERROR: MM Transposed mismatch!\n";
        
        multiply_mm_tiled(matA, SIZE, SIZE, matB, SIZE, SIZE, res_test_mm, 64);
        if(!verify_results(res_baseline_mm, res_test_mm, num_elements)) std::cout << "ERROR: MM Tiled mismatch!\n";

        std::vector<double> times_mv_row(ITERATIONS), times_mv_col(ITERATIONS), times_mm_naive(ITERATIONS), times_mm_trans(ITERATIONS), times_mm_tiled(ITERATIONS);

        for(int i=0; i<ITERATIONS; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            multiply_mm_naive(matA, SIZE, SIZE, matB, SIZE, SIZE, res_baseline_mm);
            times_mm_naive[i] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

            start = std::chrono::high_resolution_clock::now();
            multiply_mm_transposed_b(matA, SIZE, SIZE, matB_T, SIZE, SIZE, res_test_mm);
            times_mm_trans[i] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

            start = std::chrono::high_resolution_clock::now();
            multiply_mm_tiled(matA, SIZE, SIZE, matB, SIZE, SIZE, res_test_mm, 64);
            times_mm_tiled[i] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        }

        print_stats("MM Naive", times_mm_naive);
        print_stats("MM Transposed B", times_mm_trans);
        print_stats("MM Tiled (Block=64)", times_mm_tiled);

        free_aligned(matA); free_aligned(matB); free_aligned(matB_T); free_aligned(vec);
        free_aligned(res_baseline_mv); free_aligned(res_test_mv); free_aligned(res_baseline_mm); free_aligned(res_test_mm);
    }

    
    size_t elements = 1024 * 1024;
    double* matA_aligned = allocate_aligned(elements);
    double* matB_aligned = allocate_aligned(elements);
    double* res_aligned = allocate_aligned(elements);
    
    double* matA_unaligned = new double[elements]; // Standard unaligned allocation
    double* matB_unaligned = new double[elements];
    double* res_unaligned = new double[elements];

    fill_random(matA_aligned, elements); fill_random(matB_aligned, elements);
    fill_random(matA_unaligned, elements); fill_random(matB_unaligned, elements);

    std::vector<double> times_aligned(ITERATIONS), times_unaligned(ITERATIONS);

    for(int i=0; i<ITERATIONS; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        multiply_mm_naive(matA_unaligned, 1024, 1024, matB_unaligned, 1024, 1024, res_unaligned);
        times_unaligned[i] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

        start = std::chrono::high_resolution_clock::now();
        multiply_mm_naive(matA_aligned, 1024, 1024, matB_aligned, 1024, 1024, res_aligned);
        times_aligned[i] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    }

    print_stats("Unaligned MM Naive", times_unaligned);
    print_stats("Aligned (64-byte) MM Naive", times_aligned);

    free_aligned(matA_aligned); free_aligned(matB_aligned); free_aligned(res_aligned);
    delete[] matA_unaligned; delete[] matB_unaligned; delete[] res_unaligned;

    return 0;
}