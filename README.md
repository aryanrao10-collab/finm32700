# High-Performance Linear Algebra Kernels

**Team Members:** Aryan Rao (Solo Team)

## Build Instructions (macOS)
To compile the project and leverage maximum compiler optimizations, use the `-O3` flag. 

g++ -O3 -std=c++14 linalg.cpp main.cpp -o linalg_bench

To run the benchmarks:

./linalg_bench

Discussion Questions

1. Explain the key differences between pointers and references in C++. When would you choose to use a pointer over a reference, and vice versa?

A pointer is a variable that stores a memory address and can be reassigned to point elsewhere, or be set to nullptr. A reference is a direct alias to an existing variable, must be initialized upon declaration, and cannot be reassigned or be null. In numerical algorithms, I use pointers for dynamically allocated contiguous blocks of memory and when interacting with C-style APIs. I use references for passing complex objects into functions safely without copying, ensuring they exist and are valid.

2. How does the row-major and column-major storage order affect memory access patterns?

In C/C++, contiguous memory is traversed linearly. In a row-major matrix, iterating across a row means accessing elements directly next to each other in RAM. In a column-major setup, iterating across a row requires jumping forward in memory by a stride equal to the number of rows. Our benchmarks proved this: multiply_mv_row_major was significantly faster than col_major because the CPU's cache line fetches were fully utilized in the row-major version, whereas the column-major version suffered from constant cache misses.

3. Describe how CPU caches work (L1, L2, L3) and temporal/spatial locality.

CPUs use a memory hierarchy to avoid the massive latency of main RAM. L1 is the smallest/fastest, tied to a specific core; L2 is larger but slightly slower; L3 is the largest and shared across cores.

Spatial Locality: Accessing memory addresses close to one another. 

Temporal Locality: Reusing the same memory address frequently.

4. What is memory alignment, and why is it important for performance?

Memory alignment ensures that the starting address of a data structure is a multiple of a specific byte boundary (e.g., 64 bytes). Modern CPUs read memory in chunks. If a double array crosses an unaligned boundary, the CPU might require two separate fetch operations to retrieve a single piece of data. By using posix_memalign, we ensured the matrices perfectly align with the CPU's 64-byte cache lines, eliminating misaligned read penalties.

5. Discuss the role of compiler optimizations (like inlining).

The inline keyword hints to the compiler to replace a function call with the actual code body, eliminating the stack setup and teardown overhead—crucial for tiny functions called millions of times inside nested loops. Using -O3 radically changed performance; the compiler auto-vectorized loops and aggressively inlined on its own. The drawback of aggressive optimization is that compiling takes longer, the binary size increases, and debugging becomes very difficult because the source code no longer maps 1:1 to the assembly execution.

6. What were the main performance bottlenecks? How did profiling guide you?

Profiling in macOS Instruments revealed that nearly 90% of the execution time in multiply_mm_naive was spent waiting on memory fetches for Matrix B. Because the inner loop traverses Matrix B by column, it causes a cache miss on almost every iteration. This guided the immediate implementation of multiply_mm_transposed_b, which flips B beforehand so both matrices can be traversed with perfect spatial locality.

7. Reflect on the teamwork aspect of this assignment.

As a solo student managing a group assignment, the initial challenge was context-switching between writing raw baseline implementations, building a rigorous benchmarking framework, and diving deep into macOS profiling tools. However, tackling this alone forced a holistic understanding of the entire pipeline. Seeing firsthand how a simple structural change in my own code dramatically altered the hardware-level cache performance made the connection between abstract algorithmic complexity and physical hardware limitations much clearer.