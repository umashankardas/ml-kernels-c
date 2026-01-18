### Matrix Multiplication (Naive)

This implementation serves as the correctness baseline for all future
optimizations. It uses a straightforward triple-loop approach with
row-major memory layout and no cache or SIMD optimizations. All optimized
variants will be validated against this version.
