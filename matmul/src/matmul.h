// Host-side entry points implemented in kernels.cu.
// The driver (driver.cpp) is plain C++ and only ever sees these.

void matmul_alloc(float *A, float *B, int n, int k, int m); // upload A, B; allocate OUT
void matmul_copyout(float *OUT, int n, int m);              // OUT_c -> host (no free)
void matmul_free();                                         // release device buffers
void matmul_get(float *OUT, int n, int m);                  // copyout + free (legacy)

// Each launcher runs one kernel into OUT_c (after matmul_alloc).
void run_naive(int n, int k, int m);      // 0. one thread per output
void run_shared(int n, int k, int m);     // 1. shared-memory tiling
void run_reg(int n, int k, int m);        // 2. 2D register tiling (8x8)
void run_float4(int n, int k, int m);     // 3. + float4 vectorization
void run_warptile(int n, int k, int m);   // 4. + warp tiling (scalar/strided A read)
void run_warptile_T(int n, int k, int m); // 4b. + transposed-As (float4 A read)
void matmul_gpu(int n, int k, int m);     // 5. + double buffering (the final kernel)
void matmul_gpu_cublas(int n, int k, int m); // cuBLAS reference
