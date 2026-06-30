#include "matmul.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

/*

   0 1 2 3 4 5
   1 2 3 4 5 6
   1 2 3 4 5 6
   1 2 3 4 5 6

 */

void matmul_cpu_cache_friendly(float *A, float *B, float *OUT, int n, int k, int m) {
  for (int i = 0; i < n; i++) {
    for (int mid = 0; mid < k; mid++) {
      for (int j = 0; j < m; j++) {
        // Accumulate the partials directly
        OUT[i * m + j] += A[i * k + mid] * B[mid * m + j];
      }
    }
  }
}

void matmul_cpu_naive(float *A, float *B, float *OUT, int n, int k, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      float s = 0;
      for (int mid = 0; mid < k; mid++) {
        s += A[i * k + mid] * B[mid * m + j];
      }
      OUT[i * m + j] = s;
    }
  }
}

// Raw-binary dump: [int32 rows][int32 cols][float32 * rows*cols].
// numpy.fromfile reads the exact same bytes back — that's the whole interop.
void dump(const char *path, const float *data, int rows, int cols) {
  FILE *f = fopen(path, "wb");
  fwrite(&rows, sizeof(int), 1, f);
  fwrite(&cols, sizeof(int), 1, f);
  fwrite(data, sizeof(float), (size_t)rows * cols, f);
  fclose(f);
}

void bench(int n, int k, int m, bool print = true, bool verify = false) {
  float *A = new float[n * k];
  float *B = new float[k * m];
  if (verify) {
    // Random inputs so the oracle actually tests indexing/transpose bugs.
    for (int i = 0; i < n * k; i++)
      A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < k * m; i++)
      B[i] = (float)rand() / RAND_MAX;
  } else {
    for (int i = 0; i < n * k; i++)
      A[i] = 1;
    for (int i = 0; i < k * m; i++)
      B[i] = 1;
  }

  float *OUT = new float[n * m];
  std::memset(OUT, 0, n * m * sizeof(float));
  matmul_alloc(A, B, n, k, m);

  auto start = std::chrono::steady_clock::now();
  matmul_gpu_cublas(n, k, m);
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  matmul_get(OUT, n, m);

  if (verify) {
    // Dump for oracle.py (host arrays still valid after the device->host copy).
    dump("A.bin", A, n, k);
    dump("B.bin", B, k, m);
    dump("C.bin", OUT, n, m);
  }

  long double FLOPS = 2.0 * n * k * m / duration.count() * 1e6;
  std::string unit = "FLOP/s";
  if (FLOPS >= 1e12) {
    FLOPS /= 1e12;
    unit = "TFLOP/s";
  } else if (FLOPS >= 1e9) {
    FLOPS /= 1e9;
    unit = "GFLOP/s";
  } else if (FLOPS >= 1e6) {
    FLOPS /= 1e6;
    unit = "MFLOP/s";
  } else if (FLOPS >= 1e3) {
    FLOPS /= 1e3;
    unit = "KFLOP/s";
  }

  if (print) {
    std::cout << "(" << n << " X " << k << ") @ (" << k << " X " << m << ") took " << duration << " AKA " << FLOPS
              << " " << unit << std::endl;
  }

  // std::cout << OUT[0] << " " << OUT[1] << std::endl;
  if (!verify) {
    // Cheap all-ones smoke test; the oracle (verify=true) does the real check.
    for (int i = 0; i < n * m; i++) {
      if (OUT[i] != k) {
        throw std::runtime_error("matmul returned wrong anwser");
      }
    }
  }
}

int main() {
  // Just a dummy benchmark
  // bench(43, 93, 103, false, true); // verify=true -> random inputs + dump for oracle.py
  bench(1024, 1024, 1024, false, true);

  // Actual ones
  bench(16384, 1024, 16384);
  bench(2048, 1024, 2048);
  bench(1024, 16384, 2048); // Maybe? Just  GPU
  bench(16384, 16384, 16384);
  bench(16384, 32768, 16384); // Oh? Reg tiling
  bench(32768, 32768, 32768); // Oh? Reg tiling
  /* Bigger Faster Better Future
   */
  return 0;
}
