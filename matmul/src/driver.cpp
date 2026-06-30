// Benchmark + correctness harness for the hand-written matmul vs cuBLAS.
//
//   Section 1 (ladder): every kernel naive -> ... -> double-buf at one size,
//                       each timed and checked against cuBLAS.
//   Section 2 (sweep):  the final kernel vs cuBLAS across square sizes.
//
// The 1024^3 sweep case also dumps A/B/C.bin so bench/oracle.py can do the
// definitive float64 (PyTorch) check.

#include "matmul.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using clk = std::chrono::steady_clock;
using kernel_fn = void (*)(int, int, int);

static void dump(const char *path, const float *data, int rows, int cols) {
  FILE *f = fopen(path, "wb");
  fwrite(&rows, sizeof(int), 1, f);
  fwrite(&cols, sizeof(int), 1, f);
  fwrite(data, sizeof(float), (size_t)rows * cols, f);
  fclose(f);
}

// Average microseconds per call, after one warmup (each call syncs internally).
static double time_call(kernel_fn fn, int n, int k, int m, int iters) {
  fn(n, k, m); // warmup: JIT / cuBLAS handle / first-touch
  auto t0 = clk::now();
  for (int i = 0; i < iters; i++)
    fn(n, k, m);
  auto t1 = clk::now();
  double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  return us / iters;
}

static double tflops(int n, int k, int m, double us) {
  return 2.0 * n * k * m / us / 1e6;
}

// max relative error of `got` vs `ref`
static double max_rel(const float *got, const float *ref, size_t n) {
  double mr = 0;
  for (size_t i = 0; i < n; i++) {
    double d = std::fabs((double)got[i] - (double)ref[i]);
    double r = d / (std::fabs((double)ref[i]) + 1e-6);
    if (r > mr)
      mr = r;
  }
  return mr;
}

static void fill_random(float *p, size_t n) {
  for (size_t i = 0; i < n; i++)
    p[i] = (float)rand() / RAND_MAX;
}

struct Kern {
  const char *name;
  kernel_fn fn;
};

// Section 1: run every kernel at one size, compare each to cuBLAS.
static void ladder(int n, int k, int m, int iters) {
  size_t na = (size_t)n * k, nb = (size_t)k * m, nc = (size_t)n * m;
  float *A = new float[na], *B = new float[nb];
  float *Cmine = new float[nc], *Cref = new float[nc];
  fill_random(A, na);
  fill_random(B, nb);

  matmul_alloc(A, B, n, k, m);

  double us_ref = time_call(matmul_gpu_cublas, n, k, m, iters);
  matmul_copyout(Cref, n, m);
  double tf_ref = tflops(n, k, m, us_ref);

  Kern kerns[] = {
      {"0  naive", run_naive},
      {"1  shared tiling", run_shared},
      {"2  register tiling (8x8)", run_reg},
      {"3  + float4", run_float4},
      {"4  + warp tiling", run_warptile},
      {"4b + transpose-As", run_warptile_T},
      {"5  + double buffering", matmul_gpu},
  };

  printf("\n== ladder @ %d^3 (each kernel vs cuBLAS) ==\n", n);
  printf("  kernel                     TFLOP/s   %%cuBLAS   maxrel\n");
  printf("  ------------------------   -------   -------   --------\n");
  for (Kern kr : kerns) {
    double us = time_call(kr.fn, n, k, m, iters);
    matmul_copyout(Cmine, n, m);
    double tf = tflops(n, k, m, us);
    double mr = max_rel(Cmine, Cref, nc);
    printf("  %-24s   %7.1f   %6.1f%%   %.1e %s\n", kr.name, tf, 100.0 * tf / tf_ref, mr,
           mr < 1e-2 ? "" : " <-- MISMATCH");
  }
  printf("  %-24s   %7.1f   %6.1f%%   %.1e\n", "   cuBLAS", tf_ref, 100.0, 0.0);

  matmul_free();
  delete[] A;
  delete[] B;
  delete[] Cmine;
  delete[] Cref;
}

// Section 2: final kernel vs cuBLAS across sizes.
static void sweep_point(int n, int k, int m, int iters, bool dump_oracle) {
  size_t na = (size_t)n * k, nb = (size_t)k * m, nc = (size_t)n * m;
  float *A = new float[na], *B = new float[nb];
  float *Cmine = new float[nc], *Cref = new float[nc];
  fill_random(A, na);
  fill_random(B, nb);

  matmul_alloc(A, B, n, k, m);
  double us_mine = time_call(matmul_gpu, n, k, m, iters);
  matmul_copyout(Cmine, n, m);
  double us_ref = time_call(matmul_gpu_cublas, n, k, m, iters);
  matmul_copyout(Cref, n, m);
  matmul_free();

  double mr = max_rel(Cmine, Cref, nc);
  double tf_mine = tflops(n, k, m, us_mine), tf_ref = tflops(n, k, m, us_ref);
  printf("%6d  %8.1f  %8.1f   %5.1f%%   maxrel=%.1e   %s\n", n, tf_mine, tf_ref,
         100.0 * tf_mine / tf_ref, mr, mr < 1e-2 ? "PASS" : "FAIL");

  if (dump_oracle) {
    dump("A.bin", A, n, k);
    dump("B.bin", B, k, m);
    dump("C.bin", Cmine, n, m);
  }
  delete[] A;
  delete[] B;
  delete[] Cmine;
  delete[] Cref;
}

int main() {
  // Section 1: full ladder at a representative size (small enough that naive finishes).
  ladder(4096, 4096, 4096, 3);

  // Section 2: final kernel across sizes.
  printf("\n== final kernel vs cuBLAS, by size ==\n");
  printf("  size     yours    cuBLAS   %%ceil   correctness(vs cuBLAS)\n");
  printf("  ----   -------   -------   ------   ----------------------\n");
  int sizes[] = {512, 1024, 2048, 4096, 8192, 16384};
  for (int s : sizes) {
    int iters = s <= 2048 ? 20 : (s <= 8192 ? 5 : 2);
    sweep_point(s, s, s, iters, /*dump_oracle=*/s == 1024);
  }
  return 0;
}
