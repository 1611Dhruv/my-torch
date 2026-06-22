// Spec-by-tests for torch::matmul, across CPU, CUDA, and the device dispatcher.
//
// The oracle here is a plain triple-loop reference matmul (reference_matmul):
// it's the trusted, obviously-correct implementation, and every kernel result
// is checked against it. To keep float comparisons exact, inputs are small
// integer-valued floats -- products and partial sums stay well inside the
// 2^24 exact-integer range, so summation order (which differs between the
// naive reference and the tiled GPU kernel) can't introduce drift.
//
// CudaMatmulTest cases launch the real kernel and require a CUDA device at
// runtime. Data is staged host->device with cudaMemcpy (no .to(device) yet).
// The non-divisible sizes (67, 70x45x33) intentionally straddle the 32-wide
// tile so the kernel's edge guards get exercised.
//
// Run:  ctest --test-dir build -R Matmul          (all)
//       ctest --test-dir build -R CudaMatmul      (GPU only)

#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include "mytorch/tensor.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using torch::CPU;
using torch::CUDA;
using torch::DType;
using torch::Tensor;
using Shape = std::vector<int64_t>;

// --- helpers ---------------------------------------------------------------

// Deterministic, varied, small-integer data (range [-4, 4]) over flat index.
// Negatives included so sign mistakes in the kernel don't slip through.
static std::vector<float> gen(int64_t n) {
  std::vector<float> v(n);
  for (int64_t p = 0; p < n; ++p)
    v[p] = static_cast<float>((p % 9) - 4);
  return v;
}

// The oracle: naive row-major matmul, C[MxN] = A[MxK] * B[KxN].
static std::vector<float> reference_matmul(const std::vector<float> &A, const std::vector<float> &B, int64_t M,
                                           int64_t K, int64_t N) {
  std::vector<float> C(M * N, 0.0f);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = s;
    }
  return C;
}

// Build a contiguous CPU float tensor from flat row-major values.
static Tensor make_cpu(const Shape &shape, const std::vector<float> &vals) {
  Tensor t(shape, DType::Float32, CPU);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  float *d = t.data_ptr<float>();
  for (size_t p = 0; p < vals.size(); ++p)
    d[p] = vals[p];
  return t;
}

// Assert a contiguous float tensor matches `want` elementwise.
static void expect_close(Tensor &got, const std::vector<float> &want) {
  ASSERT_EQ(static_cast<int64_t>(want.size()), got.numel());
  ASSERT_TRUE(got.is_contiguous());
  const float *g = got.data_ptr<float>();
  for (size_t p = 0; p < want.size(); ++p)
    EXPECT_NEAR(g[p], want[p], 1e-3f);
}

// Stage A,B to device, run torch::cuda::matmul, copy the result back to host.
static std::vector<float> run_cuda_matmul(const std::vector<float> &hA, const std::vector<float> &hB, int64_t M,
                                          int64_t K, int64_t N) {
  Tensor a({M, K}, DType::Float32, CUDA);
  Tensor b({K, N}, DType::Float32, CUDA);
  CUDA_CHECK(cudaMemcpy(a.data_ptr<float>(), hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data_ptr<float>(), hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  Tensor c = torch::cuda::matmul(a, b);
  EXPECT_EQ(c.shape(), Shape({M, N}));

  std::vector<float> hC(M * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), c.data_ptr<float>(), M * N * sizeof(float), cudaMemcpyDeviceToHost));
  return hC;
}

// ===========================================================================
// CPU matmul
// ===========================================================================

TEST(MatmulCpuTest, SquareKnownValues) {
  // 2x2 * 2x2 with hand-checkable numbers.
  // [1 2] [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
  // [3 4]*[7 8] = [3*5+4*7  3*6+4*8] = [43 50]
  Tensor a = make_cpu({2, 2}, {1, 2, 3, 4});
  Tensor b = make_cpu({2, 2}, {5, 6, 7, 8});
  Tensor out = torch::cpu::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({2, 2}));
  expect_close(out, {19, 22, 43, 50});
}

TEST(MatmulCpuTest, RectangularMatchesOracle) {
  // Non-square, distinct M, K, N -- catches row/col/stride index swaps.
  const int64_t M = 2, K = 3, N = 4;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({M, K}, ha);
  Tensor b = make_cpu({K, N}, hb);
  Tensor out = torch::cpu::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({M, N}));
  expect_close(out, reference_matmul(ha, hb, M, K, N));
}

TEST(MatmulCpuTest, IdentityIsNoOp) {
  // A * I == A.
  const int64_t M = 3, K = 3;
  auto ha = gen(M * K);
  std::vector<float> id(K * K, 0.0f);
  for (int64_t d = 0; d < K; ++d)
    id[d * K + d] = 1.0f;
  Tensor a = make_cpu({M, K}, ha);
  Tensor i = make_cpu({K, K}, id);
  Tensor out = torch::cpu::matmul(a, i);
  expect_close(out, ha);
}

TEST(MatmulCpuTest, LargerNonDivisibleMatchesOracle) {
  const int64_t M = 17, K = 23, N = 13;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({M, K}, ha);
  Tensor b = make_cpu({K, N}, hb);
  Tensor out = torch::cpu::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({M, N}));
  expect_close(out, reference_matmul(ha, hb, M, K, N));
}

TEST(MatmulCpuTest, DoesNotMutateInputs) {
  const int64_t M = 2, K = 3, N = 2;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({M, K}, ha);
  Tensor b = make_cpu({K, N}, hb);
  (void)torch::cpu::matmul(a, b);
  expect_close(a, ha);
  expect_close(b, hb);
}

// ===========================================================================
// CUDA matmul  (requires a device; sizes straddle the 32-wide tile)
// ===========================================================================

TEST(CudaMatmulTest, SquareWithinOneTile) {
  // 4x4: fits in a single 32x32 tile, no K-loop iteration past the first.
  const int64_t M = 4, K = 4, N = 4;
  auto ha = gen(M * K), hb = gen(K * N);
  auto got = run_cuda_matmul(ha, hb, M, K, N);
  auto want = reference_matmul(ha, hb, M, K, N);
  for (int64_t p = 0; p < M * N; ++p)
    EXPECT_NEAR(got[p], want[p], 1e-3f);
}

TEST(CudaMatmulTest, MultiTileSquare) {
  // 64x64: exactly 2x2 tiles, clean multiple of TILE -- multi-K-tile path.
  const int64_t M = 64, K = 64, N = 64;
  auto ha = gen(M * K), hb = gen(K * N);
  auto got = run_cuda_matmul(ha, hb, M, K, N);
  auto want = reference_matmul(ha, hb, M, K, N);
  for (int64_t p = 0; p < M * N; ++p)
    EXPECT_NEAR(got[p], want[p], 1e-3f);
}

TEST(CudaMatmulTest, NonDivisibleSquareExercisesEdgeGuards) {
  // 67x67x67: none of the dims are a multiple of 32, so every boundary tile
  // has out-of-range loads. This is THE edge-guard test.
  const int64_t M = 67, K = 67, N = 67;
  auto ha = gen(M * K), hb = gen(K * N);
  auto got = run_cuda_matmul(ha, hb, M, K, N);
  auto want = reference_matmul(ha, hb, M, K, N);
  for (int64_t p = 0; p < M * N; ++p)
    EXPECT_NEAR(got[p], want[p], 1e-3f);
}

TEST(CudaMatmulTest, RectangularNonDivisible) {
  // Distinct, non-divisible M, K, N -- ragged edges on all three dims at once.
  const int64_t M = 70, K = 45, N = 33;
  auto ha = gen(M * K), hb = gen(K * N);
  auto got = run_cuda_matmul(ha, hb, M, K, N);
  auto want = reference_matmul(ha, hb, M, K, N);
  for (int64_t p = 0; p < M * N; ++p)
    EXPECT_NEAR(got[p], want[p], 1e-3f);
}

TEST(CudaMatmulTest, ThrowsOnNonContiguous) {
  Tensor a({4, 4}, DType::Float32, CUDA);
  Tensor b({4, 4}, DType::Float32, CUDA);
  EXPECT_THROW(torch::cuda::matmul(a, b.transpose(0, 1)), std::invalid_argument);
}

// ===========================================================================
// Dispatcher  (torch::matmul -- validation + routing)
// ===========================================================================

TEST(MatmulDispatchTest, ValidCpuShapesDoNotThrow) {
  // The canary for the inverted inner-dim check: a *valid* 2x3 * 3x4 must NOT
  // throw. (If dispatch.cpp uses `==` instead of `!=`, this fails here.)
  Tensor a({2, 3}, DType::Float32, CPU);
  Tensor b({3, 4}, DType::Float32, CPU);
  EXPECT_NO_THROW(torch::matmul(a, b));
}

TEST(MatmulDispatchTest, RoutesCpuAndComputes) {
  // End-to-end through the dispatcher on CPU, checked against the oracle.
  const int64_t M = 3, K = 4, N = 2;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({M, K}, ha);
  Tensor b = make_cpu({K, N}, hb);
  Tensor out = torch::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({M, N}));
  expect_close(out, reference_matmul(ha, hb, M, K, N));
}

TEST(MatmulDispatchTest, InnerDimMismatchThrows) {
  // 2x3 * 4x5: inner dims 3 != 4 -> invalid.
  Tensor a({2, 3}, DType::Float32, CPU);
  Tensor b({4, 5}, DType::Float32, CPU);
  EXPECT_THROW(torch::matmul(a, b), std::invalid_argument);
}

TEST(MatmulDispatchTest, NonMatrixRankThrows) {
  // 1-D and 3-D operands are not matrices.
  Tensor a({3}, DType::Float32, CPU);
  Tensor b({3, 3}, DType::Float32, CPU);
  EXPECT_THROW(torch::matmul(a, b), std::invalid_argument);

  Tensor c({2, 3, 4}, DType::Float32, CPU);
  Tensor d({4, 2}, DType::Float32, CPU);
  EXPECT_THROW(torch::matmul(c, d), std::invalid_argument);
}

TEST(MatmulDispatchTest, DtypeMismatchThrows) {
  Tensor a({2, 2}, DType::Float32, CPU);
  Tensor b({2, 2}, DType::Int32, CPU);
  EXPECT_THROW(torch::matmul(a, b), std::invalid_argument);
}

TEST(MatmulDispatchTest, DeviceMismatchThrows) {
  Tensor a({2, 2}, DType::Float32, CPU);
  Tensor b({2, 2}, DType::Float32, CUDA);
  EXPECT_THROW(torch::matmul(a, b), std::invalid_argument);
}

TEST(MatmulDispatchTest, RoutesCudaAndComputes) {
  // Valid CUDA operands route to the GPU kernel and match the oracle.
  const int64_t M = 33, K = 40, N = 17;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a({M, K}, DType::Float32, CUDA);
  Tensor b({K, N}, DType::Float32, CUDA);
  CUDA_CHECK(cudaMemcpy(a.data_ptr<float>(), ha.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data_ptr<float>(), hb.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  Tensor c = torch::matmul(a, b);
  ASSERT_EQ(c.shape(), Shape({M, N}));
  std::vector<float> hC(M * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), c.data_ptr<float>(), M * N * sizeof(float), cudaMemcpyDeviceToHost));

  auto want = reference_matmul(ha, hb, M, K, N);
  for (int64_t p = 0; p < M * N; ++p)
    EXPECT_NEAR(hC[p], want[p], 1e-3f);
}
