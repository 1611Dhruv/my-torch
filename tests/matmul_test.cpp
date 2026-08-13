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
//
// The flat index is hashed rather than used directly. A plain `p % 9` makes
// accidentally *symmetric* square matrices: for an RxR matrix, element (i,j)
// is at p = i*R + j, so `p % 9` depends only on (i*R + j) % 9 -- and whenever
// R % 9 == 1 (R = 64, 100, 190, ...) that collapses to (i + j) % 9, which is
// symmetric in i and j. A transposed operand would then be indistinguishable
// from a contiguous one and a broken transpose path would silently pass.
static std::vector<float> gen(int64_t n) {
  std::vector<float> v(n);
  for (int64_t p = 0; p < n; ++p) {
    uint32_t h = static_cast<uint32_t>(p) * 2654435761u; // Knuth multiplicative
    v[p] = static_cast<float>(static_cast<int32_t>(h % 9u) - 4);
  }
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

// Host-side transpose of a row-major RxC buffer into a row-major CxR one.
static std::vector<float> host_transpose(const std::vector<float> &v, int64_t R, int64_t C) {
  std::vector<float> t(v.size());
  for (int64_t i = 0; i < R; ++i)
    for (int64_t j = 0; j < C; ++j)
      t[j * R + i] = v[i * C + j];
  return t;
}

// Upload a row-major host buffer as a contiguous CUDA tensor of `shape`.
static Tensor make_cuda(const Shape &shape, const std::vector<float> &vals) {
  Tensor t(shape, DType::Float32, CUDA);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  CUDA_CHECK(cudaMemcpy(t.data_ptr<float>(), vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice));
  return t;
}

// Copy a contiguous device tensor back to host.
static std::vector<float> to_host(const Tensor &t) {
  std::vector<float> h(t.numel());
  CUDA_CHECK(cudaMemcpy(h.data(), t.data_ptr<float>(), h.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

// A CUDA operand whose *logical* value is `logical` (row-major RxC), built one
// of two ways:
//   via_transpose = false -> a contiguous {R,C} tensor, strides {C,1}
//   via_transpose = true  -> a {C,R} tensor viewed through transpose(0,1),
//                            so shape is {R,C} but strides are {1,R}
// Both must matmul to the same answer; only the kernel's load path differs.
static Tensor cuda_operand(const std::vector<float> &logical, int64_t R, int64_t C, bool via_transpose) {
  if (!via_transpose)
    return make_cuda({R, C}, logical);
  Tensor storage = make_cuda({C, R}, host_transpose(logical, R, C));
  Tensor view = storage.transpose(0, 1);
  EXPECT_EQ(view.shape(), Shape({R, C}));
  EXPECT_FALSE(view.is_contiguous());
  return view;
}

// Run A[MxK] * B[KxN] on CUDA with either operand optionally handed in as a
// transposed view, and check the result against the oracle.
static void check_cuda_matmul(int64_t M, int64_t K, int64_t N, bool a_view, bool b_view) {
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = cuda_operand(ha, M, K, a_view);
  Tensor b = cuda_operand(hb, K, N, b_view);

  Tensor c = torch::matmul(a, b);
  ASSERT_EQ(c.shape(), Shape({M, N}));
  ASSERT_TRUE(c.is_contiguous());

  auto got = to_host(c);
  auto want = reference_matmul(ha, hb, M, K, N);
  for (int64_t p = 0; p < M * N; ++p)
    EXPECT_NEAR(got[p], want[p], 1e-3f) << "at flat index " << p << " (M=" << M << " K=" << K << " N=" << N
                                        << " a_view=" << a_view << " b_view=" << b_view << ")";
}

// Stage A,B to device, run torch::matmul, copy the result back to host.
static std::vector<float> run_cuda_matmul(const std::vector<float> &hA, const std::vector<float> &hB, int64_t M,
                                          int64_t K, int64_t N) {
  Tensor a({M, K}, DType::Float32, CUDA);
  Tensor b({K, N}, DType::Float32, CUDA);
  CUDA_CHECK(cudaMemcpy(a.data_ptr<float>(), hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data_ptr<float>(), hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

  Tensor c = torch::matmul(a, b);
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
  Tensor out = torch::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({2, 2}));
  expect_close(out, {19, 22, 43, 50});
}

TEST(MatmulCpuTest, RectangularMatchesOracle) {
  // Non-square, distinct M, K, N -- catches row/col/stride index swaps.
  const int64_t M = 2, K = 3, N = 4;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({M, K}, ha);
  Tensor b = make_cpu({K, N}, hb);
  Tensor out = torch::matmul(a, b);
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
  Tensor out = torch::matmul(a, i);
  expect_close(out, ha);
}

TEST(MatmulCpuTest, LargerNonDivisibleMatchesOracle) {
  const int64_t M = 17, K = 23, N = 13;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({M, K}, ha);
  Tensor b = make_cpu({K, N}, hb);
  Tensor out = torch::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({M, N}));
  expect_close(out, reference_matmul(ha, hb, M, K, N));
}

TEST(MatmulCpuTest, DoesNotMutateInputs) {
  const int64_t M = 2, K = 3, N = 2;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({M, K}, ha);
  Tensor b = make_cpu({K, N}, hb);
  (void)torch::matmul(a, b);
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

// ===========================================================================
// CUDA matmul on transposed views
//
// The kernel no longer requires contiguous operands: it picks a
// matmul_kernel<a_transp, b_transp> specialization and loads the transposed
// operand with swapped row/col indexing. These tests pin that down.
//
// The invariant every case checks: a transposed *view* of the flipped data and
// a contiguous tensor of the same logical values must produce identical
// results. Only the shared-memory load path differs.
//
// Sizes matter here. The transposed load walks memory with a different stride
// than the contiguous one, so a size that's a clean multiple of the block tile
// (BM=BN=128, BK=4) can hide a guard bug that a ragged size exposes.
// ===========================================================================

TEST(CudaMatmulTransposeTest, NeitherTransposedIsTheBaseline) {
  // Sanity anchor: same helper, no views. If this fails, the harness is wrong,
  // not the transpose handling.
  check_cuda_matmul(/*M=*/33, /*K=*/40, /*N=*/17, /*a_view=*/false, /*b_view=*/false);
}

TEST(CudaMatmulTransposeTest, ATransposedSquare) {
  check_cuda_matmul(64, 64, 64, /*a_view=*/true, /*b_view=*/false);
}

TEST(CudaMatmulTransposeTest, BTransposedSquare) {
  check_cuda_matmul(64, 64, 64, /*a_view=*/false, /*b_view=*/true);
}

TEST(CudaMatmulTransposeTest, BothTransposedSquare) {
  check_cuda_matmul(64, 64, 64, /*a_view=*/true, /*b_view=*/true);
}

TEST(CudaMatmulTransposeTest, ATransposedRectangularDistinctDims) {
  // M != K != N so a row/col swap in the transposed load path can't cancel out.
  check_cuda_matmul(24, 40, 56, /*a_view=*/true, /*b_view=*/false);
}

TEST(CudaMatmulTransposeTest, BTransposedRectangularDistinctDims) {
  check_cuda_matmul(24, 40, 56, /*a_view=*/false, /*b_view=*/true);
}

TEST(CudaMatmulTransposeTest, BothTransposedRectangularDistinctDims) {
  check_cuda_matmul(24, 40, 56, /*a_view=*/true, /*b_view=*/true);
}

TEST(CudaMatmulTransposeTest, ATransposedNonDivisibleEdgeGuards) {
  // 67x67x67: no dim is a multiple of BM/BN/BK, so the transposed load path
  // hits out-of-range reads on every boundary tile.
  check_cuda_matmul(67, 67, 67, /*a_view=*/true, /*b_view=*/false);
}

TEST(CudaMatmulTransposeTest, BTransposedNonDivisibleEdgeGuards) {
  check_cuda_matmul(67, 67, 67, /*a_view=*/false, /*b_view=*/true);
}

TEST(CudaMatmulTransposeTest, BothTransposedNonDivisibleEdgeGuards) {
  check_cuda_matmul(67, 67, 67, /*a_view=*/true, /*b_view=*/true);
}

TEST(CudaMatmulTransposeTest, BothTransposedRaggedDistinctDims) {
  // The nastiest combination: ragged on all three dims, both operands strided.
  check_cuda_matmul(70, 45, 33, /*a_view=*/true, /*b_view=*/true);
}

TEST(CudaMatmulTransposeTest, TransposedMatchesMaterializedContiguous) {
  // The equivalence stated directly: x.transpose(0,1) and
  // x.transpose(0,1).contiguous() are the same tensor logically, so they must
  // matmul to bit-identical results (same values, same accumulation order --
  // the kernel differs only in how it fills shared memory).
  const int64_t M = 70, K = 45, N = 33;
  auto ha = gen(M * K), hb = gen(K * N);

  Tensor a_view = cuda_operand(ha, M, K, /*via_transpose=*/true);
  Tensor b_view = cuda_operand(hb, K, N, /*via_transpose=*/true);
  Tensor a_flat = make_cuda({M, K}, ha);
  Tensor b_flat = make_cuda({K, N}, hb);

  auto strided = to_host(torch::matmul(a_view, b_view));
  auto materialized = to_host(torch::matmul(a_flat, b_flat));
  ASSERT_EQ(strided.size(), materialized.size());
  for (size_t p = 0; p < strided.size(); ++p)
    EXPECT_FLOAT_EQ(strided[p], materialized[p]) << "at flat index " << p;
}

TEST(CudaMatmulTransposeTest, DoubleTransposeIsContiguousAgain) {
  // transpose(0,1) twice restores the original strides, so this must take the
  // plain (non-transposed) kernel path -- a regression guard on the
  // `!is_contiguous()` inference in cuda::matmul.
  Tensor a = make_cuda({8, 8}, gen(64));
  Tensor twice = a.transpose(0, 1).transpose(0, 1);
  ASSERT_TRUE(twice.is_contiguous());
  ASSERT_EQ(twice.shape(), Shape({8, 8}));

  auto ha = gen(64), hb = gen(64);
  Tensor b = make_cuda({8, 8}, hb);
  Tensor a2 = make_cuda({8, 8}, ha);
  auto got = to_host(torch::matmul(a2.transpose(0, 1).transpose(0, 1), b));
  auto want = reference_matmul(ha, hb, 8, 8, 8);
  for (size_t p = 0; p < want.size(); ++p)
    EXPECT_NEAR(got[p], want[p], 1e-3f);
}

TEST(CudaMatmulTransposeTest, TransposeIsNotMutatedByMatmul) {
  // The kernel must only read through the view; the backing storage of a
  // transposed operand stays untouched.
  const int64_t M = 12, K = 20, N = 8;
  auto ha = gen(M * K);
  auto backing = host_transpose(ha, M, K); // what lives in storage as {K,M}
  Tensor storage = make_cuda({K, M}, backing);
  Tensor a = storage.transpose(0, 1);
  Tensor b = make_cuda({K, N}, gen(K * N));

  (void)torch::matmul(a, b);
  EXPECT_EQ(to_host(storage), backing);
}

TEST(CudaMatmulTransposeTest, DispatcherAcceptsTransposedOperands) {
  // torch::matmul validates on *logical* shape, so a transposed view with
  // matching inner dims must route through rather than throw.
  const int64_t M = 33, K = 40, N = 17;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = cuda_operand(ha, M, K, /*via_transpose=*/true);
  Tensor b = cuda_operand(hb, K, N, /*via_transpose=*/true);

  Tensor c = torch::matmul(a, b);
  ASSERT_EQ(c.shape(), Shape({M, N}));
  auto got = to_host(c);
  auto want = reference_matmul(ha, hb, M, K, N);
  for (int64_t p = 0; p < M * N; ++p)
    EXPECT_NEAR(got[p], want[p], 1e-3f);
}

TEST(CudaMatmulTransposeTest, DispatcherStillChecksLogicalInnerDims) {
  // Transposing must not smuggle a shape error past validation: a {2,3} view
  // of a {3,2} tensor times a {4,5} is still 3 != 4.
  Tensor a = Tensor({3, 2}, DType::Float32, CUDA).transpose(0, 1); // logical {2,3}
  Tensor b({4, 5}, DType::Float32, CUDA);
  EXPECT_THROW(torch::matmul(a, b), std::invalid_argument);
}

// ===========================================================================
// CPU matmul on transposed views
//
// cpu::matmul normalizes with .contiguous() rather than reading strides, so a
// transposed operand is legal now -- it just costs a copy. These tests pin the
// *result*, which is what callers care about; the normalization strategy is
// free to change to stride-aware indexing later without touching them.
// ===========================================================================

TEST(CpuMatmulTransposeTest, TransposedMatchesOracle) {
  const int64_t M = 5, K = 7, N = 3;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({K, M}, host_transpose(ha, M, K)).transpose(0, 1);
  Tensor b = make_cpu({N, K}, host_transpose(hb, K, N)).transpose(0, 1);
  ASSERT_FALSE(a.is_contiguous());
  ASSERT_FALSE(b.is_contiguous());

  Tensor out = torch::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({M, N}));
  expect_close(out, reference_matmul(ha, hb, M, K, N));
}

TEST(CpuMatmulTransposeTest, OnlyATransposedMatchesOracle) {
  const int64_t M = 5, K = 7, N = 3;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({K, M}, host_transpose(ha, M, K)).transpose(0, 1);
  Tensor b = make_cpu({K, N}, hb);
  Tensor out = torch::matmul(a, b);
  expect_close(out, reference_matmul(ha, hb, M, K, N));
}

TEST(CpuMatmulTransposeTest, OnlyBTransposedMatchesOracle) {
  const int64_t M = 5, K = 7, N = 3;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({M, K}, ha);
  Tensor b = make_cpu({N, K}, host_transpose(hb, K, N)).transpose(0, 1);
  Tensor out = torch::matmul(a, b);
  expect_close(out, reference_matmul(ha, hb, M, K, N));
}

TEST(CpuMatmulTransposeTest, NormalizationDoesNotMutateTheInput) {
  // .contiguous() must copy, not rewrite the view's storage in place.
  const int64_t M = 4, K = 6, N = 3;
  auto ha = gen(M * K);
  auto backing = host_transpose(ha, M, K); // lives in storage as {K,M}
  Tensor storage = make_cpu({K, M}, backing);
  Tensor a = storage.transpose(0, 1);
  Tensor b = make_cpu({K, N}, gen(K * N));

  (void)torch::matmul(a, b);
  expect_close(storage, backing);
  EXPECT_FALSE(a.is_contiguous()); // the view itself is untouched too
}

TEST(CpuMatmulTransposeTest, DispatcherRoutesTransposedCpuOperands) {
  const int64_t M = 5, K = 7, N = 3;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor a = make_cpu({M, K}, ha);
  Tensor b = make_cpu({N, K}, host_transpose(hb, K, N)).transpose(0, 1);
  Tensor out = torch::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({M, N}));
  expect_close(out, reference_matmul(ha, hb, M, K, N));
}

// ===========================================================================
// Batched CPU matmul
//
// cpu::matmul peels the trailing two dims, collapses the leading ones into a
// single B, runs B independent matmuls, and reshapes back to
// batch_dims ++ {M, N}. Two things need pinning: every batch slab must be the
// right product (not just batch 0), and the *logical* output rank must survive
// the round trip through the collapsed {B,M,N} form.
//
// Batch dims must currently match exactly -- broadcasting {B,M,K} @ {K,N} is
// not implemented, and that gap is asserted below rather than left silent.
// ===========================================================================

// Oracle: B independent matmuls over contiguous {B,M,K} x {B,K,N} buffers.
static std::vector<float> reference_batched_matmul(const std::vector<float> &A, const std::vector<float> &B_, int64_t B,
                                                  int64_t M, int64_t K, int64_t N) {
  std::vector<float> C;
  C.reserve(B * M * N);
  for (int64_t b = 0; b < B; ++b) {
    std::vector<float> a_slab(A.begin() + b * M * K, A.begin() + (b + 1) * M * K);
    std::vector<float> b_slab(B_.begin() + b * K * N, B_.begin() + (b + 1) * K * N);
    auto c_slab = reference_matmul(a_slab, b_slab, M, K, N);
    C.insert(C.end(), c_slab.begin(), c_slab.end());
  }
  return C;
}

TEST(BatchedCpuMatmulTest, Rank3MatchesPerBatchOracle) {
  // B=2 with distinct M,K,N. gen() varies over the flat index, so the two
  // slabs hold different data -- a kernel that ignores `b` fails here.
  const int64_t B = 2, M = 3, K = 4, N = 5;
  auto ha = gen(B * M * K), hb = gen(B * K * N);
  Tensor a = make_cpu({B, M, K}, ha);
  Tensor b = make_cpu({B, K, N}, hb);
  Tensor out = torch::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({B, M, N}));
  expect_close(out, reference_batched_matmul(ha, hb, B, M, K, N));
}

TEST(BatchedCpuMatmulTest, BatchOfOneEqualsPlainMatmul) {
  // {1,M,K} @ {1,K,N} must agree with the 2-D path and keep its rank-3 shape.
  const int64_t M = 3, K = 4, N = 5;
  auto ha = gen(M * K), hb = gen(K * N);
  Tensor out = torch::matmul(make_cpu({1, M, K}, ha), make_cpu({1, K, N}, hb));
  EXPECT_EQ(out.shape(), Shape({1, M, N}));
  expect_close(out, reference_matmul(ha, hb, M, K, N));
}

TEST(BatchedCpuMatmulTest, Rank4CollapsesAndRestoresLogicalShape) {
  // Two batch dims collapse to B=6 internally; the caller must still see
  // {2,3,M,N}, not {6,M,N}. This is the reshape-back test.
  const int64_t B0 = 2, B1 = 3, M = 3, K = 4, N = 2;
  const int64_t B = B0 * B1;
  auto ha = gen(B * M * K), hb = gen(B * K * N);
  Tensor a = make_cpu({B0, B1, M, K}, ha);
  Tensor b = make_cpu({B0, B1, K, N}, hb);
  Tensor out = torch::matmul(a, b);
  EXPECT_EQ(out.shape(), Shape({B0, B1, M, N}));
  expect_close(out, reference_batched_matmul(ha, hb, B, M, K, N));
}

TEST(BatchedCpuMatmulTest, BatchedTransposedOperandIsNormalized) {
  // The composition that matters: matmul's own backward is all
  // transpose(-1,-2) on batched tensors, so a per-batch-transposed view has to
  // survive both the .contiguous() copy and the collapse to {B,M,K}.
  const int64_t B = 2, M = 3, K = 4, N = 5;
  auto ha = gen(B * M * K), hb = gen(B * K * N);

  // Store A as {B,K,M} (each slab transposed), then view it back as {B,M,K}.
  std::vector<float> backing;
  backing.reserve(ha.size());
  for (int64_t s = 0; s < B; ++s) {
    std::vector<float> slab(ha.begin() + s * M * K, ha.begin() + (s + 1) * M * K);
    auto t = host_transpose(slab, M, K);
    backing.insert(backing.end(), t.begin(), t.end());
  }
  Tensor a = make_cpu({B, K, M}, backing).transpose(-1, -2);
  ASSERT_EQ(a.shape(), Shape({B, M, K}));
  ASSERT_FALSE(a.is_contiguous());

  Tensor out = torch::matmul(a, make_cpu({B, K, N}, hb));
  EXPECT_EQ(out.shape(), Shape({B, M, N}));
  expect_close(out, reference_batched_matmul(ha, hb, B, M, K, N));
}

TEST(BatchedCpuMatmulTest, MismatchedBatchDimsThrow) {
  Tensor a = make_cpu({2, 3, 4}, gen(24));
  Tensor b = make_cpu({3, 4, 5}, gen(60));
  EXPECT_THROW(torch::matmul(a, b), std::invalid_argument);
}

TEST(BatchedCpuMatmulTest, SameBatchProductDifferentBatchShapeThrows) {
  // {2,3,...} and {6,...} collapse to the same B=6. Comparing the batch dims
  // *before* collapsing is what rejects this; a product-only check would not.
  Tensor a = make_cpu({2, 3, 2, 2}, gen(24));
  Tensor b = make_cpu({6, 2, 2}, gen(24));
  EXPECT_THROW(torch::matmul(a, b), std::invalid_argument);
}

TEST(BatchedCpuMatmulTest, InnerDimCheckedOnTrailingDimsNotBatchDims) {
  // A shape bug that reads a batch dim instead of K hides whenever K happens
  // to equal a batch dim. Here K=4 and the batch dim is 4 as well, so an
  // operand pair that is genuinely invalid (K=4 vs 7) must still be rejected.
  Tensor a = make_cpu({4, 3, 4}, gen(48));
  Tensor b = make_cpu({4, 7, 5}, gen(140));
  EXPECT_THROW(torch::matmul(a, b), std::invalid_argument);
}

// --- not implemented yet: broadcasting a bare matrix over a batch -----------
//
// {B,M,K} @ {K,N} is the nn::Linear case and is the next thing to land (via a
// batch stride of 0 on the un-batched operand). Until then it must throw, not
// silently misread. Delete these two and enable the DISABLED_ ones below when
// broadcasting goes in.

TEST(BatchedCpuMatmulTest, Rank3TimesRank2ThrowsForNow) {
  Tensor a = make_cpu({2, 3, 4}, gen(24));
  Tensor b = make_cpu({4, 5}, gen(20));
  EXPECT_THROW(torch::matmul(a, b), std::invalid_argument);
}

TEST(BatchedCpuMatmulTest, DISABLED_BroadcastsBareMatrixOverBatch) {
  // The nn::Linear shape: x{B,T,in} @ W{in,out} -> {B,T,out}, same W reused.
  const int64_t B = 2, M = 3, K = 4, N = 5;
  auto ha = gen(B * M * K), hb = gen(K * N);
  Tensor out = torch::matmul(make_cpu({B, M, K}, ha), make_cpu({K, N}, hb));
  ASSERT_EQ(out.shape(), Shape({B, M, N}));

  std::vector<float> tiled; // the same B replicated across the batch
  for (int64_t s = 0; s < B; ++s)
    tiled.insert(tiled.end(), hb.begin(), hb.end());
  expect_close(out, reference_batched_matmul(ha, tiled, B, M, K, N));
}

TEST(BatchedCpuMatmulTest, DISABLED_BroadcastsBareMatrixOnTheLeft) {
  const int64_t B = 2, M = 3, K = 4, N = 5;
  auto ha = gen(M * K), hb = gen(B * K * N);
  Tensor out = torch::matmul(make_cpu({M, K}, ha), make_cpu({B, K, N}, hb));
  ASSERT_EQ(out.shape(), Shape({B, M, N}));

  std::vector<float> tiled;
  for (int64_t s = 0; s < B; ++s)
    tiled.insert(tiled.end(), ha.begin(), ha.end());
  expect_close(out, reference_batched_matmul(tiled, hb, B, M, K, N));
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

// ===========================================================================
// Batched CUDA matmul
//
// The kernel gets one launch with gridDim.z == B and offsets its three
// pointers by blockIdx.z. The offsets are *different per operand* --
// A advances by M*K, B by K*N, C by M*N -- so a single shared offset, or one
// derived from the tile constants (BM*BN) rather than the matrix dims, is
// wrong in a way only a B>1 test with distinct M,K,N can see.
//
// Every case below therefore uses B > 1 and M != K != N.
// ===========================================================================

// Stage contiguous {B,M,K} x {B,K,N} to device, matmul, check per-batch.
// If a_transposed, A is stored as {B,K,M} and viewed through transpose(-1,-2),
// so the strided load path and the batch offset have to compose correctly.
static void check_cuda_batched(int64_t B, int64_t M, int64_t K, int64_t N, bool a_transposed) {
  auto ha = gen(B * M * K), hb = gen(B * K * N);

  Tensor a = [&] {
    if (!a_transposed)
      return make_cuda({B, M, K}, ha);
    std::vector<float> backing;
    backing.reserve(ha.size());
    for (int64_t s = 0; s < B; ++s) {
      std::vector<float> slab(ha.begin() + s * M * K, ha.begin() + (s + 1) * M * K);
      auto t = host_transpose(slab, M, K);
      backing.insert(backing.end(), t.begin(), t.end());
    }
    Tensor view = make_cuda({B, K, M}, backing).transpose(-1, -2);
    EXPECT_EQ(view.shape(), Shape({B, M, K}));
    EXPECT_FALSE(view.is_contiguous());
    return view;
  }();

  Tensor c = torch::matmul(a, make_cuda({B, K, N}, hb));
  ASSERT_EQ(c.shape(), Shape({B, M, N}));

  auto got = to_host(c);
  auto want = reference_batched_matmul(ha, hb, B, M, K, N);
  ASSERT_EQ(got.size(), want.size());
  for (size_t p = 0; p < want.size(); ++p)
    EXPECT_NEAR(got[p], want[p], 1e-3f)
        << "batch " << (p / (M * N)) << ", offset " << (p % (M * N)) << " (B=" << B << " M=" << M << " K=" << K
        << " N=" << N << " a_transposed=" << a_transposed << ")";
}

TEST(CudaBatchedMatmulTest, BatchOfOneMatchesPlainMatmul) {
  // B=1 => blockIdx.z is always 0, so this passes even with a broken batch
  // offset. It's the control: if this fails, the bug is not in the batching.
  check_cuda_batched(/*B=*/1, /*M=*/70, /*K=*/45, /*N=*/33, /*a_transposed=*/false);
}

TEST(CudaBatchedMatmulTest, Rank3DistinctDimsMatchesPerBatchOracle) {
  // THE batch-offset test. M*K=3150, K*N=1485, M*N=2310 are all different from
  // each other and from BM*BN=16384, so a single or tile-derived offset can't
  // survive. Dims are also non-divisible, keeping the edge guards live.
  check_cuda_batched(/*B=*/3, /*M=*/70, /*K=*/45, /*N=*/33, /*a_transposed=*/false);
}

TEST(CudaBatchedMatmulTest, SmallMatricesManyBatches) {
  // Matrices far smaller than one 128x128 block tile, so every batch is a
  // single mostly-masked block. Catches an offset that scales with the tile
  // size instead of the matrix size.
  check_cuda_batched(/*B=*/8, /*M=*/5, /*K=*/7, /*N=*/3, /*a_transposed=*/false);
}

TEST(CudaBatchedMatmulTest, MultiTileWithBatches) {
  // Larger than one block tile in both M and N *and* batched, so blockIdx.x,
  // .y and .z are all non-trivial at once.
  check_cuda_batched(/*B=*/2, /*M=*/200, /*K=*/45, /*N=*/150, /*a_transposed=*/false);
}

TEST(CudaBatchedMatmulTest, BatchedTransposedOperand) {
  // The shape autograd actually produces: transpose(-1,-2) on a batched
  // tensor. The strided load path and the batch offset must compose.
  check_cuda_batched(/*B=*/3, /*M=*/70, /*K=*/45, /*N=*/33, /*a_transposed=*/true);
}

TEST(CudaBatchedMatmulTest, Rank4CollapsesAndRestoresLogicalShape) {
  const int64_t B0 = 2, B1 = 3, M = 24, K = 40, N = 17;
  const int64_t B = B0 * B1;
  auto ha = gen(B * M * K), hb = gen(B * K * N);
  Tensor c = torch::matmul(make_cuda({B0, B1, M, K}, ha), make_cuda({B0, B1, K, N}, hb));
  ASSERT_EQ(c.shape(), Shape({B0, B1, M, N}));

  auto got = to_host(c);
  auto want = reference_batched_matmul(ha, hb, B, M, K, N);
  ASSERT_EQ(got.size(), want.size());
  for (size_t p = 0; p < want.size(); ++p)
    EXPECT_NEAR(got[p], want[p], 1e-3f) << "at flat index " << p;
}
