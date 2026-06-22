// Spec-by-tests for torch elementwise ops (add to start), both CPU and CUDA.
//
// CPU tests (ElementwiseTest) pin the behavior of the CPU kernels: correct
// values on the contiguous fast path, correct strided traversal on
// non-contiguous / offset views, no mutation of inputs, and loud errors on
// misuse. A good trick for the strided cases: don't hand-compute the expected
// values, just compare against the same op run on .contiguous() inputs (the
// fast path is your oracle).
//
// CUDA tests (ElementwiseCudaTest) actually launch the GPU kernel, so they
// require a CUDA device at runtime. Data is staged host->device with cudaMemcpy
// (there's no .to(device) yet), the kernel runs, and the result is copied back.
//
// Run:  ctest --test-dir build -R Elementwise           (both)
//       ctest --test-dir build -R ElementwiseCuda       (GPU only)

#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include "mytorch/tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using torch::CPU;
using torch::CUDA;
using torch::DType;
using torch::Tensor;
using Shape = std::vector<int64_t>;

// --- helpers ---------------------------------------------------------------

// Fill a contiguous float tensor from a vector, in flat row-major order.
static void fill(Tensor &t, const std::vector<float> &vals) {
  ASSERT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  int64_t N = vals.size();
  float *data = t.data_ptr<float>();

  for (int64_t i = 0; i < N; i++) {
    data[i] = vals[i];
  }
}

// Assert a float tensor's flat contents match `want` elementwise.
// Expected to called on contiguous tensors
static void expect_close(Tensor &got, const std::vector<float> &want) {
  ASSERT_EQ(static_cast<int64_t>(want.size()), got.numel());
  ASSERT_TRUE(got.is_contiguous());

  int64_t N = want.size();
  float *data = got.data_ptr<float>();
  const float threshold = 1e-7;
  for (int64_t i = 0; i < N; i++) {
    EXPECT_NEAR(data[i], want[i], threshold);
  }
}

// Oracle comparison: assert two contiguous float tensors are elementwise equal.
// For strided cases, build `want` by running the same op on .contiguous()
// inputs -- that routes through the trusted fast path, so you never hand-compute
// expected values (and never have to reason about permuted axis order).
static void expect_tensors_close(Tensor &got, Tensor &want) {
  ASSERT_EQ(got.shape(), want.shape());
  ASSERT_TRUE(got.is_contiguous());
  ASSERT_TRUE(want.is_contiguous());

  int64_t N = got.numel();
  const float *g = got.data_ptr<float>();
  const float *w = want.data_ptr<float>();
  for (int64_t i = 0; i < N; i++) {
    EXPECT_NEAR(g[i], w[i], 1e-7);
  }
}

// --- happy path (fast path: both contiguous) -------------------------------

TEST(ElementwiseTest, AddSameShapeContiguous) {
  // 2x3 + 2x3 with hand-computed expected. Exercises the contiguous fast path.
  Tensor a({2, 3});
  Tensor b({2, 3});

  std::vector<float> vals_a = {1, 2, 3, 4, 5, 6};
  std::vector<float> vals_b = {6, 5, 4, 3, 2, 1};
  fill(a, vals_a);
  fill(b, vals_b);

  Tensor out = torch::cpu::add(a, b);
  std::vector<float> expected = {7, 7, 7, 7, 7, 7};
  expect_close(out, expected);
}

TEST(ElementwiseTest, AddNonDivisibleShape) {
  // e.g. {3,5,7} so you're not accidentally relying on powers of two.
  Tensor a({3, 5, 7, 9});
  Tensor b({3, 5, 7, 9});

  int N = 3 * 5 * 7 * 9;
  std::vector<float> vals_a(N, 0);
  std::vector<float> vals_b(N, 0);

  for (int i = 0; i < N; i++) {
    vals_a[i] = i;
    vals_b[i] = N - i - 1;
  }
  fill(a, vals_a);
  fill(b, vals_b);

  Tensor out = torch::cpu::add(a, b);
  std::vector<float> expected(N, N - 1);
  expect_close(out, expected);
}

TEST(ElementwiseTest, AddZeroDimScalars) {
  // ndim==0 tensors: leaf hit immediately, exactly one add. Edge case.
  Tensor a({});
  Tensor b({});
  ASSERT_NO_THROW(torch::cpu::add(a, b));
}

// --- the stride/offset bugs: these are the ones that matter ----------------

TEST(ElementwiseTest, AddTransposedInputMatchesContiguous) {
  // b = something.transpose(0,1): non-contiguous but same shape as a.
  // Compare add(a, b) against add(a, b.contiguous()). Hits the slow path.
  Tensor a({3, 2});
  Tensor b({2, 3});

  std::vector<float> vals_a = {1, 2, 3, 4, 5, 6};
  std::vector<float> vals_b = {1, 2, 3, 4, 5, 6};
  fill(a, vals_a);
  fill(b, vals_b);

  /*
   * 1 2   1 4
   * 3 4 + 2 5
   * 5 6   3 6
   */

  Tensor out = torch::cpu::add(a, b.transpose(-1, -2));
  EXPECT_EQ(out.shape(), std::vector<int64_t>({3, 2}));
  std::vector<float> vals_want = {2, 6, 5, 9, 8, 12};
  expect_close(out, vals_want);
}

TEST(ElementwiseTest, AddBothInputsNonContiguous) {
  // both a and b are transposed views. Slow path for both operands.
  Tensor a({3, 5, 7, 9});
  Tensor b({3, 5, 7, 9});

  int N = 3 * 5 * 7 * 9;
  std::vector<float> vals_a(N, 0);
  std::vector<float> vals_b(N, 0);

  for (int i = 0; i < N; i++) {
    vals_a[i] = i;
    vals_b[i] = N - i - 1;
  }
  fill(a, vals_a);
  fill(b, vals_b);

  Tensor out = torch::cpu::add(a.transpose(-1, -2), b.transpose(-1, -2));
  EXPECT_EQ(out.shape(), std::vector<int64_t>({3, 5, 9, 7}));

  std::vector<float> expected(N, N - 1);
  expect_close(out, expected);
}

TEST(ElementwiseTest, AddInputWithNonzeroOffset) {
  // a[2] has offset != 0; the transpose also makes it non-contiguous, so this
  // exercises the strided slow path WITH a nonzero base offset -- the exact
  // case the offset double-count bug breaks. No hand math: the oracle is the
  // same add on .contiguous() inputs, which goes through the fast path.
  Tensor a({3, 5, 7, 9});
  Tensor b({3, 5, 7, 9});

  int N = 3 * 5 * 7 * 9;
  std::vector<float> vals_a(N, 0);
  std::vector<float> vals_b(N, 0);

  for (int i = 0; i < N; i++) {
    vals_a[i] = i;
    vals_b[i] = i;
  }
  fill(a, vals_a);
  fill(b, vals_b);

  Tensor lhs = a[2].transpose(-1, -2); // offset != 0 AND non-contiguous
  Tensor rhs = b[2].transpose(-1, -2);

  Tensor got = torch::cpu::add(lhs, rhs);                              // slow path
  Tensor oracle = torch::cpu::add(lhs.contiguous(), rhs.contiguous()); // fast path

  EXPECT_TRUE(got.is_contiguous());
  EXPECT_EQ(got.shape(), std::vector<int64_t>({5, 9, 7}));
  expect_tensors_close(got, oracle);
}

// --- guarantees about the output -------------------------------------------

TEST(ElementwiseTest, AddDoesNotMutateInputs) {
  // snapshot a and b's values, run add, assert both inputs are unchanged.
  Tensor a({2, 3});
  Tensor b({2, 3});
  std::vector<float> vals_a = {1, 2, 3, 4, 5, 6};
  std::vector<float> vals_b = {6, 5, 4, 3, 2, 1};
  fill(a, vals_a);
  fill(b, vals_b);

  Tensor out = torch::cpu::add(a, b);

  // add writes only into a fresh `out`; the operands stay exactly as filled.
  expect_close(a, vals_a);
  expect_close(b, vals_b);
}

TEST(ElementwiseTest, AddOutputIsFreshContiguous) {
  // out.is_contiguous() == true, and out.data_ptr != a.data_ptr (no aliasing).
  Tensor a = Tensor::zeros({2, 3}, DType::Float32, CPU);
  Tensor b = Tensor::zeros({2, 3}, DType::Float32, CPU);
  Tensor out = torch::cpu::add(a, b);
  EXPECT_TRUE(out.is_contiguous());
  EXPECT_NE(a.data_ptr<float>(), out.data_ptr<float>());
  EXPECT_NE(b.data_ptr<float>(), out.data_ptr<float>());
}

// --- error paths -----------------------------------------------------------

// NOTE: Eventual dispatch will handle all of this
TEST(ElementwiseTest, AddShapeMismatchThrows) {
  // same dtype, incompatible shapes -> throws (no broadcasting yet).
  Tensor a({2, 3});
  Tensor b({3, 2});
  EXPECT_THROW(torch::add(a, b), std::invalid_argument);
}

TEST(ElementwiseTest, AddDtypeMismatchThrows) {
  // matching shapes, different dtypes -> throws (no casting yet).
  Tensor a({2, 3}, DType::Float32);
  Tensor b({2, 3}, DType::Int32);
  EXPECT_THROW(torch::add(a, b), std::invalid_argument);
}

TEST(ElementwiseTest, AddDeviceMismatchThrows) {
  // matching shapes, matching dtypes, different device -> throws (no .to() yet).
  Tensor a({2, 3}, DType::Float32, CPU);
  Tensor b({2, 3}, DType::Float32, CUDA);
  EXPECT_THROW(torch::add(a, b), std::invalid_argument);
}

// --- CUDA strided test helpers ----------------------------------------------

// Skips every test in this fixture when there's no GPU, so the suite stays green
// on CPU-only machines instead of hard-failing inside cudaMemcpy.
class ElementwiseCudaStrided : public ::testing::Test {
protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
      GTEST_SKIP() << "no CUDA device available";
  }
};

// Allocate a CUDA tensor and stage host values into it (row-major contiguous).
static Tensor cuda_from(const Shape &shape, const std::vector<float> &vals) {
  Tensor t(shape, DType::Float32, CUDA);
  CUDA_CHECK(cudaMemcpy(t.data_ptr<float>(), vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice));
  return t;
}

// Copy a contiguous CUDA tensor's values back to host.
static std::vector<float> cuda_to_host(const Tensor &t) {
  std::vector<float> out(t.numel());
  CUDA_CHECK(cudaMemcpy(out.data(), t.data_ptr<float>(), t.numel() * sizeof(float), cudaMemcpyDeviceToHost));
  return out;
}

// Oracle: run the same op on identical data -- CPU vs CUDA -- through a
// transposed (non-contiguous) view, and assert they agree. The CPU strided path
// is already tested above, so it's the trusted reference; we never hand-compute
// expected values for the GPU strided kernel.
using BinaryOp = Tensor (*)(const Tensor &, const Tensor &);
static void expect_cuda_matches_cpu_transposed(BinaryOp cpu_op, BinaryOp gpu_op, const Shape &shape,
                                               const std::vector<float> &va, const std::vector<float> &vb) {
  Tensor ca(shape);
  Tensor cb(shape);
  fill(ca, va);
  fill(cb, vb);
  Tensor cpu_out = cpu_op(ca.transpose(-1, -2), cb.transpose(-1, -2));

  Tensor ga = cuda_from(shape, va);
  Tensor gb = cuda_from(shape, vb);
  Tensor gpu_out = gpu_op(ga.transpose(-1, -2), gb.transpose(-1, -2));

  ASSERT_EQ(cpu_out.shape(), gpu_out.shape());
  std::vector<float> got = cuda_to_host(gpu_out);
  const float *want = cpu_out.data_ptr<float>();
  for (int64_t i = 0; i < cpu_out.numel(); ++i)
    EXPECT_FLOAT_EQ(got[i], want[i]);
}

// --- CUDA: the GPU add kernel (requires a device at runtime) ----------------

// n is intentionally NOT a multiple of the 256-thread block size, so the last
// block is ragged and the kernel's `i >= n` bounds guard is exercised.
TEST(ElementwiseCudaTest, AddsElementwise) {
  constexpr int64_t n = 1000;

  std::vector<float> ha(n), hb(n);
  for (int64_t i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i + 1);
  }

  Tensor a({n}, DType::Float32, CUDA);
  Tensor b({n}, DType::Float32, CUDA);
  CUDA_CHECK(cudaMemcpy(a.data_ptr<float>(), ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data_ptr<float>(), hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  Tensor c = torch::cuda::add(a, b);

  std::vector<float> hc(n);
  CUDA_CHECK(cudaMemcpy(hc.data(), c.data_ptr<float>(), n * sizeof(float), cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < n; ++i)
    EXPECT_FLOAT_EQ(hc[i], ha[i] + hb[i]);
}

TEST(ElementwiseCudaTest, SubsElementwise) {
  constexpr int64_t n = 1000;

  std::vector<float> ha(n), hb(n);
  for (int64_t i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i + 1);
  }

  Tensor a({n}, DType::Float32, CUDA);
  Tensor b({n}, DType::Float32, CUDA);
  CUDA_CHECK(cudaMemcpy(a.data_ptr<float>(), ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data_ptr<float>(), hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  Tensor c = torch::cuda::sub(a, b);

  std::vector<float> hc(n);
  CUDA_CHECK(cudaMemcpy(hc.data(), c.data_ptr<float>(), n * sizeof(float), cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < n; ++i)
    EXPECT_FLOAT_EQ(hc[i], ha[i] - hb[i]);
}

TEST(ElementwiseCudaTest, MultipliesElementwise) {
  constexpr int64_t n = 1000;

  std::vector<float> ha(n), hb(n);
  for (int64_t i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i + 1);
  }

  Tensor a({n}, DType::Float32, CUDA);
  Tensor b({n}, DType::Float32, CUDA);
  CUDA_CHECK(cudaMemcpy(a.data_ptr<float>(), ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data_ptr<float>(), hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  Tensor c = torch::cuda::mult(a, b);

  std::vector<float> hc(n);
  CUDA_CHECK(cudaMemcpy(hc.data(), c.data_ptr<float>(), n * sizeof(float), cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < n; ++i)
    EXPECT_FLOAT_EQ(hc[i], ha[i] * hb[i]);
}

// --- CUDA: non-contiguous (strided) is now SUPPORTED, validated vs CPU oracle ---

TEST_F(ElementwiseCudaStrided, AddTransposedMatchesCpu) {
  // NOTE: asymmetric data on purpose -- a+b must NOT be constant, or a linear
  // (wrong) read and the correct strided read would both pass.
  expect_cuda_matches_cpu_transposed(torch::cpu::add, torch::cuda::add, {2, 3}, {1, 2, 3, 4, 5, 6},
                                     {10, 20, 30, 40, 50, 60});
}

TEST_F(ElementwiseCudaStrided, SubTransposedMatchesCpu) {
  expect_cuda_matches_cpu_transposed(torch::cpu::sub, torch::cuda::sub, {2, 3}, {1, 2, 3, 4, 5, 6}, {6, 5, 4, 3, 2, 1});
}

TEST_F(ElementwiseCudaStrided, MultTransposedMatchesCpu) {
  expect_cuda_matches_cpu_transposed(torch::cpu::mult, torch::cuda::mult, {2, 3}, {1, 2, 3, 4, 5, 6},
                                     {6, 5, 4, 3, 2, 1});
}

// Higher-rank, non-power-of-two shape -> exercises the ndim unravel loop and a
// ragged final block, still purely against the CPU oracle.
TEST_F(ElementwiseCudaStrided, AddTransposedHighRankMatchesCpu) {
  const int n = 3 * 5 * 7 * 9;
  std::vector<float> va(n), vb(n);
  for (int i = 0; i < n; ++i) {
    va[i] = static_cast<float>(i);
    vb[i] = static_cast<float>(2 * i + 1); // not n-i: that makes a+b constant and hides a linear-read bug
  }
  expect_cuda_matches_cpu_transposed(torch::cpu::add, torch::cuda::add, {3, 5, 7, 9}, va, vb);
}
