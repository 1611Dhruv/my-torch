// Spec-by-tests for torch::cpu elementwise ops (add to start).
//
// These pin the behavior of the CPU elementwise kernels: correct values on the
// contiguous fast path, correct strided traversal on non-contiguous / offset
// views, no mutation of inputs, and loud errors on misuse.
//
// Bodies are intentionally left as TODOs -- fill them in. A good trick for the
// strided cases: don't hand-compute the expected values, just compare against
// the same op run on .contiguous() inputs (the fast path is your oracle).
//
// Run:  ctest --test-dir build -R Elementwise

#include "mytorch/ops/ops.h"
#include "mytorch/tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using torch::CPU;
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

TEST(ElementwiseTest, AddShapeMismatchThrows) {
  // same dtype, incompatible shapes -> throws (no broadcasting yet).
  Tensor a({2, 3});
  Tensor b({3, 2});
  EXPECT_THROW(torch::cpu::add(a, b), std::invalid_argument);
}

TEST(ElementwiseTest, AddDtypeMismatchThrows) {
  // matching shapes, different dtypes -> throws (no casting yet).
  Tensor a({2, 3}, DType::Float32);
  Tensor b({2, 3}, DType::Int32);
  EXPECT_THROW(torch::cpu::add(a, b), std::invalid_argument);
}
