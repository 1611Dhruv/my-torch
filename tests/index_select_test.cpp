// Spec-by-tests for torch::index_select along dim 0 (the embedding lookup).
//
// Contract: out.shape == idx.shape ++ a.shape[1:], and flat row k of `out`
// is a copy of row idx.flatten()[k] of `a`. Oracle for strided inputs is the
// same op run on a.contiguous(), as in elementwise_test.cpp.
//
// Run:  ctest --test-dir build -R IndexSelect

#include "mytorch/ops.h"
#include "mytorch/tensor.h"
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <vector>

using torch::CPU;
using torch::DType;
using torch::Tensor;
using Shape = std::vector<int64_t>;

// --- helpers ---------------------------------------------------------------

static void fill(Tensor &t, const std::vector<float> &vals) {
  ASSERT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  float *data = t.data_ptr<float>();
  for (int64_t i = 0; i < t.numel(); i++)
    data[i] = vals[i];
}

// a of shape `shape`, filled with 0,1,2,... so row r of a 2-D tensor is
// [r*cols, r*cols+1, ...] and bugs show up as recognizable numbers.
static Tensor arange(Shape shape) {
  Tensor t(shape);
  std::vector<float> vals(t.numel());
  for (int64_t i = 0; i < t.numel(); i++)
    vals[i] = static_cast<float>(i);
  fill(t, vals);
  return t;
}

static Tensor indices(Shape shape, const std::vector<int32_t> &vals) {
  Tensor t = Tensor::zeros(shape, DType::Int32, CPU);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  int32_t *data = t.data_ptr<int32_t>();
  for (int64_t i = 0; i < t.numel(); i++)
    data[i] = vals[i];
  return t;
}

static void expect_close(Tensor &got, const std::vector<float> &want) {
  ASSERT_EQ(static_cast<int64_t>(want.size()), got.numel());
  ASSERT_TRUE(got.is_contiguous());
  const float *data = got.data_ptr<float>();
  for (int64_t i = 0; i < got.numel(); i++)
    EXPECT_NEAR(data[i], want[i], 1e-7) << "at flat index " << i;
}

static void expect_tensors_close(Tensor &got, Tensor &want) {
  ASSERT_EQ(got.shape(), want.shape());
  ASSERT_TRUE(got.is_contiguous());
  ASSERT_TRUE(want.is_contiguous());
  const float *g = got.data_ptr<float>();
  const float *w = want.data_ptr<float>();
  for (int64_t i = 0; i < got.numel(); i++)
    EXPECT_NEAR(g[i], w[i], 1e-7) << "at flat index " << i;
}

// --- shape contract --------------------------------------------------------

TEST(IndexSelectTest, OutputShapeIsIdxShapePlusTrailingDims) {
  // (vocab=5, d=3) looked up with (B=2, T=2) ids -> (2, 2, 3)
  Tensor a = arange({5, 3});
  Tensor idx = indices({2, 2}, {4, 0, 0, 2});
  Tensor out = torch::index_select(a, idx);
  EXPECT_EQ(out.shape(), Shape({2, 2, 3}));
}

TEST(IndexSelectTest, OneDimIdxGivesTwoDimOutput) {
  Tensor a = arange({5, 3});
  Tensor idx = indices({4}, {4, 0, 0, 2});
  Tensor out = torch::index_select(a, idx);
  EXPECT_EQ(out.shape(), Shape({4, 3}));
}

TEST(IndexSelectTest, ScalarIdxGivesSingleRow) {
  // 0-dim index: out.shape == a.shape[1:]. Probes reshape on a 0-dim tensor.
  Tensor a = arange({5, 3});
  Tensor idx = indices({}, {3});
  Tensor out = torch::index_select(a, idx);
  EXPECT_EQ(out.shape(), Shape({3}));
  expect_close(out, {9, 10, 11});
}

TEST(IndexSelectTest, HigherRankRowsAreCopiedWhole) {
  // a is (4, 2, 3): each "row" is a 2x3 block of 6 elements.
  Tensor a = arange({4, 2, 3});
  Tensor idx = indices({2}, {3, 1});
  Tensor out = torch::index_select(a, idx);
  EXPECT_EQ(out.shape(), Shape({2, 2, 3}));
  // row 3 = 18..23, row 1 = 6..11
  expect_close(out, {18, 19, 20, 21, 22, 23, 6, 7, 8, 9, 10, 11});
}

// --- values ----------------------------------------------------------------

TEST(IndexSelectTest, ValuesMatchHandComputed) {
  // rows of a: r0=[0,1,2] r1=[3,4,5] r2=[6,7,8] r3=[9,10,11] r4=[12,13,14]
  // idx flat = [4, 0, 0, 2]. The repeated 0 matters for backward later.
  Tensor a = arange({5, 3});
  Tensor idx = indices({2, 2}, {4, 0, 0, 2});
  Tensor out = torch::index_select(a, idx);
  expect_close(out, {12, 13, 14, 0, 1, 2, 0, 1, 2, 6, 7, 8});
}

TEST(IndexSelectTest, DuplicateIndicesProduceIndependentCopies) {
  // Writing into one selected row must not change the other copy of it.
  Tensor a = arange({5, 3});
  Tensor idx = indices({2}, {1, 1});
  Tensor out = torch::index_select(a, idx);
  out.data_ptr<float>()[0] = 999.f;
  EXPECT_FLOAT_EQ(out.data_ptr<float>()[3], 3.f);
}

TEST(IndexSelectTest, OutputDoesNotAliasInput) {
  Tensor a = arange({5, 3});
  Tensor idx = indices({1}, {2});
  Tensor out = torch::index_select(a, idx);
  out.data_ptr<float>()[0] = -1.f;
  EXPECT_FLOAT_EQ(a.data_ptr<float>()[6], 6.f);
}

// --- strides / offsets: the bugs that matter -------------------------------

TEST(IndexSelectTest, TransposedInputMatchesContiguous) {
  // a = base.transpose(0,1) is (5,3) but non-contiguous: row r of `a` is
  // column r of `base`. Oracle = same op on a.contiguous().
  Tensor base = arange({3, 5});
  Tensor a = base.transpose(0, 1);
  ASSERT_FALSE(a.is_contiguous());

  Tensor idx = indices({3}, {4, 1, 4});
  Tensor got = torch::index_select(a, idx);
  Tensor a_c = a.contiguous();
  Tensor want = torch::index_select(a_c, idx);
  expect_tensors_close(got, want);

  // and the first row really is column 4 of base: [4, 9, 14]
  expect_close(got, {4, 9, 14, 1, 6, 11, 4, 9, 14});
}

TEST(IndexSelectTest, SlicedInputWithOffsetMatchesContiguous) {
  // a = base.slice(0, 2, 5): rows 2..4 of base, non-zero storage offset.
  Tensor base = arange({5, 3});
  Tensor a = base.slice(0, 2, 5);
  ASSERT_EQ(a.shape(), Shape({3, 3}));

  Tensor idx = indices({2}, {0, 2});
  Tensor got = torch::index_select(a, idx);
  // row 0 of a = row 2 of base = [6,7,8]; row 2 of a = row 4 = [12,13,14]
  expect_close(got, {6, 7, 8, 12, 13, 14});
}

TEST(IndexSelectTest, NonContiguousIdxIsFlattenedInLogicalOrder) {
  // idx as a transposed view: logical order must be respected, not storage order.
  Tensor a = arange({5, 3});
  Tensor idx_base = indices({2, 2}, {4, 0, 1, 2}); // rows [4,0],[1,2]
  Tensor idx = idx_base.transpose(0, 1);           // logical [4,1],[0,2]
  Tensor out = torch::index_select(a, idx);
  EXPECT_EQ(out.shape(), Shape({2, 2, 3}));
  expect_close(out, {12, 13, 14, 3, 4, 5, 0, 1, 2, 6, 7, 8});
}

// --- misuse ----------------------------------------------------------------

TEST(IndexSelectTest, OutOfRangeIndexThrows) {
  Tensor a = arange({5, 3});
  EXPECT_THROW(torch::index_select(a, indices({1}, {5})), std::invalid_argument);
  EXPECT_THROW(torch::index_select(a, indices({1}, {-1})), std::invalid_argument);
}

TEST(IndexSelectTest, NonInt32IndexThrows) {
  Tensor a = arange({5, 3});
  Tensor idx = arange({2}); // Float32
  EXPECT_THROW(torch::index_select(a, idx), std::invalid_argument);
}

TEST(IndexSelectTest, ZeroDimInputThrows) {
  Tensor a({});
  Tensor idx = indices({1}, {0});
  EXPECT_THROW(torch::index_select(a, idx), std::invalid_argument);
}

// --- CUDA: same contract, CPU result is the oracle --------------------------
//
// Inputs are built on the CPU with the helpers above, moved with .to(), run on
// the device, and copied back. Requires a CUDA device at runtime.
//
// Run:  ctest --test-dir build -R IndexSelectCuda

using torch::CUDA;

static Tensor to_cuda(const Tensor &t) { return t.to(t.dtype(), CUDA); }
static Tensor to_cpu(const Tensor &t) { return t.to(t.dtype(), CPU); }

// Run index_select on CPU and on CUDA with the same inputs and compare.
static void expect_cuda_matches_cpu(const Tensor &a, const Tensor &idx) {
  Tensor want = torch::index_select(a, idx);
  Tensor got_dev = torch::index_select(to_cuda(a), idx);
  ASSERT_EQ(got_dev.device(), CUDA);
  Tensor got = to_cpu(got_dev);
  expect_tensors_close(got, want);
}

TEST(IndexSelectCudaTest, SmallLookupMatchesCpu) {
  Tensor a = arange({5, 3});
  Tensor idx = indices({2, 2}, {4, 0, 0, 2});
  expect_cuda_matches_cpu(a, idx);
}

TEST(IndexSelectCudaTest, OutputShapeOnDevice) {
  Tensor a = to_cuda(arange({5, 3}));
  Tensor idx = indices({2, 2}, {4, 0, 0, 2});
  Tensor out = torch::index_select(a, idx);
  EXPECT_EQ(out.shape(), Shape({2, 2, 3}));
  EXPECT_EQ(out.device(), CUDA);
}

TEST(IndexSelectCudaTest, IdxAlreadyOnDevice) {
  // dispatch only moves idx when devices differ; the already-on-CUDA path
  // must behave the same.
  Tensor a = arange({5, 3});
  Tensor idx = indices({3}, {1, 4, 1});
  Tensor want = torch::index_select(a, idx);
  Tensor got = to_cpu(torch::index_select(to_cuda(a), to_cuda(idx)));
  expect_tensors_close(got, want);
}

TEST(IndexSelectCudaTest, RowLongerThanOneBlock) {
  // d = 1000 is not a multiple of 256, so a per-row kernel with a 256-thread
  // block has a ragged tail. Exercises the bounds guard within a row.
  Tensor a = arange({7, 1000});
  Tensor idx = indices({3}, {6, 0, 3});
  expect_cuda_matches_cpu(a, idx);
}

TEST(IndexSelectCudaTest, ManyRowsTransformerShaped) {
  // (vocab=50, d=64) looked up with (B=4, T=37) ids, 37 chosen so the number
  // of output rows is not a nice multiple of anything.
  constexpr int64_t vocab = 50, d = 64, B = 4, T = 37;
  Tensor a = arange({vocab, d});
  std::vector<int32_t> ids(B * T);
  for (int64_t i = 0; i < B * T; i++)
    ids[i] = static_cast<int32_t>((i * 7 + 3) % vocab);
  Tensor idx = indices({B, T}, ids);
  expect_cuda_matches_cpu(a, idx);
}

TEST(IndexSelectCudaTest, HigherRankRows) {
  Tensor a = arange({4, 2, 3});
  Tensor idx = indices({2}, {3, 1});
  expect_cuda_matches_cpu(a, idx);
}

TEST(IndexSelectCudaTest, TransposedInputMatchesCpu) {
  // Non-contiguous a on the device. Whether the kernel walks strides or the
  // dispatcher calls .contiguous() first, the answer must match.
  Tensor base = arange({3, 5});
  Tensor a_cpu = base.transpose(0, 1);
  Tensor idx = indices({3}, {4, 1, 4});

  Tensor want = torch::index_select(a_cpu, idx);
  Tensor a_dev = to_cuda(base).transpose(0, 1);
  ASSERT_FALSE(a_dev.is_contiguous());
  Tensor got = to_cpu(torch::index_select(a_dev, idx));
  expect_tensors_close(got, want);
}

TEST(IndexSelectCudaTest, OutputDoesNotAliasInput) {
  Tensor a = to_cuda(arange({5, 3}));
  Tensor idx = indices({1}, {2});
  Tensor out = torch::index_select(a, idx);
  Tensor out_cpu = to_cpu(out);
  out_cpu.data_ptr<float>()[0] = -1.f;
  Tensor a_back = to_cpu(a);
  EXPECT_FLOAT_EQ(a_back.data_ptr<float>()[6], 6.f);
}

// --- backward: index_select_back is the transpose (scatter-add) ------------
//
// Contract: out = zeros(M, a.shape[1:]); for each flat i,
//   out[idx[i], :] += g[i, :]
// Duplicate indices accumulate (sum, never average). Untouched rows stay 0.
//
// Beyond the hand-computed toy, the general check is the adjoint identity for
// any linear map P and its transpose: <P a, g> == <a, P^T g>. Here P is
// index_select and P^T is index_select_back, so for arbitrary a and g,
//   sum(index_select(a, idx) * g) == sum(a * index_select_back(g, idx, M)).
// That holds for every idx with no reference implementation needed.
//
// Run:  ctest --test-dir build -R IndexSelectBack

static float dot(Tensor &x, Tensor &y) {
  EXPECT_EQ(x.shape(), y.shape());
  EXPECT_TRUE(x.is_contiguous());
  EXPECT_TRUE(y.is_contiguous());
  const float *px = x.data_ptr<float>();
  const float *py = y.data_ptr<float>();
  double acc = 0;
  for (int64_t i = 0; i < x.numel(); i++)
    acc += static_cast<double>(px[i]) * py[i];
  return static_cast<float>(acc);
}

static void expect_tensors_close_tol(Tensor &got, Tensor &want, float tol) {
  ASSERT_EQ(got.shape(), want.shape());
  ASSERT_TRUE(got.is_contiguous());
  ASSERT_TRUE(want.is_contiguous());
  const float *g = got.data_ptr<float>();
  const float *w = want.data_ptr<float>();
  for (int64_t i = 0; i < got.numel(); i++)
    EXPECT_NEAR(g[i], w[i], tol * (1.f + std::abs(w[i]))) << "at flat index " << i;
}

TEST(IndexSelectBackTest, ToyHandComputed) {
  // a = (5,3), idx = [[4,0],[0,2]], g = ones(2,2,3).
  // Row 0 was selected twice -> 2. Rows 2 and 4 once -> 1. Rows 1, 3 -> 0.
  Tensor idx = indices({2, 2}, {4, 0, 0, 2});
  Tensor g = Tensor::ones({2, 2, 3});
  Tensor grad_a = torch::index_select_back(g, idx, 5);
  EXPECT_EQ(grad_a.shape(), Shape({5, 3}));
  expect_close(grad_a, {2, 2, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1});
}

TEST(IndexSelectBackTest, DistinctGradRowsLandInTheRightPlace) {
  // g rows are distinguishable so a swapped arrow (g[idx[i]] into out[i])
  // produces visibly wrong numbers instead of coincidentally right ones.
  Tensor idx = indices({3}, {2, 0, 2});
  Tensor g = arange({3, 2}); // rows [0,1], [2,3], [4,5]
  Tensor grad_a = torch::index_select_back(g, idx, 4);
  // row 0 <- g[1] = [2,3]; row 2 <- g[0] + g[2] = [4,6]; rows 1,3 zero
  expect_close(grad_a, {2, 3, 0, 0, 4, 6, 0, 0});
}

TEST(IndexSelectBackTest, AllIndicesTheSameAccumulateNotAverage) {
  Tensor idx = indices({4}, {1, 1, 1, 1});
  Tensor g = Tensor::ones({4, 3});
  Tensor grad_a = torch::index_select_back(g, idx, 3);
  expect_close(grad_a, {0, 0, 0, 4, 4, 4, 0, 0, 0});
}

TEST(IndexSelectBackTest, ScalarIdx) {
  // idx is 0-dim, so g.shape == a.shape[1:] with no leading dims at all.
  Tensor idx = indices({}, {3});
  Tensor g = arange({3}); // [0,1,2]
  Tensor grad_a = torch::index_select_back(g, idx, 5);
  EXPECT_EQ(grad_a.shape(), Shape({5, 3}));
  expect_close(grad_a, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0});
}

TEST(IndexSelectBackTest, HigherRankRows) {
  // a is (4, 2, 3); each row is a 2x3 block. idx = [3, 1, 3].
  Tensor idx = indices({3}, {3, 1, 3});
  Tensor g = arange({3, 2, 3}); // blocks 0..5, 6..11, 12..17
  Tensor grad_a = torch::index_select_back(g, idx, 4);
  EXPECT_EQ(grad_a.shape(), Shape({4, 2, 3}));
  std::vector<float> want(24, 0.f);
  for (int j = 0; j < 6; j++) {
    want[1 * 6 + j] = static_cast<float>(6 + j);             // g[1]
    want[3 * 6 + j] = static_cast<float>(j) + (12.f + j);    // g[0] + g[2]
  }
  expect_close(grad_a, want);
}

TEST(IndexSelectBackTest, AdjointIdentityHoldsForArbitraryInputs) {
  // <index_select(a, idx), g> == <a, index_select_back(g, idx, M)>
  Tensor a = arange({7, 5});
  Tensor idx = indices({2, 3}, {6, 0, 3, 0, 0, 5});
  Tensor g = arange({2, 3, 5});
  // make g less structured than arange so the two sides can't match by luck
  float *pg = g.data_ptr<float>();
  for (int64_t i = 0; i < g.numel(); i++)
    pg[i] = std::sin(0.37f * pg[i]) * 10.f;

  Tensor Pa = torch::index_select(a, idx);
  Tensor PTg = torch::index_select_back(g, idx, 7);
  EXPECT_NEAR(dot(Pa, g), dot(a, PTg), 1e-3);
}

TEST(IndexSelectBackTest, NonContiguousGradMatchesContiguous) {
  // g arrives as a transposed view. Oracle is the same op on g.contiguous().
  Tensor base = arange({3, 2, 2});
  Tensor g = base.transpose(0, 2); // shape (2, 2, 3), non-contiguous
  ASSERT_FALSE(g.is_contiguous());
  Tensor idx = indices({2, 2}, {4, 0, 0, 2});

  Tensor got = torch::index_select_back(g, idx, 5);
  Tensor g_c = g.contiguous();
  Tensor want = torch::index_select_back(g_c, idx, 5);
  expect_tensors_close(got, want);
}

TEST(IndexSelectBackTest, NonContiguousIdxIsFlattenedInLogicalOrder) {
  Tensor idx_base = indices({2, 2}, {4, 0, 1, 2});
  Tensor idx = idx_base.transpose(0, 1); // logical [[4,1],[0,2]]
  Tensor g = arange({2, 2, 1});          // rows 0,1,2,3 in logical order
  Tensor grad_a = torch::index_select_back(g, idx, 5);
  // logical flat idx = [4,1,0,2] paired with g = [0,1,2,3]
  expect_close(grad_a, {2, 1, 3, 0, 0});
}

TEST(IndexSelectBackTest, OutOfRangeIndexThrows) {
  Tensor g = Tensor::ones({1, 3});
  EXPECT_THROW(torch::index_select_back(g, indices({1}, {5}), 5), std::invalid_argument);
  EXPECT_THROW(torch::index_select_back(g, indices({1}, {-1}), 5), std::invalid_argument);
}

TEST(IndexSelectBackTest, NonInt32IndexThrows) {
  Tensor g = Tensor::ones({2, 3});
  Tensor idx = arange({2}); // Float32
  EXPECT_THROW(torch::index_select_back(g, idx, 5), std::invalid_argument);
}

TEST(IndexSelectBackTest, IdxRankExceedingGradRankThrows) {
  Tensor g = Tensor::ones({4});
  Tensor idx = indices({2, 2}, {0, 1, 2, 3});
  EXPECT_THROW(torch::index_select_back(g, idx, 5), std::invalid_argument);
}

// --- backward on CUDA: CPU result is the oracle -----------------------------
//
// atomicAdd on floats is order-nondeterministic, so comparisons here use a
// relative tolerance rather than exact equality.

static void expect_cuda_back_matches_cpu(const Tensor &g, const Tensor &idx,
                                         int64_t M, float tol = 1e-5f) {
  Tensor want = torch::index_select_back(g, idx, M);
  Tensor got_dev = torch::index_select_back(to_cuda(g), idx, M);
  ASSERT_EQ(got_dev.device(), CUDA);
  Tensor got = to_cpu(got_dev);
  expect_tensors_close_tol(got, want, tol);
}

TEST(IndexSelectBackCudaTest, ToyMatchesCpu) {
  Tensor idx = indices({2, 2}, {4, 0, 0, 2});
  Tensor g = arange({2, 2, 3});
  expect_cuda_back_matches_cpu(g, idx, 5);
}

TEST(IndexSelectBackCudaTest, IdxAlreadyOnDevice) {
  Tensor idx = indices({3}, {1, 4, 1});
  Tensor g = arange({3, 3});
  Tensor want = torch::index_select_back(g, idx, 5);
  Tensor got = to_cpu(torch::index_select_back(to_cuda(g), to_cuda(idx), 5));
  expect_tensors_close_tol(got, want, 1e-5f);
}

TEST(IndexSelectBackCudaTest, HeavyDuplicationLikePaddingTokens) {
  // 4*37 = 148 rows, ~80% of them index 0, like a padded batch. Stresses
  // atomic contention on one row and the accumulate-not-overwrite contract.
  constexpr int64_t vocab = 50, d = 64, B = 4, T = 37;
  std::vector<int32_t> ids(B * T);
  for (int64_t i = 0; i < B * T; i++)
    ids[i] = (i % 5 == 0) ? static_cast<int32_t>((i * 7 + 3) % vocab) : 0;
  Tensor idx = indices({B, T}, ids);
  Tensor g = arange({B, T, d});
  expect_cuda_back_matches_cpu(g, idx, vocab, 1e-4f);
}

TEST(IndexSelectBackCudaTest, RowLongerThanOneWarpStride) {
  // d = 1000 so each warp loops over its row many times with a ragged tail.
  Tensor idx = indices({3}, {6, 0, 6});
  Tensor g = arange({3, 1000});
  expect_cuda_back_matches_cpu(g, idx, 7, 1e-4f);
}

TEST(IndexSelectBackCudaTest, HigherRankRows) {
  Tensor idx = indices({3}, {3, 1, 3});
  Tensor g = arange({3, 2, 3});
  expect_cuda_back_matches_cpu(g, idx, 4);
}

TEST(IndexSelectBackCudaTest, NonContiguousGradOnDevice) {
  // The device kernel indexes g densely, so a strided g must be made
  // contiguous somewhere upstream. This fails if nobody does that.
  Tensor base = arange({3, 2, 2});
  Tensor idx = indices({2, 2}, {4, 0, 0, 2});
  Tensor want = torch::index_select_back(base.transpose(0, 2), idx, 5);
  Tensor g_dev = to_cuda(base).transpose(0, 2);
  ASSERT_FALSE(g_dev.is_contiguous());
  Tensor got = to_cpu(torch::index_select_back(g_dev, idx, 5));
  expect_tensors_close_tol(got, want, 1e-5f);
}

TEST(IndexSelectBackCudaTest, AdjointIdentityOnDevice) {
  Tensor a = arange({7, 5});
  Tensor idx = indices({2, 3}, {6, 0, 3, 0, 0, 5});
  Tensor g = arange({2, 3, 5});
  float *pg = g.data_ptr<float>();
  for (int64_t i = 0; i < g.numel(); i++)
    pg[i] = std::sin(0.37f * pg[i]) * 10.f;

  Tensor Pa = to_cpu(torch::index_select(to_cuda(a), idx));
  Tensor PTg = to_cpu(torch::index_select_back(to_cuda(g), idx, 7));
  EXPECT_NEAR(dot(Pa, g), dot(a, PTg), 1e-3);
}
