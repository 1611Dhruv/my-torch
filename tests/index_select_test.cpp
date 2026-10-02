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
