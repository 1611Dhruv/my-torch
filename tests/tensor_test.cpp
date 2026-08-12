// Spec-by-tests for torch::Tensor.
//
// These are the spec for Tensor's metadata/view/access behavior. They link
// against the TEMPORARY leaky Storage stub, so any failures here are about
// *Tensor*, not Storage. Red until you implement the Tensor bodies.
//
// HEADS UP on templates: at<T>(), data_ptr<T>(), item<T>() are member
// TEMPLATES. Template definitions must be visible where they're instantiated,
// so either define them in tensor.h, or add explicit instantiations in
// tensor.cpp (e.g. `template float *Tensor::data_ptr<float>();`). Otherwise
// these tests fail to LINK with "undefined reference to ...<float>".
//
// Run:  ctest --test-dir build -R Tensor

#include "mytorch/tensor.h"
#include <gtest/gtest.h>
#include <vector>

using torch::CPU;
using torch::DType;
using torch::Tensor;

using Shape = std::vector<int64_t>;

// --- construction & metadata ------------------------------------------------

TEST(TensorTest, ContiguousConstructionMetadata) {
  Tensor t({2, 3}); // defaults: Float32, CPU
  EXPECT_EQ(t.ndim(), 2);
  EXPECT_EQ(t.numel(), 6);
  EXPECT_EQ(t.shape(), (Shape{2, 3}));
  EXPECT_EQ(t.strides(), (Shape{3, 1})); // row-major: right-to-left products
  EXPECT_EQ(t.offset(), 0);
  EXPECT_EQ(t.dtype(), DType::Float32);
  EXPECT_EQ(t.device(), CPU);
  EXPECT_TRUE(t.is_contiguous());
}

TEST(TensorTest, ThreeDimStrides) {
  Tensor t({2, 3, 4});
  EXPECT_EQ(t.numel(), 24);
  EXPECT_EQ(t.strides(), (Shape{12, 4, 1}));
}

TEST(TensorTest, DTypeIsRemembered) {
  Tensor t({4}, DType::Int32);
  EXPECT_EQ(t.dtype(), DType::Int32);
}

// --- data access ------------------------------------------------------------

TEST(TensorTest, DataPtrRoundTrip) {
  Tensor t({4}, DType::Float32);
  float *p = t.data_ptr<float>();
  ASSERT_NE(p, nullptr);
  for (int i = 0; i < 4; ++i)
    p[i] = static_cast<float>(i) + 0.5f;
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(t.data_ptr<float>()[i], static_cast<float>(i) + 0.5f);
}

TEST(TensorTest, AtIndexingMatchesRowMajorLayout) {
  Tensor t({2, 3}, DType::Float32);
  // fill via flat buffer: value = row*10 + col
  float *p = t.data_ptr<float>();
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      p[i * 3 + j] = static_cast<float>(i * 10 + j);
  // read back via at<T>({i,j})
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ((t.at<float>({i, j})), static_cast<float>(i * 10 + j));
}

TEST(TensorTest, ItemReadsCorrectBytes) {
  Tensor t({2}, DType::Float32);
  float *p = t.data_ptr<float>();
  p[1] = 3;
  EXPECT_EQ(t[1].item<float>(), t.at<float>({1}));
}

// --- transpose: metadata-only view, shares storage --------------------------

TEST(TensorTest, TransposeSwapsShapeAndStrides) {
  Tensor t({2, 3});
  Tensor tt = t.transpose(0, 1);
  EXPECT_EQ(tt.shape(), (Shape{3, 2}));
  EXPECT_EQ(tt.strides(), (Shape{1, 3})); // strides swapped
}

TEST(TensorTest, TransposeSharesStorageNoCopy) {
  Tensor t({2, 3});
  Tensor tt = t.transpose(0, 1);
  EXPECT_EQ(tt.data_ptr<float>(), t.data_ptr<float>()); // SAME buffer
}

TEST(TensorTest, TransposeIsNonContiguous) {
  Tensor t({2, 3});
  Tensor tt = t.transpose(0, 1);
  EXPECT_FALSE(tt.is_contiguous());
}

TEST(TensorTest, TransposeLogicalValuesAreFlipped) {
  Tensor t({2, 3}, DType::Float32);
  float *p = t.data_ptr<float>();
  for (int i = 0; i < 6; ++i)
    p[i] = static_cast<float>(i);
  Tensor tt = t.transpose(0, 1);
  // tt[j,i] must equal t[i,j]
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ((tt.at<float>({j, i})), (t.at<float>({i, j})));
}

TEST(TensorTest, TransposeNegativeIndicesWork) {
  Tensor t({2, 3}, DType::Float32);
  float *p = t.data_ptr<float>();
  for (int i = 0; i < 6; ++i)
    p[i] = static_cast<float>(i);
  Tensor tt = t.transpose(0, -1);
  // tt[j,i] must equal t[i,j]
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ((tt.at<float>({j, i})), (t.at<float>({i, j})));
}

TEST(TensorTest, TransposePreservesOffset) {
  Tensor t({2, 2, 3}, DType::Float32);
  t = t[1];
  float *p = t.data_ptr<float>();
  for (int i = 0; i < 6; ++i)
    p[i] = static_cast<float>(i);
  Tensor tt = t.transpose(0, 1);
  EXPECT_EQ(t.offset(), tt.offset());
}

// --- reshape ----------------------------------------------------------------

TEST(TensorTest, ReshapePreservesNumelAndIsContiguous) {
  Tensor t({2, 3});
  Tensor r = t.reshape({6});
  EXPECT_EQ(r.shape(), (Shape{6}));
  EXPECT_EQ(r.numel(), 6);
  EXPECT_TRUE(r.is_contiguous());
}

TEST(TensorTest, ReshapeOfContiguousSharesStorage) {
  Tensor t({2, 3});
  Tensor r = t.reshape({3, 2});
  EXPECT_EQ(r.data_ptr<float>(), t.data_ptr<float>());
}

// --- operator[] returns a sub-tensor view -----------------------------------

TEST(TensorTest, IndexReturnsSubview) {
  Tensor t({2, 3});
  Tensor row = t[1];

  float *row_data = row.data_ptr<float>();
  row_data[1] = 3.;

  EXPECT_EQ(row.ndim(), 1);
  EXPECT_EQ(row.shape(), (Shape{3}));
  EXPECT_EQ(t.at<float>({1, 1}), 3.); // shares storage
}

// --- contiguous() -----------------------------------------------------------

TEST(TensorTest, ContiguousMaterializesTranspose) {
  Tensor t({2, 3}, DType::Float32);
  float *p = t.data_ptr<float>();
  for (int i = 0; i < 6; ++i)
    p[i] = static_cast<float>(i);
  Tensor c = t.transpose(0, 1).contiguous();
  EXPECT_TRUE(c.is_contiguous());
  // logical values preserved: c[j,i] == t[i,j]
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ((c.at<float>({j, i})), (t.at<float>({i, j})));
}

// --- factories --------------------------------------------------------------

TEST(TensorTest, ZerosAreAllZero) {
  Tensor t = Tensor::zeros({2, 2}, DType::Float32, CPU);
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(t.data_ptr<float>()[i], 0.0f);
}

TEST(TensorTest, OnesAreAllOne) {
  Tensor t = Tensor::ones({5}, DType::Float32, CPU);
  for (int i = 0; i < 5; ++i)
    EXPECT_FLOAT_EQ(t.data_ptr<float>()[i], 1.0f);
}

// --- rng --------------------------------------------------------------------

TEST(TensorTest, ManualSeedIsReproducible) {
  torch::manual_seed(42);
  Tensor a = Tensor::rand({5}, CPU);
  torch::manual_seed(42);
  Tensor b = Tensor::rand({5}, CPU);
  for (int i = 0; i < 5; ++i)
    EXPECT_FLOAT_EQ(a.data_ptr<float>()[i], b.data_ptr<float>()[i]);
}

TEST(TensorTest, DifferentSeedsDiffer) {
  torch::manual_seed(1);
  Tensor a = Tensor::rand({5}, CPU);
  torch::manual_seed(2);
  Tensor b = Tensor::rand({5}, CPU);
  bool all_equal = true;
  for (int i = 0; i < 5; ++i)
    if (a.data_ptr<float>()[i] != b.data_ptr<float>()[i])
      all_equal = false;
  EXPECT_FALSE(all_equal);
}

// ===========================================================================
// CUDA contiguous() / strided unary ops
//
// Tensor::contiguous() on a CUDA tensor dispatches to a device-side gather
// (cuda::contiguous -> the strided unary kernel). These tests require a device.
//
// The load-bearing case is the multi-head-attention layout: reshaping
// {B, T, H*d} to {B, T, H, d} and transposing dims 1 and 2 produces strides
// {T*H*d, d, H*d, 1} -- neither contiguous nor a plain 2-D transpose, and the
// batch dims don't collapse to a single stride. The matmul kernel can't consume
// it directly, so every attention layer depends on this copy being right.
// ===========================================================================

#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include <cuda_runtime_api.h>

using torch::CUDA;

// Upload a row-major host buffer as a contiguous CUDA tensor.
static Tensor cuda_from(const Shape &shape, const std::vector<float> &vals) {
  Tensor t(shape, DType::Float32, CUDA);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  CUDA_CHECK(cudaMemcpy(t.data_ptr<float>(), vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice));
  return t;
}

static std::vector<float> host_of(const Tensor &t) {
  std::vector<float> h(t.numel());
  CUDA_CHECK(cudaMemcpy(h.data(), t.data_ptr<float>(), h.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

// Distinct values so any index mix-up shows up (not periodic like p % 9).
static std::vector<float> seq(int64_t n) {
  std::vector<float> v(n);
  for (int64_t p = 0; p < n; ++p)
    v[p] = static_cast<float>(p);
  return v;
}

TEST(CudaContiguousTest, AlreadyContiguousDoesNotCopy) {
  Tensor t = cuda_from({4, 6}, seq(24));
  Tensor c = t.contiguous();
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_EQ(host_of(c), seq(24));
  // Same storage, so the copy was skipped entirely.
  EXPECT_EQ(c.data_ptr<float>(), t.data_ptr<float>());
}

TEST(CudaContiguousTest, TransposeMaterializesCorrectly) {
  const int64_t R = 4, C = 6;
  Tensor t = cuda_from({R, C}, seq(R * C));
  Tensor tt = t.transpose(0, 1);
  ASSERT_EQ(tt.shape(), Shape({C, R}));
  ASSERT_FALSE(tt.is_contiguous());

  Tensor c = tt.contiguous();
  ASSERT_TRUE(c.is_contiguous());
  ASSERT_EQ(c.shape(), Shape({C, R}));
  ASSERT_NE(c.data_ptr<float>(), t.data_ptr<float>()); // a real copy happened

  std::vector<float> want(R * C); // row-major transpose of seq(R*C)
  for (int64_t i = 0; i < R; ++i)
    for (int64_t j = 0; j < C; ++j)
      want[j * R + i] = static_cast<float>(i * C + j);
  EXPECT_EQ(host_of(c), want);
}

TEST(CudaContiguousTest, MhaLayoutMaterializesCorrectly) {
  // {B,T,H,d} -> transpose(1,2) -> {B,H,T,d}. Strides become
  // {T*H*d, d, H*d, 1}: row stride H*d != d, so it is neither contiguous nor a
  // transpose, and the {B,H} batch region does not collapse. THE case.
  const int64_t B = 2, T = 3, H = 4, d = 5;
  Tensor x = cuda_from({B, T, H, d}, seq(B * T * H * d));
  Tensor v = x.transpose(1, 2);
  ASSERT_EQ(v.shape(), Shape({B, H, T, d}));
  ASSERT_EQ(v.strides(), Shape({T * H * d, d, H * d, 1}));
  ASSERT_FALSE(v.is_contiguous());

  Tensor c = v.contiguous();
  ASSERT_TRUE(c.is_contiguous());
  ASSERT_EQ(c.shape(), Shape({B, H, T, d}));

  std::vector<float> want(B * H * T * d);
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t t = 0; t < T; ++t)
        for (int64_t e = 0; e < d; ++e)
          want[((b * H + h) * T + t) * d + e] = static_cast<float>(((b * T + t) * H + h) * d + e);
  EXPECT_EQ(host_of(c), want);
}

TEST(CudaContiguousTest, ReshapeOfNonContiguousWorksOnDevice) {
  // reshape() falls back to contiguous() for non-contiguous input, so this is
  // the path every QKV split and head merge takes.
  const int64_t B = 2, T = 3, H = 4, d = 5;
  Tensor x = cuda_from({B, T, H, d}, seq(B * T * H * d));
  Tensor merged = x.transpose(1, 2).reshape({B, H, T * d});
  ASSERT_EQ(merged.shape(), Shape({B, H, T * d}));
  ASSERT_TRUE(merged.is_contiguous());
  EXPECT_EQ(host_of(merged), host_of(x.transpose(1, 2).contiguous()));
}

TEST(CudaContiguousTest, StridedUnaryOpMatchesReference) {
  // The same strided kernel now backs neg/sin/cos/exp, which used to throw on
  // non-contiguous input. Check one against the materialize-then-op route.
  const int64_t R = 4, C = 6;
  Tensor t = cuda_from({R, C}, seq(R * C));
  Tensor strided = torch::cuda::neg(t.transpose(0, 1));
  Tensor materialized = torch::cuda::neg(t.transpose(0, 1).contiguous());
  ASSERT_EQ(strided.shape(), materialized.shape());
  EXPECT_EQ(host_of(strided), host_of(materialized));
}

TEST(CudaContiguousTest, OffsetViewMaterializesCorrectly) {
  // operator[] moves _offset, which data_ptr() folds in -- the gather must not
  // add it a second time.
  Tensor x = cuda_from({3, 4, 5}, seq(60));
  Tensor slab = x[1].transpose(0, 1); // {5,4}, offset 20, strides {1,5}
  ASSERT_EQ(slab.shape(), Shape({5, 4}));
  ASSERT_FALSE(slab.is_contiguous());

  std::vector<float> want(20);
  for (int64_t i = 0; i < 4; ++i)
    for (int64_t j = 0; j < 5; ++j)
      want[j * 4 + i] = static_cast<float>(20 + i * 5 + j);
  EXPECT_EQ(host_of(slab.contiguous()), want);
}
