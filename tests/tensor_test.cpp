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
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j)
      p[i * 3 + j] = static_cast<float>(i * 10 + j);
  // read back via at<T>({i,j})
  for (int64_t i = 0; i < 2; ++i)
    for (int64_t j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ((t.at<float>({i, j})), static_cast<float>(i * 10 + j));
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
  EXPECT_EQ(row.ndim(), 1);
  EXPECT_EQ(row.shape(), (Shape{3}));
  EXPECT_EQ(row.data_ptr<float>(), t.data_ptr<float>()); // shares storage
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
