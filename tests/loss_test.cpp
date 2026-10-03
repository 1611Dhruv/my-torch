// Spec-by-tests for torch::CrossEntropy on logits.
//
// Contract: pred is {.., V} logits, act is a one-hot of the same shape.
//   per-row loss = logsumexp(z) - z_y
//   loss()       = mean of per-row losses over every leading position
//   d loss / d z = (softmax(z) - onehot) / N   with N the number of rows
// Hand-computed values below; no PyTorch on this box, so the oracle for the
// batched case is an independent double-precision implementation in the test.
//
// Run:  ctest --test-dir build -R CrossEntropy

#include "mytorch/autograd.h"
#include "mytorch/loss.h"
#include "mytorch/tensor.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using torch::CPU;
using torch::CrossEntropy;
using torch::DType;
using torch::Tensor;
using torch::autograd::Variable;
using torch::autograd::VarPtr;
using Shape = std::vector<int64_t>;

// --- helpers ----------------------------------------------------------------

static VarPtr leaf(Shape shape, const std::vector<float> &vals, bool grad) {
  Tensor t(shape, DType::Float32, CPU);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  float *p = t.data_ptr<float>();
  for (size_t i = 0; i < vals.size(); ++i)
    p[i] = vals[i];
  return Variable::leaf(t, grad);
}

// One-hot leaf of shape `shape` (last dim V) from class ids, row-major.
static VarPtr onehot(Shape shape, const std::vector<int32_t> &ids) {
  int64_t V = shape.back();
  int64_t rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i)
    rows *= shape[i];
  EXPECT_EQ(static_cast<int64_t>(ids.size()), rows);
  std::vector<float> vals(rows * V, 0.f);
  for (int64_t r = 0; r < rows; ++r)
    vals[r * V + ids[r]] = 1.f;
  return leaf(shape, vals, false);
}

static std::vector<float> grad_of(const VarPtr &v) {
  EXPECT_TRUE(v->has_grad());
  Tensor g = v->grad()->contiguous();
  const float *p = g.data_ptr<float>();
  return std::vector<float>(p, p + g.numel());
}

// Independent reference: mean over rows of logsumexp(z) - z_y, in double.
static double reference_ce(const std::vector<float> &z, int64_t V,
                           const std::vector<int32_t> &ids) {
  int64_t rows = ids.size();
  double total = 0;
  for (int64_t r = 0; r < rows; ++r) {
    double m = -1e300;
    for (int64_t k = 0; k < V; ++k)
      m = std::max(m, static_cast<double>(z[r * V + k]));
    double s = 0;
    for (int64_t k = 0; k < V; ++k)
      s += std::exp(z[r * V + k] - m);
    total += m + std::log(s) - z[r * V + ids[r]];
  }
  return total / rows;
}

// --- values -----------------------------------------------------------------

TEST(CrossEntropyTest, SingleRowHandComputed) {
  // z = [1,2,3], y = 2. logsumexp = 3 + ln(e^-2 + e^-1 + 1) = 3.40760596.
  // loss = 0.40760596.
  auto z = leaf({1, 3}, {1, 2, 3}, false);
  CrossEntropy ce(z, onehot({1, 3}, {2}));
  EXPECT_NEAR(ce.loss(), 0.40760596f, 1e-5f);
}

TEST(CrossEntropyTest, MeanOverTwoRowsHandComputed) {
  // Row 0: [1,2,3], y=2 -> 0.40760596.  Row 1: [0,0,0], y=0 -> ln 3.
  // Mean = (0.40760596 + 1.09861229) / 2 = 0.75310913.
  // A loss that reads only element 0, or sums instead of averaging, fails.
  auto z = leaf({2, 3}, {1, 2, 3, 0, 0, 0}, false);
  CrossEntropy ce(z, onehot({2, 3}, {2, 0}));
  EXPECT_NEAR(ce.loss(), 0.75310913f, 1e-5f);
}

TEST(CrossEntropyTest, UniformLogitsGiveLnV) {
  auto z = leaf({1, 8}, std::vector<float>(8, 0.f), false);
  CrossEntropy ce(z, onehot({1, 8}, {5}));
  EXPECT_NEAR(ce.loss(), std::log(8.f), 1e-6f);
}

TEST(CrossEntropyTest, ConfidentCorrectPredictionIsNearZero) {
  auto z = leaf({1, 4}, {-20, 30, -20, -20}, false);
  CrossEntropy ce(z, onehot({1, 4}, {1}));
  EXPECT_NEAR(ce.loss(), 0.f, 1e-6f);
}

TEST(CrossEntropyTest, ConfidentWrongPredictionIsLarge) {
  // Target 0 but all the mass on class 1: loss ~= 50 (the logit gap).
  auto z = leaf({1, 4}, {-20, 30, -20, -20}, false);
  CrossEntropy ce(z, onehot({1, 4}, {0}));
  EXPECT_NEAR(ce.loss(), 50.f, 1e-3f);
}

TEST(CrossEntropyTest, LargeLogitsDoNotOverflow) {
  // e^1000 overflows float32; the max-subtraction trick must make this ln 3.
  // Tolerance is loose on purpose: 1000 + ln3 - 1000 in float32 loses ~6e-5
  // to cancellation (one ulp at 1000). PyTorch has the same rounding here.
  auto z = leaf({1, 3}, {1000, 1000, 1000}, false);
  CrossEntropy ce(z, onehot({1, 3}, {0}));
  EXPECT_TRUE(std::isfinite(ce.loss()));
  EXPECT_NEAR(ce.loss(), std::log(3.f), 1e-4f);
}

TEST(CrossEntropyTest, ThreeDimLogitsMatchReference) {
  // {B=2, T=3, V=4}: the Transformer's shape. Reduction is over B*T rows.
  Shape shape = {2, 3, 4};
  std::vector<float> z(24);
  for (int i = 0; i < 24; ++i)
    z[i] = std::sin(0.7f * i) * 3.f;
  std::vector<int32_t> ids = {0, 3, 1, 2, 2, 0};
  auto zl = leaf(shape, z, false);
  CrossEntropy ce(zl, onehot(shape, ids));
  EXPECT_NEAR(ce.loss(), reference_ce(z, 4, ids), 1e-5);
}

// --- gradient ---------------------------------------------------------------

TEST(CrossEntropyTest, GradientIsSoftmaxMinusOnehotOverN) {
  // Row 0: softmax([1,2,3]) = [0.09003057, 0.24472847, 0.66524096], y=2.
  // Row 1: softmax([0,0,0]) = [1/3, 1/3, 1/3],                     y=0.
  // d(mean)/dz = (softmax - onehot) / 2.
  auto z = leaf({2, 3}, {1, 2, 3, 0, 0, 0}, true);
  CrossEntropy ce(z, onehot({2, 3}, {2, 0}));
  ce.backward();
  auto g = grad_of(z);
  ASSERT_EQ(g.size(), 6u);
  std::vector<float> want = {0.09003057f / 2, 0.24472847f / 2, -0.33475904f / 2,
                             (1.f / 3 - 1) / 2, (1.f / 3) / 2, (1.f / 3) / 2};
  for (int i = 0; i < 6; ++i)
    EXPECT_NEAR(g[i], want[i], 1e-6f) << "at " << i;
}

TEST(CrossEntropyTest, GradientRowsSumToZero) {
  // softmax sums to 1 and the one-hot sums to 1, so every row's gradient sums
  // to 0. Holds for any logits; a wrong broadcast of the max or the sum
  // usually breaks it.
  Shape shape = {2, 3, 5};
  std::vector<float> z(30);
  for (int i = 0; i < 30; ++i)
    z[i] = std::cos(1.3f * i) * 4.f;
  std::vector<int32_t> ids = {4, 0, 2, 2, 1, 3};
  auto zl = leaf(shape, z, true);
  CrossEntropy ce(zl, onehot(shape, ids));
  ce.backward();
  auto g = grad_of(zl);
  ASSERT_EQ(g.size(), 30u);
  for (int r = 0; r < 6; ++r) {
    float s = 0;
    for (int k = 0; k < 5; ++k)
      s += g[r * 5 + k];
    EXPECT_NEAR(s, 0.f, 1e-6f) << "row " << r;
  }
}

TEST(CrossEntropyTest, GradientMatchesFiniteDifferences) {
  // Central differences in the forward against the analytical backward.
  Shape shape = {2, 4};
  std::vector<float> z = {0.5f, -1.2f, 2.0f, 0.1f, -0.3f, 0.8f, -2.0f, 1.5f};
  std::vector<int32_t> ids = {2, 1};

  auto zl = leaf(shape, z, true);
  CrossEntropy ce(zl, onehot(shape, ids));
  ce.backward();
  auto g = grad_of(zl);

  const float h = 1e-2f;
  for (size_t i = 0; i < z.size(); ++i) {
    auto zp = z, zm = z;
    zp[i] += h;
    zm[i] -= h;
    CrossEntropy lp(leaf(shape, zp, false), onehot(shape, ids));
    CrossEntropy lm(leaf(shape, zm, false), onehot(shape, ids));
    float fd = (lp.loss() - lm.loss()) / (2 * h);
    EXPECT_NEAR(g[i], fd, 1e-3f) << "at " << i;
  }
}
