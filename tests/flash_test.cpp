// FlashAttention vs. naive attention.
//
// gradient_check in autograd_test.cpp validates flash_back against flash_atten
// -- it finite-differences the flash forward itself, so a self-consistent error
// in the forward passes clean. This file closes that gap: it builds attention a
// second, independent way out of matmul/softmax/matmul and compares both the
// forward values and the analytic dQ/dK/dV.
//
// Analytic-vs-analytic, so there is no finite-difference cancellation and the
// tolerances can be much tighter than the f32 gradcheck's 1e-2.
//
// Run:  ctest --test-dir build -R Flash

#include "mytorch/autograd.h"
#include "mytorch/tensor.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using torch::Tensor;
using torch::autograd::Variable;
using torch::autograd::VarPtr;
namespace ag = torch::autograd;

namespace {

// Flash uses __expf and accumulates the softmax denominator in a different
// order than the naive path, so exact equality is not on the table. These are
// still ~10x tighter than what an f32 gradcheck can assert.
constexpr double kFwdAtol = 1e-4, kFwdRtol = 1e-3;
constexpr double kGradAtol = 1e-4, kGradRtol = 5e-3;

std::vector<float> host(const Tensor &t) {
  Tensor c = t.to(torch::DType::Float32, torch::Device::CPU);
  const float *p = c.data_ptr<float>();
  return std::vector<float>(p, p + c.numel());
}

// Additive mask, same convention as MultiHeadSelfAttention: 0 where a query may
// look, a large negative where it may not.
Tensor causal_mask(int64_t T, torch::DType dt, torch::Device dev) {
  Tensor m({T, T});
  for (int64_t i = 0; i < T; i++)
    for (int64_t j = 0; j < T; j++)
      m.at<float>({i, j}) = (j > i) ? -1e11f : 0.0f;
  return m.to(dt, dev);
}

// The oracle: attention the obvious way, materializing the full T x T matrix.
VarPtr naive_atten(VarPtr Q, VarPtr K, VarPtr V, bool causal) {
  const auto &shp = Q->data().shape();
  const int64_t T = shp[shp.size() - 2];
  const int64_t d = shp[shp.size() - 1];

  auto s = ag::scale(ag::matmul(Q, ag::transpose(K, -1, -2)),
                     1.0 / std::sqrt(static_cast<double>(d)));
  if (causal) {
    auto m = Variable::leaf(
        causal_mask(T, Q->data().dtype(), Q->data().device()), false);
    s = ag::add(s, m);
  }
  return ag::matmul(ag::softmax(s, -1), V);
}

// Mixed abs/rel comparison that reports the worst element rather than the first,
// so a failure tells you whether one entry is off or the whole tensor is.
void expect_near(const std::vector<float> &ref, const std::vector<float> &got,
                 double atol, double rtol, const std::string &what) {
  ASSERT_EQ(ref.size(), got.size()) << what << ": size mismatch";

  double worst_excess = 0.0, worst_abs = 0.0, max_rel = 0.0;
  size_t worst = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    const double a = ref[i], b = got[i];
    const double diff = std::fabs(a - b);
    const double allow = atol + rtol * std::max(std::fabs(a), std::fabs(b));
    const double rel = diff / std::max(1e-9, std::max(std::fabs(a), std::fabs(b)));
    max_rel = std::max(max_rel, rel);
    if (diff - allow > worst_excess) {
      worst_excess = diff - allow;
      worst_abs = diff;
      worst = i;
    }
  }
  EXPECT_LE(worst_excess, 0.0)
      << what << ": worst at flat index " << worst << " naive=" << ref[worst]
      << " flash=" << got[worst] << " abs_diff=" << worst_abs
      << " (max_rel over tensor = " << max_rel << ")";
}

struct Case {
  int64_t B, T, D;
  bool causal;
  std::string name;
};

// Both paths read the same input tensors but own separate Variables, so their
// gradients accumulate independently.
void compare(const Case &c) {
  const auto dev = torch::Device::CUDA;
  auto q = Tensor::randn({c.B, c.T, c.D}, dev);
  auto k = Tensor::randn({c.B, c.T, c.D}, dev);
  auto v = Tensor::randn({c.B, c.T, c.D}, dev);

  auto qf = Variable::leaf(q, true), kf = Variable::leaf(k, true),
       vf = Variable::leaf(v, true);
  auto qn = Variable::leaf(q, true), kn = Variable::leaf(k, true),
       vn = Variable::leaf(v, true);

  auto out_f = ag::flash_atten(qf, kf, vf, c.causal);
  auto out_n = naive_atten(qn, kn, vn, c.causal);

  expect_near(host(out_n->data()), host(out_f->data()), kFwdAtol, kFwdRtol,
              c.name + " forward");

  // One shared upstream gradient, so both backward passes see the same seed.
  auto R = Variable::leaf(Tensor::randn({c.B, c.T, c.D}, dev), false);
  ag::sum(ag::mult(out_f, R), {})->backward();
  ag::sum(ag::mult(out_n, R), {})->backward();

  const std::pair<VarPtr, VarPtr> pairs[] = {{qn, qf}, {kn, kf}, {vn, vf}};
  const char *names[] = {"dQ", "dK", "dV"};
  for (int i = 0; i < 3; i++) {
    ASSERT_TRUE(pairs[i].first->grad().has_value())
        << c.name << ": naive " << names[i] << " missing";
    ASSERT_TRUE(pairs[i].second->grad().has_value())
        << c.name << ": flash " << names[i] << " missing";
    expect_near(host(*pairs[i].first->grad()), host(*pairs[i].second->grad()),
                kGradAtol, kGradRtol, c.name + " " + names[i]);
  }
}

} // namespace

// --- d_head = 32 (BS = 48 branch) -------------------------------------------

TEST(FlashVsNaive, D32) { compare({2, 64, 32, false, "d32"}); }
TEST(FlashVsNaive, D32Causal) { compare({2, 64, 32, true, "d32-causal"}); }

// --- d_head = 64 (BS = 32 branch, different dCNT and tile shape) ------------

TEST(FlashVsNaive, D64) { compare({2, 64, 64, false, "d64"}); }
TEST(FlashVsNaive, D64Causal) { compare({2, 64, 64, true, "d64-causal"}); }

// --- ragged N: the last tile is mostly padding ------------------------------

TEST(FlashVsNaive, RaggedN) { compare({1, 67, 32, false, "ragged"}); }
TEST(FlashVsNaive, RaggedNCausal) { compare({1, 67, 32, true, "ragged-causal"}); }

// --- structural: an exact assertion with no tolerance -----------------------
//
// Under a causal mask, query row 0 attends to exactly one key (itself), so
// P[0] = [1, 0, ...] and O_0 = V_0. Then D_0 = dO_0 . O_0 = dO_0 . V_0, which
// is exactly dP[0][0], so dS[0][0] = dP*P - P*D = 0. dQ's first row must be
// identically zero -- no tolerance argument required.
TEST(FlashVsNaive, CausalFirstQueryRowHasZeroGrad) {
  const auto dev = torch::Device::CUDA;
  const int64_t B = 1, T = 64, D = 32;

  auto qf = Variable::leaf(Tensor::randn({B, T, D}, dev), true);
  auto kf = Variable::leaf(Tensor::randn({B, T, D}, dev), true);
  auto vf = Variable::leaf(Tensor::randn({B, T, D}, dev), true);

  auto out = ag::flash_atten(qf, kf, vf, true);
  auto R = Variable::leaf(Tensor::randn({B, T, D}, dev), false);
  ag::sum(ag::mult(out, R), {})->backward();

  ASSERT_TRUE(qf->grad().has_value());
  auto dq = host(*qf->grad());
  for (int64_t j = 0; j < D; j++)
    EXPECT_NEAR(dq[j], 0.0f, 1e-6f) << "dQ[0][0][" << j << "] should be exactly 0";
}
