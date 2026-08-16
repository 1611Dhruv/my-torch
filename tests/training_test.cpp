// End-to-end training tests: Module + autograd + SGD + MSE as one loop.
//
// Everything below this layer is already covered elsewhere -- ops by
// elementwise/matmul/reduction tests, backwards by the numerical gradchecks.
// What is NOT covered by any of those is the *loop*: whether the optimizer
// actually reaches the parameters, whether the update has the right sign,
// whether zero_grad fires, and whether a fresh graph is built each step.
//
// Every one of those failures looks identical from the outside ("the loss
// doesn't move") and none of them raise an error, which is why they get
// dedicated tests rather than being inferred from a converging model.
//
// Ordered cheapest-diagnosis-first: wiring, then one step, then convergence.
//
// Run:  ctest --test-dir build -R Training

#include "mytorch/autograd.h"
#include "mytorch/loss.h"
#include "mytorch/nn/module.h"
#include "mytorch/optim.h"
#include "mytorch/tensor.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using torch::CPU;
using torch::DType;
using torch::MSE;
using torch::SGD;
using torch::Tensor;
using torch::autograd::VarPtr;
using torch::autograd::Variable;
namespace nn = torch::nn;

// --- helpers ---------------------------------------------------------------

// A {rows, cols} Float32 leaf from row-major values. requires_grad=false: these
// are data, not parameters, so nothing should be learning them.
static VarPtr input(int64_t rows, int64_t cols, const std::vector<float> &vals) {
  Tensor t({rows, cols}, DType::Float32, CPU);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  float *p = t.data_ptr<float>();
  for (size_t i = 0; i < vals.size(); ++i)
    p[i] = vals[i];
  return Variable::leaf(t, false);
}

// Look a parameter up by the tail of its registered name, so these tests read
// as "the weight" rather than depending on registration order.
static VarPtr param_named(const nn::Module &m, const std::string &suffix) {
  for (auto &[name, p] : m.named_params())
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      return p;
  ADD_FAILURE() << "no parameter whose name ends in '" << suffix << "'";
  return nullptr;
}

static float scalar_of(const VarPtr &v) { return v->data().data_ptr<float>()[0]; }

// One training step. Returns the loss *before* the update.
static float step(nn::Module &model, SGD &opt, const VarPtr &x, const VarPtr &y) {
  opt.zero_grad();
  MSE loss(model(x), y);
  float before = loss.loss();
  loss.backward();
  opt.step();
  return before;
}

// ===========================================================================
// Wiring -- these fail fast and point at a specific broken link
// ===========================================================================

TEST(TrainingWiringTest, SequentialExposesEveryNestedParameter) {
  // Sequential declares its own _modules, which shadows Module::_modules. If
  // its constructor ever stops calling register_module, params() silently
  // returns a shorter list, the optimizer gets a subset, and the model
  // half-trains with no error anywhere.
  auto model = nn::Sequential({std::make_shared<nn::Linear>(3, 4),
                               std::make_shared<nn::ReLU>(),
                               std::make_shared<nn::Linear>(4, 2)});
  EXPECT_EQ(model.params().size(), 4u); // 2 Linears x (weight + bias)

  // And the names should be prefixed by their module, not collide.
  auto named = model.named_params();
  ASSERT_EQ(named.size(), 4u);
  for (size_t i = 0; i < named.size(); ++i)
    for (size_t j = i + 1; j < named.size(); ++j)
      EXPECT_NE(named[i].first, named[j].first) << "duplicate parameter name";
}

TEST(TrainingWiringTest, SgdRejectsAnEmptyParameterList) {
  // The failure mode this guards: constructing an optimizer over a model whose
  // parameters were never registered. Silently training nothing is much worse
  // than refusing to start.
  EXPECT_THROW(SGD({}, 0.1f), std::invalid_argument);
}

TEST(TrainingWiringTest, BackwardPopulatesEveryParameterGradient) {
  nn::Linear lin(2, 3);
  auto x = input(4, 2, {1, 2, 3, 4, 5, 6, 7, 8});
  auto y = input(4, 3, std::vector<float>(12, 0.5f));

  MSE loss(lin(x), y);
  loss.backward();

  for (auto &[name, p] : lin.named_params())
    EXPECT_TRUE(p->has_grad()) << name << " received no gradient";
}

TEST(TrainingWiringTest, ZeroGradClearsGradientsBetweenSteps) {
  // Without this, gradients accumulate across iterations and the effective
  // learning rate grows every step -- which looks like divergence, not a bug.
  nn::Linear lin(2, 2);
  auto x = input(2, 2, {1, 0, 0, 1});
  auto y = input(2, 2, {0, 1, 1, 0});
  SGD opt(lin.params(), 0.01f);

  MSE(lin(x), y).backward();
  for (auto &p : lin.params())
    ASSERT_TRUE(p->has_grad());

  opt.zero_grad();
  for (auto &p : lin.params())
    EXPECT_FALSE(p->has_grad()) << "zero_grad left a stale gradient";
}

TEST(TrainingWiringTest, StepActuallyMutatesTheParameters) {
  // Catches the optimizer holding copies rather than references to the live
  // parameters -- in which case it updates orphans and the model never moves.
  nn::Linear lin(2, 2);
  auto x = input(2, 2, {1, 0, 0, 1});
  auto y = input(2, 2, {5, 5, 5, 5});
  SGD opt(lin.params(), 0.1f);

  auto w = param_named(lin, "weight");
  ASSERT_NE(w, nullptr);
  std::vector<float> before(w->data().data_ptr<float>(),
                            w->data().data_ptr<float>() + w->data().numel());

  step(lin, opt, x, y);

  const float *after = w->data().data_ptr<float>();
  bool changed = false;
  for (size_t i = 0; i < before.size(); ++i)
    changed |= (before[i] != after[i]);
  EXPECT_TRUE(changed) << "weights are identical after a step";
}

TEST(TrainingWiringTest, OneStepReducesTheLoss) {
  // The narrowest possible convergence check: if the update has the wrong
  // sign, this fails immediately instead of after a thousand iterations.
  nn::Linear lin(1, 1);
  auto x = input(4, 1, {-2, -1, 1, 2});
  auto y = input(4, 1, {-8, -4, 4, 8});
  SGD opt(lin.params(), 0.01f);

  float first = step(lin, opt, x, y);
  float second = MSE(lin(x), y).loss();
  EXPECT_LT(second, first) << "loss rose after one step (sign error in update?)";
}

// ===========================================================================
// Convergence
// ===========================================================================

TEST(TrainingTest, LinearRecoversAnAffineFunction) {
  // y = 4x + 20, so the learned weight and bias have known values -- this
  // checks the loop found the *right* answer, not merely a smaller loss.
  torch::manual_seed(0);
  const std::vector<float> xs = {-2, -1, 0, 1, 2};
  std::vector<float> ys;
  for (float v : xs)
    ys.push_back(4.0f * v + 20.0f);

  auto x = input(5, 1, xs);
  auto y = input(5, 1, ys);

  nn::Linear lin(1, 1);
  SGD opt(lin.params(), 0.05f);

  float first = 0.0f, last = 0.0f;
  for (int i = 0; i < 2000; ++i) {
    float l = step(lin, opt, x, y);
    if (i == 0)
      first = l;
    last = l;
  }

  EXPECT_LT(last, first);
  EXPECT_LT(last, 1e-3f) << "did not converge; final loss " << last;
  EXPECT_NEAR(scalar_of(param_named(lin, "weight")), 4.0f, 0.05f);
  EXPECT_NEAR(scalar_of(param_named(lin, "bias")), 20.0f, 0.05f);
}

TEST(TrainingTest, LossDecreasesMonotonicallyEarlyOn) {
  torch::manual_seed(0);
  auto x = input(5, 1, {-2, -1, 0, 1, 2});
  auto y = input(5, 1, {-8, -4, 0, 4, 8});
  nn::Linear lin(1, 1);
  SGD opt(lin.params(), 0.02f);

  float prev = std::numeric_limits<float>::infinity();
  for (int i = 0; i < 20; ++i) {
    float l = step(lin, opt, x, y);
    EXPECT_LT(l, prev) << "loss increased at iteration " << i;
    prev = l;
  }
}

TEST(TrainingTest, MlpLearnsXor) {
  // The load-bearing test. XOR is not linearly separable, so a model without a
  // working nonlinearity provably cannot fit it -- passing here means the
  // activation and the chain rule through two layers are both correct, which
  // no single-layer test can establish.
  torch::manual_seed(0);
  auto x = input(4, 2, {0, 0, 0, 1, 1, 0, 1, 1});
  auto y = input(4, 1, {0, 1, 1, 0});

  nn::Sequential model({std::make_shared<nn::Linear>(2, 16),
                        std::make_shared<nn::ReLU>(),
                        std::make_shared<nn::Linear>(16, 1)});
  ASSERT_EQ(model.params().size(), 4u);
  SGD opt(model.params(), 0.05f);

  float last = 0.0f;
  for (int i = 0; i < 4000; ++i)
    last = step(model, opt, x, y);

  EXPECT_LT(last, 0.01f) << "XOR did not converge; final loss " << last;

  // And check the predictions themselves, not just the loss scalar.
  // Hold the VarPtr: model(x) returns a shared_ptr by value, so binding only
  // its data_ptr would leave a dangling pointer once the temporary dies at the
  // end of the statement.
  auto out = model(x);
  const float *pred = out->data().data_ptr<float>();
  const float want[4] = {0, 1, 1, 0};
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(pred[i], want[i], 0.15f) << "wrong prediction at row " << i;
}

// ===========================================================================
// Normalization layers
//
// Both norms had six bugs at once when first written, and every one of them
// was silent: unregistered parameters (the optimizer never sees gamma/beta, so
// the model trains slightly worse forever), a missing keep_dim (broadcasts
// only line up when d_model happens to equal seq_len), and a missing epsilon
// (NaN on any constant row). These pin all three shapes of failure.
// ===========================================================================

// Shapes chosen so H(=8) != T(=3) != B(=2): if a reduction collapses the wrong
// axis, the broadcast cannot accidentally succeed.
static constexpr int64_t NB = 2, NT = 3, NH = 8;

TEST(NormTest, ParametersAreRegistered) {
  // register_param is what puts gamma/beta in params(). Skip it and the module
  // still runs -- it just never learns, with no error anywhere.
  EXPECT_EQ(nn::RMSNorm(NH).params().size(), 1u) << "RMSNorm should expose gain";
  EXPECT_EQ(nn::LayerNorm(NH).params().size(), 2u)
      << "LayerNorm should expose gain and bias";
}

TEST(NormTest, GainStartsAtOneAndBiasAtZero) {
  // gamma initialized to randn (or zeros) rescales every feature at init and
  // looks exactly like a broken backward.
  nn::LayerNorm ln(NH);
  for (auto &[name, p] : ln.named_params()) {
    const float *v = p->data().data_ptr<float>();
    float want = (name.find("bias") != std::string::npos) ? 0.0f : 1.0f;
    for (int64_t i = 0; i < p->data().numel(); ++i)
      EXPECT_FLOAT_EQ(v[i], want) << name << "[" << i << "]";
  }
}

TEST(NormTest, PreservesShapeWhenFeatureDimDiffersFromSeqLen) {
  // The keep_dim guard. With H != T a collapsed reduction throws or produces
  // the wrong shape instead of silently working.
  auto x = input(NB * NT, NH, std::vector<float>(NB * NT * NH, 0.7f));
  nn::RMSNorm rms(NH);
  nn::LayerNorm ln(NH);
  EXPECT_EQ(rms(x)->data().shape(), std::vector<int64_t>({NB * NT, NH}));
  EXPECT_EQ(ln(x)->data().shape(), std::vector<int64_t>({NB * NT, NH}));
}

TEST(NormTest, ConstantRowStaysFiniteBecauseOfEpsilon) {
  // A constant row has zero variance, so LayerNorm divides by sqrt(0) without
  // eps. Padded and zero-initialized sequences look exactly like this, and the
  // resulting NaN passes silently through every tolerance comparison you have.
  std::vector<float> flat(NB * NT * NH, 3.5f);
  auto x = input(NB * NT, NH, flat);

  nn::LayerNorm ln(NH);
  nn::RMSNorm rms(NH);
  auto a = ln(x);
  auto b = rms(x);
  const float *pa = a->data().data_ptr<float>();
  const float *pb = b->data().data_ptr<float>();
  for (int64_t i = 0; i < a->data().numel(); ++i) {
    EXPECT_TRUE(std::isfinite(pa[i])) << "LayerNorm produced non-finite at " << i;
    EXPECT_TRUE(std::isfinite(pb[i])) << "RMSNorm produced non-finite at " << i;
  }
}

TEST(NormTest, LayerNormOutputHasZeroMeanAndUnitVarianceAtDefaultInit) {
  // With gain=1 and bias=0 the definition is checkable directly: each row of
  // the output must have mean 0 and variance 1. This is the invariant test --
  // gradcheck can only tell you the backward matches the forward, not that the
  // forward is normalization.
  std::vector<float> vals;
  for (int64_t i = 0; i < NB * NT * NH; ++i)
    vals.push_back(static_cast<float>((i * 37 % 11)) - 5.0f);
  auto x = input(NB * NT, NH, vals);

  nn::LayerNorm ln(NH);
  auto y = ln(x);
  const float *p = y->data().data_ptr<float>();

  for (int64_t r = 0; r < NB * NT; ++r) {
    double mean = 0.0;
    for (int64_t c = 0; c < NH; ++c)
      mean += p[r * NH + c];
    mean /= NH;
    double var = 0.0;
    for (int64_t c = 0; c < NH; ++c) {
      double d = p[r * NH + c] - mean;
      var += d * d;
    }
    var /= NH;
    EXPECT_NEAR(mean, 0.0, 1e-5) << "row " << r << " mean";
    EXPECT_NEAR(var, 1.0, 1e-3) << "row " << r << " variance";
  }
}

TEST(NormTest, RmsNormScalesRowsToUnitRootMeanSquare) {
  std::vector<float> vals;
  for (int64_t i = 0; i < NB * NT * NH; ++i)
    vals.push_back(static_cast<float>((i * 29 % 13)) - 6.0f);
  auto x = input(NB * NT, NH, vals);

  nn::RMSNorm rms(NH);
  auto y = rms(x);
  const float *p = y->data().data_ptr<float>();

  for (int64_t r = 0; r < NB * NT; ++r) {
    double ms = 0.0;
    for (int64_t c = 0; c < NH; ++c)
      ms += static_cast<double>(p[r * NH + c]) * p[r * NH + c];
    EXPECT_NEAR(std::sqrt(ms / NH), 1.0, 1e-3) << "row " << r << " rms";
  }
}

TEST(NormTest, NormParametersReceiveGradientsAndTrain) {
  // End to end: the norm's own parameters must move under the optimizer. This
  // is what an unregistered parameter actually costs you.
  torch::manual_seed(0);
  auto x = input(4, NH, std::vector<float>(4 * NH, 0.3f));
  auto y = input(4, NH, std::vector<float>(4 * NH, 2.0f));

  nn::LayerNorm ln(NH);
  SGD opt(ln.params(), 0.05f);

  auto gain = param_named(ln, "gain");
  ASSERT_NE(gain, nullptr);
  std::vector<float> before(gain->data().data_ptr<float>(),
                            gain->data().data_ptr<float>() + NH);

  float first = 0.0f, last = 0.0f;
  for (int i = 0; i < 200; ++i) {
    float l = step(ln, opt, x, y);
    if (i == 0)
      first = l;
    last = l;
  }
  EXPECT_LT(last, first) << "norm parameters are not learning";

  const float *after = gain->data().data_ptr<float>();
  bool moved = false;
  for (int64_t i = 0; i < NH; ++i)
    moved |= (before[i] != after[i]);
  EXPECT_TRUE(moved) << "gain never changed -- is it registered?";
}
