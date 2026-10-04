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
#include <cstdio>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using torch::CPU;
using torch::DType;
using torch::MSE;
using torch::SGD;
using torch::Tensor;
using torch::autograd::Variable;
using torch::autograd::VarPtr;
namespace nn = torch::nn;

// --- helpers ---------------------------------------------------------------

// A {rows, cols} Float32 leaf from row-major values. requires_grad=false: these
// are data, not parameters, so nothing should be learning them.
static VarPtr input(int64_t rows, int64_t cols,
                    const std::vector<float> &vals) {
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

static float scalar_of(const VarPtr &v) {
  return v->data().data_ptr<float>()[0];
}

// One training step. Returns the loss *before* the update.
static float step(nn::Module &model, SGD &opt, const VarPtr &x,
                  const VarPtr &y) {
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
  EXPECT_LT(second, first)
      << "loss rose after one step (sign error in update?)";
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
  EXPECT_EQ(nn::RMSNorm(NH).params().size(), 1u)
      << "RMSNorm should expose gain";
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
    EXPECT_TRUE(std::isfinite(pa[i]))
        << "LayerNorm produced non-finite at " << i;
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

// ===========================================================================
// MultiHeadAttention
//
// Attention is the easiest layer in the stack to get "plausible but wrong":
// every bug below still produces a correctly-shaped tensor of finite numbers,
// and a model containing it still trains -- just worse. So these tests check
// properties, not values.
//
// The load-bearing one is IsCausal. A decoder's output at position t must not
// depend on any input after t. That single property catches an inverted mask,
// a missing mask, a mask added after the softmax, and a mask broadcast along
// the wrong axis -- none of which change the output shape or produce a NaN.
// ===========================================================================

// A {B,T,D} Float32 leaf from row-major values.
static VarPtr seq_input(int64_t B, int64_t T, int64_t D,
                        const std::vector<float> &vals) {
  Tensor t({B, T, D}, DType::Float32, CPU);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  float *p = t.data_ptr<float>();
  for (size_t i = 0; i < vals.size(); ++i)
    p[i] = vals[i];
  return Variable::leaf(t, false);
}

static std::vector<float> varied(int64_t n, uint32_t salt = 0) {
  std::vector<float> v(n);
  for (int64_t i = 0; i < n; ++i) {
    uint32_t h = (static_cast<uint32_t>(i) + salt * 0x9E3779B9u) * 2654435761u;
    v[i] = static_cast<float>(static_cast<int32_t>(h % 17u) - 8) * 0.1f;
  }
  return v;
}

TEST(MhaTest, ConstructsAndExposesFourParameters) {
  torch::manual_seed(0);
  nn::MultiHeadAttention mha(8, 2, 4);
  EXPECT_EQ(mha.params().size(), 4u) << "expected Wq, Wk, Wv, Wo";
}

TEST(MhaTest, RejectsHeadCountThatDoesNotDivideModelDim) {
  EXPECT_THROW(nn::MultiHeadAttention(10, 4, 4), std::invalid_argument);
}

TEST(MhaTest, RejectsContextLongerThanConfigured) {
  torch::manual_seed(0);
  nn::MultiHeadAttention mha(8, 2, /*max_context=*/4);
  auto x = seq_input(1, 6, 8, varied(48));
  EXPECT_ANY_THROW(mha(x));
}

TEST(MhaTest, PreservesInputShape) {
  torch::manual_seed(0);
  // Every sublayer must return the residual-stream shape so it can be added
  // back. If the output projection is applied to the wrong operand this fails
  // (or throws) rather than silently producing the wrong rank.
  const int64_t B = 2, T = 4, D = 8;
  nn::MultiHeadAttention mha(D, 2, T);
  auto x = seq_input(B, T, D, varied(B * T * D));
  auto y = mha(x);
  EXPECT_EQ(y->data().shape(), std::vector<int64_t>({B, T, D}));
}

TEST(MhaTest, IsCausal) {
  // Perturb the LAST position of the sequence. Every output position strictly
  // before it must be bit-identical, because a causal mask forbids attending
  // forward. With the mask inverted (masking j <= i instead of j > i) position
  // 0 attends only to the future, so this fails immediately.
  const int64_t B = 1, T = 5, D = 8;
  torch::manual_seed(0);
  nn::MultiHeadAttention mha(D, 2, T);

  auto base = varied(B * T * D);
  auto y0 = mha(seq_input(B, T, D, base));
  std::vector<float> out0(y0->data().data_ptr<float>(),
                          y0->data().data_ptr<float>() + y0->data().numel());

  auto bumped = base;
  for (int64_t k = 0; k < D; ++k)
    bumped[(T - 1) * D + k] += 5.0f; // only the final token changes
  auto y1 = mha(seq_input(B, T, D, bumped));
  const float *out1 = y1->data().data_ptr<float>();

  for (int64_t t = 0; t + 1 < T; ++t)
    for (int64_t k = 0; k < D; ++k) {
      int64_t i = t * D + k;
      EXPECT_FLOAT_EQ(out0[i], out1[i])
          << "output at position " << t << " changed when position " << (T - 1)
          << " was perturbed -- attention is looking into the future";
    }
}

TEST(MhaTest, LaterPositionsDoDependOnEarlierOnes) {
  // The converse of IsCausal, so a mask that blocks *everything* can't pass by
  // making all outputs constant.
  const int64_t B = 1, T = 5, D = 8;
  torch::manual_seed(0);
  nn::MultiHeadAttention mha(D, 2, T);

  auto base = varied(B * T * D);
  auto y0 = mha(seq_input(B, T, D, base));
  std::vector<float> out0(y0->data().data_ptr<float>(),
                          y0->data().data_ptr<float>() + y0->data().numel());

  auto bumped = base;
  for (int64_t k = 0; k < D; ++k)
    bumped[k] += 5.0f; // perturb position 0
  auto y1 = mha(seq_input(B, T, D, bumped));
  const float *out1 = y1->data().data_ptr<float>();

  bool changed = false;
  for (int64_t k = 0; k < D; ++k)
    changed |= (out0[(T - 1) * D + k] != out1[(T - 1) * D + k]);
  EXPECT_TRUE(changed)
      << "the last position ignores the first -- is everything masked?";
}

TEST(MhaTest, ProducesFiniteOutputs) {
  torch::manual_seed(0);
  // -inf in the mask, or a fully-masked row, gives NaN -- which passes every
  // tolerance comparison in this file silently.
  const int64_t B = 2, T = 4, D = 8;
  nn::MultiHeadAttention mha(D, 2, T);
  auto y = mha(seq_input(B, T, D, varied(B * T * D)));
  const float *p = y->data().data_ptr<float>();
  for (int64_t i = 0; i < y->data().numel(); ++i)
    ASSERT_TRUE(std::isfinite(p[i])) << "non-finite output at " << i;
}

TEST(MhaTest, EveryParameterReceivesAGradient) {
  torch::manual_seed(0);
  const int64_t B = 2, T = 4, D = 8;
  nn::MultiHeadAttention mha(D, 2, T);
  auto x = seq_input(B, T, D, varied(B * T * D));
  auto y = seq_input(B, T, D, varied(B * T * D, 1));

  MSE(mha(x), y).backward();
  for (auto &[name, p] : mha.named_params())
    EXPECT_TRUE(p->has_grad()) << name << " received no gradient";
}

TEST(MhaTest, TrainsOnAToySequence) {
  const int64_t B = 2, T = 4, D = 8;
  torch::manual_seed(0);
  nn::MultiHeadAttention mha(D, 2, T);
  auto x = seq_input(B, T, D, varied(B * T * D));
  auto y = seq_input(B, T, D, varied(B * T * D, 1));
  SGD opt(mha.params(), 0.01f);

  float first = 0.0f, last = 0.0f;
  for (int i = 0; i < 300; ++i) {
    float l = step(mha, opt, x, y);
    if (i == 0)
      first = l;
    last = l;
  }
  EXPECT_LT(last, first) << "attention is not learning at all";
}

// ===========================================================================
// Embedding -- a learnable lookup table over Int32 ids
// ===========================================================================

// A {B, T} Int32 leaf of token ids. Never requires grad: ids are data.
static VarPtr ids(int64_t B, int64_t T, const std::vector<int32_t> &vals) {
  Tensor t({B, T}, DType::Int32, CPU);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  int32_t *p = t.data_ptr<int32_t>();
  for (size_t i = 0; i < vals.size(); ++i)
    p[i] = vals[i];
  return Variable::leaf(t, false);
}

static std::vector<float> to_vec(const Tensor &t) {
  Tensor c = t.contiguous();
  const float *p = c.data_ptr<float>();
  return std::vector<float>(p, p + c.numel());
}

TEST(EmbeddingTest, ExposesOneParameterNamedWeight) {
  torch::manual_seed(0);
  nn::Embedding emb(10, 4);
  EXPECT_EQ(emb.params().size(), 1u);
  EXPECT_NE(param_named(emb, "weight"), nullptr);
  EXPECT_EQ(param_named(emb, "weight")->data().shape(),
            std::vector<int64_t>({10, 4}));
}

TEST(EmbeddingTest, OutputShapeIsIdsShapePlusModelDim) {
  torch::manual_seed(0);
  nn::Embedding emb(10, 4);
  auto y = emb(ids(2, 3, {1, 5, 9, 0, 0, 7}));
  EXPECT_EQ(y->data().shape(), std::vector<int64_t>({2, 3, 4}));
  EXPECT_EQ(y->data().dtype(), DType::Float32);
}

TEST(EmbeddingTest, RowsMatchTheWeightTable) {
  // Output position (b, t) must be exactly row ids[b, t] of the weight.
  torch::manual_seed(0);
  const int64_t V = 6, D = 3;
  nn::Embedding emb(V, D);
  auto W = to_vec(param_named(emb, "weight")->data());

  std::vector<int32_t> toks = {5, 0, 0, 2};
  auto y = to_vec(emb(ids(2, 2, toks))->data());
  for (size_t i = 0; i < toks.size(); ++i)
    for (int64_t k = 0; k < D; ++k)
      EXPECT_FLOAT_EQ(y[i * D + k], W[toks[i] * D + k])
          << "position " << i << " dim " << k;
}

TEST(EmbeddingTest, GradientAccumulatesOnRepeatedIds) {
  // Use a target of (pred - 1) so dLoss/dPred is the same constant c at every
  // element. Then weight.grad row r == c * (# times r appears in ids) * ones.
  // So the row for an id used twice must be exactly double the row for an id
  // used once, and rows for ids never used must be zero. This pins "sum, not
  // average" without depending on MSE's normalisation constant.
  torch::manual_seed(0);
  const int64_t V = 5, D = 3;
  nn::Embedding emb(V, D);

  std::vector<int32_t> toks = {4, 0, 0, 2}; // 0 twice; 1 and 3 never
  auto pred = emb(ids(2, 2, toks));
  auto target_vals = to_vec(pred->data());
  for (auto &v : target_vals)
    v -= 1.0f;
  auto target = seq_input(2, 2, D, target_vals);

  MSE loss(pred, target);
  loss.backward();

  auto w = param_named(emb, "weight");
  ASSERT_TRUE(w->has_grad());
  auto g = to_vec(*w->grad());
  ASSERT_EQ(static_cast<int64_t>(g.size()), V * D);

  for (int64_t k = 0; k < D; ++k) {
    EXPECT_FLOAT_EQ(g[1 * D + k], 0.f) << "unused id 1 got gradient";
    EXPECT_FLOAT_EQ(g[3 * D + k], 0.f) << "unused id 3 got gradient";
    EXPECT_NE(g[2 * D + k], 0.f) << "used id 2 got no gradient";
    EXPECT_NEAR(g[0 * D + k], 2.f * g[2 * D + k], 1e-6f)
        << "id 0 appears twice, its grad row must be 2x a once-used row";
    EXPECT_NEAR(g[4 * D + k], g[2 * D + k], 1e-6f);
  }
}

TEST(EmbeddingTest, LearnsAToyLookupTable) {
  // 4 ids, 2 dims. Target: id i -> (i, -i). Pure table fitting; with a big
  // enough lr the table should converge essentially exactly.
  torch::manual_seed(0);
  nn::Embedding emb(4, 2);
  auto x = ids(1, 4, {0, 1, 2, 3});
  auto y = seq_input(1, 4, 2, {0, 0, 1, -1, 2, -2, 3, -3});

  SGD opt(emb.params(), 0.5f);
  float first = step(emb, opt, x, y);
  float last = first;
  for (int i = 0; i < 200; ++i)
    last = step(emb, opt, x, y);
  EXPECT_LT(last, first * 1e-3f) << "embedding did not fit a 4-row table";

  auto W = to_vec(param_named(emb, "weight")->data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(W[i * 2 + 0], static_cast<float>(i), 1e-2f);
    EXPECT_NEAR(W[i * 2 + 1], -static_cast<float>(i), 1e-2f);
  }
}

TEST(EmbeddingTest, RejectsFloatIds) {
  torch::manual_seed(0);
  nn::Embedding emb(4, 2);
  EXPECT_THROW(emb(seq_input(1, 2, 1, {0.f, 1.f})), std::invalid_argument);
}

// ===========================================================================
// MultiHeadAttention with causal=false -- the UNet needs bidirectional
// ===========================================================================

TEST(MhaTest, NonCausalEarlyPositionsSeeLaterOnes) {
  // Mirror image of IsCausal: perturb the LAST position and require that
  // position 0's output changes. A hardcoded causal flag anywhere in forward
  // (CPU mask or the flash call) fails this.
  const int64_t B = 1, T = 5, D = 8;
  torch::manual_seed(0);
  nn::MultiHeadAttention mha(D, 2, T, DType::Float32, CPU, /*causal=*/false);

  auto base = varied(B * T * D);
  auto out0 = to_vec(mha(seq_input(B, T, D, base))->data());

  auto bumped = base;
  for (int64_t k = 0; k < D; ++k)
    bumped[(T - 1) * D + k] += 5.0f;
  auto out1 = to_vec(mha(seq_input(B, T, D, bumped))->data());

  bool changed = false;
  for (int64_t k = 0; k < D; ++k)
    changed |= (out0[k] != out1[k]);
  EXPECT_TRUE(changed)
      << "non-causal attention: position 0 ignored a change at the last position";
}

TEST(MhaTest, NonCausalPreservesShapeAndIsFinite) {
  const int64_t B = 2, T = 4, D = 8;
  torch::manual_seed(0);
  nn::MultiHeadAttention mha(D, 2, T, DType::Float32, CPU, false);
  auto y = mha(seq_input(B, T, D, varied(B * T * D)));
  EXPECT_EQ(y->data().shape(), std::vector<int64_t>({B, T, D}));
  for (float v : to_vec(y->data()))
    EXPECT_TRUE(std::isfinite(v));
}

TEST(MhaTest, CausalDefaultStillHoldsAfterRefactor) {
  // Guard the default: constructing without the flag must remain causal.
  const int64_t B = 1, T = 4, D = 8;
  torch::manual_seed(0);
  nn::MultiHeadAttention mha(D, 2, T);
  auto base = varied(B * T * D);
  auto out0 = to_vec(mha(seq_input(B, T, D, base))->data());
  auto bumped = base;
  for (int64_t k = 0; k < D; ++k)
    bumped[(T - 1) * D + k] += 5.0f;
  auto out1 = to_vec(mha(seq_input(B, T, D, bumped))->data());
  for (int64_t i = 0; i < (T - 1) * D; ++i)
    EXPECT_FLOAT_EQ(out0[i], out1[i]);
}

// ===========================================================================
// Transformer end to end -- ids in, logits out
// ===========================================================================

static constexpr int64_t XV = 11; // vocab
static constexpr int64_t XD = 8;  // d_model
static constexpr int64_t XF = 16; // d_ff
static constexpr int64_t XL = 1;  // blocks
static constexpr int64_t XH = 2;  // heads
static constexpr int64_t XT = 6;  // max context

TEST(TransformerSmokeTest, MapsIdsToLogits) {
  torch::manual_seed(0);
  nn::Transformer model(XV, XD, XF, XL, XH, XT);
  auto logits = model(ids(2, 4, {1, 2, 3, 4, 10, 0, 0, 5}));
  EXPECT_EQ(logits->data().shape(), std::vector<int64_t>({2, 4, XV}));
  for (float v : to_vec(logits->data()))
    EXPECT_TRUE(std::isfinite(v));
}

TEST(TransformerSmokeTest, EmbedAndUnembedAreRegisteredParameters) {
  // If Transformer forgot to register_module the embedding or the unembed
  // Linear, they'd still work in forward but SGD would never update them.
  torch::manual_seed(0);
  nn::Transformer model(XV, XD, XF, XL, XH, XT);
  bool saw_embed = false, saw_unembed = false;
  for (auto &p : model.params()) {
    auto s = p->data().shape();
    if (s == std::vector<int64_t>({XV, XD}))
      saw_embed = true;
    if (s == std::vector<int64_t>({XD, XV}))
      saw_unembed = true;
  }
  EXPECT_TRUE(saw_embed) << "no (vocab, d_model) parameter: embedding not registered";
  EXPECT_TRUE(saw_unembed) << "no (d_model, vocab) parameter: unembed not registered";
}

TEST(TransformerSmokeTest, BackwardReachesEveryParameter) {
  torch::manual_seed(0);
  nn::Transformer model(XV, XD, XF, XL, XH, XT);
  auto x = ids(1, 4, {1, 2, 3, 4});
  auto target = seq_input(1, 4, XV, varied(4 * XV));
  MSE loss(model(x), target);
  loss.backward();
  for (auto &[name, p] : model.named_params())
    EXPECT_TRUE(p->has_grad()) << "no gradient on " << name;
}

TEST(TransformerSmokeTest, WholeModelIsCausal) {
  // Change the last token id; logits at earlier positions must not move.
  torch::manual_seed(0);
  nn::Transformer model(XV, XD, XF, XL, XH, XT);
  auto l0 = to_vec(model(ids(1, 4, {1, 2, 3, 4}))->data());
  auto l1 = to_vec(model(ids(1, 4, {1, 2, 3, 9}))->data());
  for (int64_t i = 0; i < 3 * XV; ++i)
    EXPECT_FLOAT_EQ(l0[i], l1[i]) << "earlier logit moved when last id changed";
  bool last_changed = false;
  for (int64_t i = 3 * XV; i < 4 * XV; ++i)
    last_changed |= (l0[i] != l1[i]);
  EXPECT_TRUE(last_changed) << "last position's logits ignore its own id";
}

TEST(TransformerSmokeTest, OneStepReducesLoss) {
  torch::manual_seed(0);
  nn::Transformer model(XV, XD, XF, XL, XH, XT);
  auto x = ids(1, 4, {1, 2, 3, 4});
  auto target = seq_input(1, 4, XV, varied(4 * XV));
  SGD opt(model.params(), 1e-3f);
  float first = step(model, opt, x, target);
  float second = step(model, opt, x, target);
  EXPECT_LT(second, first);
}

TEST(TransformerSmokeTest, ReportGradientScalePerParameter) {
  // Diagnostic, not a pass/fail gate: prints |grad| / |param| for every
  // parameter after one backward on the smoke fixture. A ratio far above the
  // others says that parameter will move disproportionately under plain SGD.
  torch::manual_seed(0);
  nn::Transformer model(XV, XD, XF, XL, XH, XT);
  auto x = ids(1, 4, {1, 2, 3, 4});
  auto target = seq_input(1, 4, XV, varied(4 * XV));
  MSE loss(model(x), target);
  loss.backward();
  for (auto &[name, p] : model.named_params()) {
    auto w = to_vec(p->data());
    auto g = to_vec(*p->grad());
    double nw = 0, ng = 0;
    for (float v : w) nw += double(v) * v;
    for (float v : g) ng += double(v) * v;
    std::printf("  %-28s |w|=%9.4f |g|=%9.4f  |g|/|w|=%8.2f\n", name.c_str(),
                std::sqrt(nw), std::sqrt(ng), std::sqrt(ng) / std::sqrt(nw));
  }
  std::printf("  initial loss %.4f\n", loss.loss());
}

// ===========================================================================
// Adam -- bias-corrected moments, sign-like first step, eps on zero grads
// ===========================================================================

TEST(AdamTest, FirstStepIsLrTimesSignOfGradient) {
  // With bias correction, m_hat = g and v_hat = g^2 at t = 1, so the update is
  // exactly lr * sign(g) per element regardless of gradient scale. Without
  // correction it's ~3.16 * lr; with the correction factors inverted it's
  // 10 * lr. The zero-gradient element exercises the eps path (0/0 -> NaN
  // without it) and must not move.
  Tensor t({4}, DType::Float32, CPU);
  float *p = t.data_ptr<float>();
  p[0] = p[1] = p[2] = p[3] = 1.0f;
  auto w = Variable::leaf(t, true);

  Tensor g({4}, DType::Float32, CPU);
  float *gp = g.data_ptr<float>();
  gp[0] = 1.f; gp[1] = -2.f; gp[2] = 1000.f; gp[3] = 0.f;
  w->accumulate_grad(g);

  torch::Adam opt({w}, 0.1);
  opt.step();

  auto got = to_vec(w->data());
  EXPECT_NEAR(got[0], 0.9f, 1e-5f) << "positive grad should move down by lr";
  EXPECT_NEAR(got[1], 1.1f, 1e-5f) << "negative grad should move up by lr";
  EXPECT_NEAR(got[2], 0.9f, 1e-5f) << "huge grad must still move by exactly lr";
  EXPECT_FLOAT_EQ(got[3], 1.0f) << "zero grad must not move (and not be NaN)";
  for (float v : got)
    EXPECT_TRUE(std::isfinite(v));
}

TEST(AdamTest, ConstantGradientKeepsStepSizeAtLr) {
  // Feed the same gradient for several steps. With correct bias correction
  // m_hat / sqrt(v_hat) stays exactly sign(g), so each step moves by lr and
  // the parameter walks in a straight line: w_t = w_0 - t * lr.
  Tensor t({1}, DType::Float32, CPU);
  t.data_ptr<float>()[0] = 0.f;
  auto w = Variable::leaf(t, true);
  torch::Adam opt({w}, 0.01);

  for (int step = 1; step <= 20; ++step) {
    w->zero_grad();
    Tensor g({1}, DType::Float32, CPU);
    g.data_ptr<float>()[0] = 3.f;
    w->accumulate_grad(g);
    opt.step();
    EXPECT_NEAR(w->data().data_ptr<float>()[0], -0.01f * step, 1e-5f)
        << "after step " << step;
  }
}

TEST(AdamTest, RejectsEmptyParameterList) {
  EXPECT_THROW(torch::Adam({}, 0.1), std::invalid_argument);
}

TEST(AdamTest, ThrowsIfStepCalledWithoutBackward) {
  auto w = Variable::leaf(Tensor::ones({2}), true);
  torch::Adam opt({w}, 0.1);
  EXPECT_THROW(opt.step(), std::logic_error);
}

TEST(AdamTest, BeatsSgdOnTheEmbeddingGradientScaleProblem) {
  // The smoke fixture where plain SGD at lr=1e-3 raised the loss because the
  // embedding's gradient was ~600x its weight norm. Adam's per-element
  // normalisation should make one step reduce the loss at a sane lr.
  torch::manual_seed(0);
  nn::Transformer model(XV, XD, XF, XL, XH, XT);
  auto x = ids(1, 4, {1, 2, 3, 4});
  auto target = seq_input(1, 4, XV, varied(4 * XV));

  torch::Adam opt(model.params(), 1e-3);
  float first, second;
  {
    opt.zero_grad();
    MSE loss(model(x), target);
    first = loss.loss();
    loss.backward();
    opt.step();
  }
  {
    opt.zero_grad();
    MSE loss(model(x), target);
    second = loss.loss();
    loss.backward();
    opt.step();
  }
  EXPECT_LT(second, first);
}
