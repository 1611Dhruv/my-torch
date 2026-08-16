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
