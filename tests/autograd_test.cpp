// Spec-by-tests for torch::autograd (add/sub/mult, then backward once the
// engine lands).
//
// Forward-value tests are runnable now. Gradient tests are stubbed + commented
// until Variable::backward() exists (the engine that seeds the output grad and
// walks the graph invoking each node's recorded _backward closure).
//
// Reading values: data() returns a CONST Tensor& and at<T>() is non-const, so
// read through the const data_ptr<T>() (see to_vec). Build inputs by writing
// at<T>() on a fresh (non-const) Tensor.
//
// Run:  ctest --test-dir build -R Autograd

#include "mytorch/autograd.h"
#include "mytorch/tensor.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using torch::Tensor;
using torch::autograd::Variable;
namespace ag = torch::autograd;

// 1-D float leaf from literal values.
static std::shared_ptr<Variable> leaf1d(std::initializer_list<float> vals) {
  Tensor t({static_cast<int64_t>(vals.size())});
  int64_t i = 0;
  for (float v : vals)
    t.at<float>({i++}) = v;
  return Variable::leaf(t);
}

static std::shared_ptr<Variable>
leaf_rand(std::initializer_list<int64_t> shape,
          torch::Device dev = torch::Device::CPU) {
  return Variable::leaf(torch::Tensor::randn(shape, dev, 6, 7));
}

// Flatten a contiguous tensor's values for easy comparison.
static std::vector<float> to_vec(const Tensor &t) {
  const float *p = t.data_ptr<float>();
  return std::vector<float>(p, p + t.numel());
}

static void expect_close(std::vector<float> &&t1, std::vector<float> &&t2) {
  EXPECT_EQ(t1.size(), t2.size());
  size_t N = t1.size();
  float thresh = 1e-5;
  for (int i = 0; i < N; i++) {
    EXPECT_NEAR(t1[i], t2[i], thresh);
  }
}

// --- Numerical Analyis ------------------------------------------
constexpr double eps = 6e-6;

template <typename Op>
void gradient_check(std::string name, Op fn, std::vector<torch::Tensor> inputs,
                    double atol = 1e-8, double rtol = 1e-5) {

  // Transform inputs into cool dudes once
  std::vector<torch::autograd::VarPtr> vars(inputs.size());
  std::ranges::transform(inputs, vars.begin(), [](auto t) {
    return torch::autograd::Variable::leaf(t, true);
  });

  // Perform our autograd to get a partial result
  auto out = fn(vars);
  auto R = torch::autograd::Variable::leaf(
      torch::Tensor::randn_like_hp(out->data()), false);

  // Manually compute this analytical loss
  auto L = torch::autograd::sum(torch::autograd::mult(out, R), {});
  // Populate the gradients
  L->backward();

  // Get the gradient
  auto eval = [&]() -> double {
    torch::autograd::VarPtr loss =
        torch::autograd::sum(torch::autograd::mult(R, fn(vars)), {});
    return loss->data().data_ptr<double>()[0];
  };

  int64_t N = inputs.size();

  for (int64_t i = 0; i < N; i++) {
    // Get a copy of the gradient
    if (!vars[i]->grad().has_value()) {
      ADD_FAILURE() << name << ": Expected grad of at input idx[" << i
                    << "] to have a value\n";
      continue;
    }
    auto grad = *vars[i]->grad();
    auto &t = inputs[i];
    auto dptr = t.data_ptr<double>();
    for (int64_t j = 0; j < t.numel(); j++) {
      auto old = dptr[j];

      dptr[j] = old + eps;
      auto f1 = eval();

      dptr[j] = old - eps;
      auto f2 = eval();

      dptr[j] = old;

      auto g_num = (f1 - f2) / (2 * eps);
      auto g_anal = grad.data_ptr<double>()[j];
      // Mixed tolerance
      if (std::fabs(g_num - g_anal) >
          atol + rtol * std::max(std::fabs(g_num), std::fabs(g_anal))) {
        FAIL() << name << ": Grad check for param [" << i << "," << j
               << "] failed";
      }
    }
  }
}

TEST(AutogradNumerical, Add) {
  auto a = torch::Tensor::rand({2, 3, 4}).to(torch::DType::Float64,
                                             torch::Device::CPU);
  auto b = torch::Tensor::rand({2, 3, 4}).to(torch::DType::Float64,
                                             torch::Device::CPU);
  gradient_check("add",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::add(inps[0], inps[1]);
                 },
                 {a, b});
}

TEST(AutogradNumerical, Sub) {
  auto a = torch::Tensor::rand({2, 3, 4}).to(torch::DType::Float64,
                                             torch::Device::CPU);
  auto b = torch::Tensor::rand({2, 3, 4}).to(torch::DType::Float64,
                                             torch::Device::CPU);
  gradient_check("sub",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::sub(inps[0], inps[1]);
                 },
                 {a, b});
}

TEST(AutogradNumerical, Mult) {
  auto a = torch::Tensor::rand({2, 3, 4}).to(torch::DType::Float64,
                                             torch::Device::CPU);
  auto b = torch::Tensor::rand({2, 3, 4}).to(torch::DType::Float64,
                                             torch::Device::CPU);
  gradient_check("mult",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::mult(inps[0], inps[1]);
                 },
                 {a, b});
}

TEST(AutogradNumerical, Div) {
  auto a = torch::Tensor::rand({2, 3, 4}).to(torch::DType::Float64,
                                             torch::Device::CPU);
  auto b = torch::Tensor::randn({2, 3, 4}, torch::Device::CPU, 100, 1)
               .to(torch::DType::Float64, torch::Device::CPU);
  gradient_check(
      "div",
      [&](std::vector<torch::autograd::VarPtr> inps) {
        return torch::autograd::div(inps[0], inps[1]);
      },
      {a, b}, 1e-6, 1e-5);
}

TEST(AutogradNumerical, Transp) {
  auto a = torch::Tensor::rand({2, 3, 4}).to(torch::DType::Float64,
                                             torch::Device::CPU);
  gradient_check("transp",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::transpose(inps[0], 0, -1);
                 },
                 {a});
}

TEST(AutogradNumerical, Reshape) {
  auto a = torch::Tensor::rand({2, 3, 4}).to(torch::DType::Float64,
                                             torch::Device::CPU);
  gradient_check("reshape",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::reshape(inps[0], {3, 8});
                 },
                 {a});
}

TEST(AutogradNumerical, Sum) {
  auto a = torch::Tensor::randn({2, 3, 4}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  gradient_check("sum",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::sum(inps[0], {0, 2});
                 },
                 {a});
}

TEST(AutogradNumerical, Matmul) {
  auto a = torch::Tensor::randn({2, 3, 4}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  auto b = torch::Tensor::randn({2, 4, 3}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  gradient_check("sum",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::matmul(inps[0], inps[1]);
                 },
                 {a, b});
}

TEST(AutogradNumerical, ReLU) {
  auto a = torch::Tensor::randn({2, 3, 4}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  gradient_check("relu",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::relu(inps[0]);
                 },
                 {a});
}

TEST(AutogradNumerical, Neg) {
  auto a = torch::Tensor::randn({2, 3, 4}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  gradient_check("neg",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::neg(inps[0]);
                 },
                 {a});
}

TEST(AutogradNumerical, Sin) {
  auto a = torch::Tensor::randn({2, 3, 4}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  gradient_check("sin",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::sin(inps[0]);
                 },
                 {a});
}

TEST(AutogradNumerical, Cos) {
  auto a = torch::Tensor::randn({2, 3, 4}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  gradient_check("cos",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::cos(inps[0]);
                 },
                 {a});
}

TEST(AutogradNumerical, Exp) {
  auto a = torch::Tensor::randn({2, 3, 4}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  gradient_check("cos",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::exp(inps[0]);
                 },
                 {a});
}

TEST(AutogradNumerical, Ln) {
  auto a = torch::Tensor::randn({2, 3, 4}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  gradient_check("ln",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::ln(inps[0]);
                 },
                 {a});
}

TEST(AutogradNumerical, Max) {
  auto a = torch::Tensor::randn({2, 3, 4}).to(torch::DType::Float64,
                                              torch::Device::CPU);
  gradient_check("max",
                 [&](std::vector<torch::autograd::VarPtr> inps) {
                   return torch::autograd::max(inps[0], {0, 2});
                 },
                 {a});
}

// --- forward values (runnable now) ------------------------------------------

TEST(AutogradForward, Add) {
  auto a = leaf1d({1, 2, 3});
  auto b = leaf1d({10, 20, 30});
  auto c = ag::add(a, b);
  expect_close(to_vec(c->data()), (std::vector<float>{11, 22, 33}));
  // forward must not mutate the inputs
  expect_close(to_vec(a->data()), (std::vector<float>{1, 2, 3}));
  expect_close(to_vec(b->data()), (std::vector<float>{10, 20, 30}));
}

TEST(AutogradForward, Sub) {
  auto a = leaf1d({1, 2, 3});
  auto b = leaf1d({10, 20, 30});

  auto c = ag::sub(a, b);
  expect_close(to_vec(c->data()), (std::vector<float>{-9, -18, -27}));
  // forward must not mutate the inputs
  expect_close(to_vec(a->data()), (std::vector<float>{1, 2, 3}));
  expect_close(to_vec(b->data()), (std::vector<float>{10, 20, 30}));
}

TEST(AutogradForward, Mult) {
  auto a = leaf1d({1, 2, 3});
  auto b = leaf1d({10, 20, 30});

  auto c = ag::mult(a, b);
  expect_close(to_vec(c->data()), (std::vector<float>{10, 40, 90}));
  // forward must not mutate the inputs
  expect_close(to_vec(a->data()), (std::vector<float>{1, 2, 3}));
  expect_close(to_vec(b->data()), (std::vector<float>{10, 20, 30}));
}

// --- gradients (uncomment once Variable::backward() exists) ------------------
//
TEST(AutogradBackward, AddGrad) {
  // c = add(a, b); c->backward();
  // grad flows through unchanged -> a.grad and b.grad are all-ones.
  auto a = leaf1d({1, 2, 3});
  auto b = leaf1d({10, 20, 30});

  auto c = ag::add(a, b);
  c->backward();

  EXPECT_TRUE(c->grad().has_value());
  EXPECT_TRUE(a->grad().has_value());
  EXPECT_TRUE(b->grad().has_value());

  expect_close(to_vec(c->grad().value()), (std::vector<float>{1, 1, 1}));
  expect_close(to_vec(a->grad().value()), (std::vector<float>{1, 1, 1}));
  expect_close(to_vec(b->grad().value()), (std::vector<float>{1, 1, 1}));
}

TEST(AutogradBackward, MultGrad) {
  // c = mult(a, b); c->backward();
  // dc/da = b, dc/db = a.
  auto a = leaf1d({1, 2, 3});
  auto b = leaf1d({10, 20, 30});

  auto c = ag::mult(a, b);
  c->backward();

  EXPECT_TRUE(c->grad().has_value());
  EXPECT_TRUE(a->grad().has_value());
  EXPECT_TRUE(b->grad().has_value());

  expect_close(to_vec(c->grad().value()), (std::vector<float>{1, 1, 1}));
  expect_close(to_vec(a->grad().value()), (std::vector<float>{10, 20, 30}));
  expect_close(to_vec(b->grad().value()), (std::vector<float>{1, 2, 3}));
}

TEST(AutogradBackward, FanOutAccumulates) {
  // y = mult(x, x)  (x used twice) -> dy/dx = 2x.
  // This is the accumulation case a naive (non-topological) walk gets wrong.
  auto x = leaf1d({1, 2, 3});

  auto y = ag::mult(x, x);
  y->backward();

  EXPECT_TRUE(y->grad().has_value());
  EXPECT_TRUE(x->grad().has_value());

  expect_close(to_vec(y->grad().value()), (std::vector<float>{1, 1, 1}));
  expect_close(to_vec(x->grad().value()), (std::vector<float>{2, 4, 6}));
}

// -- Testing combined autograds
TEST(AutogradCombined, BigDiamond) {
  auto x = leaf1d({0.5, 0.3, 0.2});

  auto a = ag::add(x, x);
  auto b = ag::mult(a, x);
  auto c = ag::sub(b, a);
  auto d = ag::mult(x, x);
  auto e = ag::add(d, c);

  e->backward();
  // Check forward
  expect_close(to_vec(x->data()), (std::vector<float>{0.5, 0.3, 0.2}));
  expect_close(to_vec(a->data()), (std::vector<float>{1.0, 0.6, 0.4}));
  expect_close(to_vec(b->data()), (std::vector<float>{0.5, 0.18, 0.08}));
  expect_close(to_vec(c->data()), (std::vector<float>{-0.5, -0.42, -0.32}));
  expect_close(to_vec(d->data()), (std::vector<float>{0.25, 0.09, 0.04}));
  expect_close(to_vec(e->data()), (std::vector<float>{-0.25, -0.33, -0.28}));

  // Check backward
  expect_close(to_vec(a->grad().value()),
               (std::vector<float>{-0.5, -0.7, -0.8}));
  expect_close(to_vec(b->grad().value()), (std::vector<float>{1.0, 1.0, 1.0}));
  expect_close(to_vec(c->grad().value()), (std::vector<float>{1.0, 1.0, 1.0}));
  expect_close(to_vec(d->grad().value()), (std::vector<float>{1.0, 1.0, 1.0}));
  expect_close(to_vec(e->grad().value()), (std::vector<float>{1.0, 1.0, 1.0}));
  expect_close(to_vec(x->grad().value()),
               (std::vector<float>{1.0, -0.2, -0.8}));
}

// --- broadcast gradients ----------------------------------------------------
//
// Once elementwise ops broadcast, an operand's gradient no longer has the same
// shape as the operand: the output grad has to be SUMMED back down to the
// operand's shape (accumulate_grad does this). Broadcasting stretched a value
// across several outputs, so every one of those outputs contributes to it.
//
// Two distinct things get undone, and the bugs live in the second:
//   rank-only   {4} vs {3,4}     -- a whole leading axis was prepended
//   size-1      {1,4} vs {3,4}   -- an existing axis was expanded, rank equal
// The nasty case is BOTH at once ({1,4} vs {B,T,4}), because the axis indices
// then live in two different spaces and off-by-`offset` errors are silent.

// A leaf of arbitrary shape, filled from flat row-major values.
static std::shared_ptr<Variable> leafnd(const std::vector<int64_t> &shape,
                                        const std::vector<float> &vals) {
  Tensor t(shape, torch::DType::Float32, torch::CPU);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  std::copy(vals.begin(), vals.end(), t.data_ptr<float>());
  return Variable::leaf(t);
}

static std::vector<float> ones(int64_t n) {
  return std::vector<float>(n, 1.0f);
}

TEST(AutogradBroadcast, RankOnlyBroadcastSumsGradToOperandShape) {
  // {3,4} + {4}: the bare vector is reused across all 3 rows, so each of its
  // entries receives 3 units of gradient.
  auto x = leafnd({3, 4}, ones(12));
  auto v = leafnd({4}, ones(4));

  auto y = ag::add(x, v);
  ASSERT_EQ(y->data().shape(), std::vector<int64_t>({3, 4}));
  y->backward();

  ASSERT_TRUE(v->grad().has_value());
  EXPECT_EQ(v->grad()->shape(), std::vector<int64_t>({4}));
  expect_close(to_vec(v->grad().value()), (std::vector<float>{3, 3, 3, 3}));

  // The un-broadcast operand is untouched: same shape, all ones.
  EXPECT_EQ(x->grad()->shape(), std::vector<int64_t>({3, 4}));
  expect_close(to_vec(x->grad().value()), ones(12));
}

TEST(AutogradBroadcast, SizeOneAxisSumsGradToOperandShape) {
  // {3,4} + {1,4}: same rank, axis 0 expanded 1 -> 3. offset == 0 here, so
  // this isolates the size-1 half of the rule.
  auto x = leafnd({3, 4}, ones(12));
  auto b = leafnd({1, 4}, ones(4));

  auto y = ag::add(x, b);
  ASSERT_EQ(y->data().shape(), std::vector<int64_t>({3, 4}));
  y->backward();

  ASSERT_TRUE(b->grad().has_value());
  EXPECT_EQ(b->grad()->shape(), std::vector<int64_t>({1, 4}));
  expect_close(to_vec(b->grad().value()), (std::vector<float>{3, 3, 3, 3}));
}

TEST(AutogradBroadcast, RankAndSizeOneTogetherIsTheBiasShape) {
  // THE case: bias {1,out} meeting activations {B,T,out}. offset == 1 AND the
  // bias has a size-1 axis, so the reduced-dim list spans both index spaces.
  // Pushing the size-1 axis in the operand's space instead of the gradient's
  // yields {1,T,out}, which then fails to reshape back to {1,out}.
  const int64_t B = 2, T = 3, OUT = 4;
  auto acts = leafnd({B, T, OUT}, ones(B * T * OUT));
  auto bias = leafnd({1, OUT}, ones(OUT));

  auto y = ag::add(acts, bias);
  ASSERT_EQ(y->data().shape(), std::vector<int64_t>({B, T, OUT}));
  y->backward();

  ASSERT_TRUE(bias->grad().has_value());
  EXPECT_EQ(bias->grad()->shape(), std::vector<int64_t>({1, OUT}));
  // Each bias entry is reused across every (b, t) pair -> B * T units.
  expect_close(to_vec(bias->grad().value()),
               (std::vector<float>{B * T, B * T, B * T, B * T}));
}

TEST(AutogradBroadcast, BroadcastGradFlowsThroughMult) {
  // mult's backward multiplies by the other operand before accumulating, so the
  // reduction has to happen on a grad that is NOT all-ones.
  auto x = leafnd({2, 3}, {1, 2, 3, 4, 5, 6});
  auto v = leafnd({3}, {10, 20, 30});

  auto y = ag::mult(x, v);
  y->backward();

  // dL/dv[j] = sum over rows of x[i][j]  ->  {1+4, 2+5, 3+6}
  ASSERT_TRUE(v->grad().has_value());
  EXPECT_EQ(v->grad()->shape(), std::vector<int64_t>({3}));
  expect_close(to_vec(v->grad().value()), (std::vector<float>{5, 7, 9}));

  // dL/dx[i][j] = v[j], broadcast back out
  expect_close(to_vec(x->grad().value()),
               (std::vector<float>{10, 20, 30, 10, 20, 30}));
}
