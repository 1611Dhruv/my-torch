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
  expect_close(to_vec(a->grad().value()), (std::vector<float>{-0.5, -0.7, -0.8}));
  expect_close(to_vec(b->grad().value()), (std::vector<float>{1.0, 1.0, 1.0}));
  expect_close(to_vec(c->grad().value()), (std::vector<float>{1.0, 1.0, 1.0}));
  expect_close(to_vec(d->grad().value()), (std::vector<float>{1.0, 1.0, 1.0}));
  expect_close(to_vec(e->grad().value()), (std::vector<float>{1.0, 1.0, 1.0}));
  expect_close(to_vec(x->grad().value()), (std::vector<float>{1.0, -0.2, -0.8}));
}
