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

// --- forward values (runnable now) ------------------------------------------

TEST(AutogradForward, Add) {
  auto a = leaf1d({1, 2, 3});
  auto b = leaf1d({10, 20, 30});
  auto c = ag::add(a, b);
  EXPECT_EQ(to_vec(c->data()), (std::vector<float>{11, 22, 33}));
  // forward must not mutate the inputs
  EXPECT_EQ(to_vec(a->data()), (std::vector<float>{1, 2, 3}));
  EXPECT_EQ(to_vec(b->data()), (std::vector<float>{10, 20, 30}));
}

TEST(AutogradForward, Sub) {
  // TODO: c = sub(a, b); expect elementwise a - b.
}

TEST(AutogradForward, Mult) {
  // TODO: c = mult(a, b); expect elementwise a * b.
}

// --- gradients (uncomment once Variable::backward() exists) ------------------
//
// TEST(AutogradBackward, AddGrad) {
//   // c = add(a, b); c->backward();
//   // grad flows through unchanged -> a.grad and b.grad are all-ones.
// }
//
// TEST(AutogradBackward, MultGrad) {
//   // c = mult(a, b); c->backward();
//   // dc/da = b, dc/db = a.
// }
//
// TEST(AutogradBackward, FanOutAccumulates) {
//   // y = mult(x, x)  (x used twice) -> dy/dx = 2x.
//   // This is the accumulation case a naive (non-topological) walk gets wrong.
// }
