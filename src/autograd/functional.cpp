#include "mytorch/autograd.h"

namespace torch {
namespace autograd {

std::shared_ptr<Variable> add(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
  // Copy the backward by value and not reference (Reference ends at the end of endop)
  // Copy the reference instead
  auto backward = [a, b](const Tensor &g) -> void {
    a->accumulate_grad(g);
    b->accumulate_grad(g);
  };
  return Variable::fromOp(torch::add(a->data(), b->data()), {a, b}, backward);
}

std::shared_ptr<Variable> sub(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
  auto backward = [a, b](const Tensor &g) -> void {
    a->accumulate_grad(g);
    b->accumulate_grad(torch::neg(g));
  };
  return Variable::fromOp(torch::sub(a->data(), b->data()), {a, b}, backward);
}

std::shared_ptr<Variable> mult(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
  auto backward = [a, b](const Tensor &g) -> void {
    a->accumulate_grad(torch::mult(b->data(), g));
    b->accumulate_grad(torch::mult(a->data(), g));
  };
  return Variable::fromOp(torch::mult(a->data(), b->data()), {a, b}, backward);
}

} // namespace autograd
} // namespace torch
