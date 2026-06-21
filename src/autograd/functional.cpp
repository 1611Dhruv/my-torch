#include "mytorch/autograd.h"
#include <stack>
#include <unordered_set>

namespace torch {
namespace autograd {

void Variable::backward() {
  // Topo sort and then accumulate

  // If inputs are empty or grad doesn't have value take lite

  // Otherwise start exploring
  std::stack<Variable *> order;
  std::unordered_set<Variable *> seen;
  std::stack<std::pair<Variable *, bool>> explore;

  explore.push({this, false});
  seen.insert(this);
  while (!explore.empty()) {
    auto [node, finalize] = explore.top();
    explore.pop();

    if (finalize) {
      order.push(node);
    } else {
      explore.push({node, true});
      for (const auto &input : node->_inputs) {
        if (seen.count(input.get()))
          continue;
        explore.push({input.get(), false});
        seen.insert(input.get());
      }
    }
  }

  while (!order.empty()) {
    auto curr = order.top();
    order.pop();

    if (curr->_inputs.empty() || !curr->has_grad())
      continue;
    curr->_backward(curr->_grad.value());
  }
}

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
