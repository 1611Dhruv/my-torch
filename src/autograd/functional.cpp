#include "mytorch/autograd.h"
#include <stack>
#include <unordered_set>

namespace torch {
namespace autograd {

void Variable::accumulate_grad(const Tensor &g) {
  std::vector<int64_t> reduce_along;
  int64_t offset = g.ndim() - _t.ndim();
  for (int64_t i = 0; i < offset; i++) {
    reduce_along.push_back(i);
  }
  for (int64_t i = 0; i < _t.ndim(); i++) {
    if (_t.shape()[i] == 1 && g.shape()[i + offset] != 1) {
      reduce_along.push_back(i + offset);
    }
  }

  Tensor ng = (!reduce_along.empty() ? torch::sum(g, reduce_along, true) : g).reshape(_t.shape());
  if (!_grad) {
    _grad = ng;
  } else {
    _grad = torch::add(*_grad, ng);
  }
}
void Variable::backward() {
  // Topo sort and then accumulate

  // If inputs are empty or grad doesn't have value take lite

  // Otherwise start exploring
  std::stack<Variable *> order;
  std::unordered_set<Variable *> seen;
  std::stack<std::pair<Variable *, bool>> explore;

  // Seed this grad to ONES
  _grad = Tensor::ones_like(_t);

  explore.push({this, false});
  while (!explore.empty()) {
    auto [node, finalize] = explore.top();
    explore.pop();
    // Node already finalized, no need to finalize again
    if (seen.count(node))
      continue;

    if (finalize) {
      order.push(node);
      seen.insert(node);
    } else {
      explore.push({node, true});
      for (const auto &input : node->_inputs) {
        explore.push({input.get(), false});
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

std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b) {
  auto backward = [a, b](const Tensor &g) -> void {
    a->accumulate_grad(torch::matmul(g, b->data().transpose(-1, -2)));
    b->accumulate_grad(torch::matmul(a->data().transpose(-1, -2), g));
  };
  return Variable::fromOp(torch::matmul(a->data(), b->data()), {a, b}, backward);
}

} // namespace autograd
} // namespace torch
