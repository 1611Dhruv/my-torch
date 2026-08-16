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

  Tensor ng = (!reduce_along.empty() ? torch::sum(g, reduce_along, true) : g)
                  .reshape(_t.shape());
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

VarPtr add(VarPtr a, VarPtr b) {
  // Copy the backward by value and not reference (Reference ends at the end of
  // endop) Copy the reference instead
  auto backward = [a, b](const Tensor &g) -> void {
    a->accumulate_grad(g);
    b->accumulate_grad(g);
  };
  return Variable::fromOp(torch::add(a->data(), b->data()), {a, b}, backward);
}

VarPtr sub(VarPtr a, VarPtr b) {
  auto backward = [a, b](const Tensor &g) -> void {
    a->accumulate_grad(g);
    b->accumulate_grad(torch::neg(g));
  };
  return Variable::fromOp(torch::sub(a->data(), b->data()), {a, b}, backward);
}

VarPtr mult(VarPtr a, VarPtr b) {
  auto backward = [a, b](const Tensor &g) -> void {
    a->accumulate_grad(torch::mult(b->data(), g));
    b->accumulate_grad(torch::mult(a->data(), g));
  };
  return Variable::fromOp(torch::mult(a->data(), b->data()), {a, b}, backward);
}

VarPtr div(VarPtr a, VarPtr b) {
  // NOTE: This is where kernel fusion would be nice...
  auto backward = [a, b](const Tensor &g) -> void {
    a->accumulate_grad(torch::div(g, b->data()));
    // -g * a / b^2
    b->accumulate_grad(torch::mult(
        g,
        torch::neg(torch::div(a->data(), torch::mult(b->data(), b->data())))));
  };
  return Variable::fromOp(torch::div(a->data(), b->data()), {a, b}, backward);
}

VarPtr matmul(VarPtr a, VarPtr b) {
  auto backward = [a, b](const Tensor &g) -> void {
    a->accumulate_grad(torch::matmul(g, b->data().transpose(-1, -2)));
    b->accumulate_grad(torch::matmul(a->data().transpose(-1, -2), g));
  };
  return Variable::fromOp(torch::matmul(a->data(), b->data()), {a, b},
                          backward);
}

VarPtr sum(VarPtr a, std::vector<int64_t> dims) {
  auto kept_dim = torch::sum(a->data(), dims, true);

  auto backward = [a, kept_dim](const Tensor &g) -> void {
    Tensor g1 = g.reshape(kept_dim.shape()).broadcast_to(a->data().shape());
    a->accumulate_grad(g1);
  };

  return Variable::fromOp(kept_dim.squeeze({}), {a}, backward);
}

VarPtr max(VarPtr a, std::vector<int64_t> dims) {
  auto kept_dim = torch::max(a->data(), dims, true);

  auto backward = [a, kept_dim](const Tensor &g) -> void {
    // max should pass grad only to values which are the maximum
    // relu_back: g * (x > 0) <--
    // g * [1 - 1 * (max > x)] <-- this could work?
    auto one = torch::Tensor::ones({1}, kept_dim.dtype(), kept_dim.device());
    auto lt_max =
        torch::sub(one, torch::relu_back(torch::sub(kept_dim, a->data()), one));
    auto g1 = torch::mult(g.reshape(kept_dim.shape()), lt_max);
    a->accumulate_grad(g1);
  };

  return Variable::fromOp(kept_dim.squeeze({}), {a}, backward);
}

VarPtr relu(VarPtr a) {
  auto backward = [a](const Tensor &g) -> void {
    a->accumulate_grad(torch::relu_back(a->data(), g));
  };
  return Variable::fromOp(torch::relu(a->data()), {a}, backward);
}

VarPtr transpose(VarPtr a, int64_t dim1, int64_t dim2) {
  auto backward = [a, dim1, dim2](const Tensor &g) -> void {
    a->accumulate_grad(g.transpose(dim2, dim1));
  };
  return Variable::fromOp(a->data().transpose(dim1, dim2), {a}, backward);
}

VarPtr reshape(VarPtr a, std::vector<int64_t> dims) {
  auto backward = [a](const Tensor &g) -> void {
    a->accumulate_grad(g.reshape(a->data().shape()));
  };
  return Variable::fromOp(a->data().reshape(dims), {a}, backward);
}

VarPtr neg(VarPtr &a) {
  auto backward = [a](const Tensor &g) -> void {
    a->accumulate_grad(torch::neg(g));
  };
  return Variable::fromOp(torch::neg(a->data()), {a}, backward);
}

VarPtr sin(VarPtr &a) {
  auto backward = [a](const Tensor &g) -> void {
    a->accumulate_grad(torch::mult(g, torch::cos(a->data())));
  };
  return Variable::fromOp(torch::sin(a->data()), {a}, backward);
}

VarPtr cos(VarPtr &a) {
  auto backward = [a](const Tensor &g) -> void {
    a->accumulate_grad(torch::mult(g, torch::neg(torch::sin(a->data()))));
  };
  return Variable::fromOp(torch::cos(a->data()), {a}, backward);
}

VarPtr exp(VarPtr &a) {
  auto out = torch::exp(a->data());
  auto backward = [a, out](const Tensor &g) -> void {
    a->accumulate_grad(torch::mult(g, out));
  };

  return Variable::fromOp(out, {a}, backward);
}

VarPtr ln(VarPtr &a) {
  auto backward = [a](const Tensor &g) -> void {
    a->accumulate_grad(torch::div(g, a->data()));
  };

  return Variable::fromOp(torch::ln(a->data()), {a}, backward);
}

} // namespace autograd
} // namespace torch
