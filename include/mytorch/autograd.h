#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "mytorch/ops.h"
#include "mytorch/tensor.h"
#include <functional>
#include <memory>
#include <optional>

namespace torch {
namespace autograd {

class Variable {
public:
  static std::shared_ptr<Variable> leaf(const Tensor &t, bool requires_grad = true) {
    return std::shared_ptr<Variable>(new Variable(t, requires_grad));
  }

  static std::shared_ptr<Variable> fromOp(const Tensor &data, std::vector<std::shared_ptr<Variable>> inputs,
                                          std::function<void(const Tensor &grad)> backward) {
    return std::shared_ptr<Variable>(new Variable(data, true, inputs, backward));
  }

  const Tensor &data() const { return _t; }

  const std::optional<Tensor> &grad() const { return _grad; }
  bool has_grad() const { return _grad.has_value(); }
  void accumulate_grad(const Tensor &g) {
    if (!_grad) {
      _grad = g;
    } else {
      _grad = torch::add(*_grad, g);
    }
  }

  void backward();

  void zero_grad() { _grad = std::nullopt; }

private:
  Variable(const Tensor &tensor, bool requires_grad)
      : _t(tensor),
        _requires_grad(requires_grad) {}

  Variable(const Tensor &tensor, bool requires_grad, std::vector<std::shared_ptr<Variable>> inputs,
           std::function<void(const Tensor &grad)> backward)
      : _t(tensor),
        _requires_grad(requires_grad),
        _inputs(inputs),
        _backward(backward) {}

  Tensor _t;
  std::optional<Tensor> _grad;
  bool _requires_grad;
  std::vector<std::shared_ptr<Variable>> _inputs;
  std::function<void(const Tensor &grad)> _backward;
};

// Functions
std::shared_ptr<Variable> add(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
std::shared_ptr<Variable> sub(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
std::shared_ptr<Variable> mult(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);

/*
 * NOTE: Future
std::shared_ptr<Variable> matmul(std::shared_ptr<Variable> a, std::shared_ptr<Variable> b);
*/

} // namespace autograd
} // namespace torch

#endif
