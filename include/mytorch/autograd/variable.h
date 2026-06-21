#ifndef VARIABLE_H
#define VARIABLE_H
#include "mytorch/tensor.h"
#include <memory>

namespace torch {
namespace autograd {
class Function;

class Variable {
public:
  static std::shared_ptr<Variable> leaf(Tensor &tensor, bool requires_grad = true) {
    return std::make_shared<Variable>(tensor, requires_grad);
  }

private:
  Variable(Tensor &tensor, bool requires_grad)
      : _t(tensor),
        _requires_grad(requires_grad),
        _grad(Tensor::zeros_like(tensor)) {}
  Tensor _t;
  Tensor _grad;
  bool _requires_grad;
  std::shared_ptr<Function> grad_fn;
};

} // namespace autograd
} // namespace torch

#endif
