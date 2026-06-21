#ifndef FUNCTION_H
#define FUNCTION_H

#include "mytorch/tensor.h"
#include <memory>

namespace torch {
namespace autograd {

// comes from variable
class Variable;

class Function {
public:
  virtual void backprop(Tensor &out_grad) = 0;

private:
  std::shared_ptr<std::vector<Variable>> _inputs;
};

} // namespace autograd

} // namespace torch
#endif
