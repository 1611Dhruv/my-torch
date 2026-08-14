#ifndef OPTIM_H
#define OPTIM_H

#include "mytorch/nn/module.h"
#include <vector>

namespace torch {
class Optim {
public:
  Optim(std::vector<std::shared_ptr<autograd::Variable>> params)
      : _params(params) {}
  void step();
  void zero_grad();

protected:
  std::vector<std::shared_ptr<autograd::Variable>> _params;
};
} // namespace torch

#endif
