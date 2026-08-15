#ifndef LOSS_H
#define LOSS_H
#include "mytorch/autograd.h"
#include <numeric>

namespace torch {
class MSE {

public:
  MSE(autograd::VarPtr pred, autograd::VarPtr act);
  const float &loss() const { return _loss->data().data_ptr<float>()[0]; };
  void backward();

private:
  autograd::VarPtr _loss;
};
} // namespace torch

#endif
