#ifndef LOSS_H
#define LOSS_H
#include "cuda_utils.h"
#include "mytorch/autograd.h"
#include <numeric>

namespace torch {
class MSE {

public:
  MSE(autograd::VarPtr pred, autograd::VarPtr act);
  const float loss() const {
    if (_loss->data().device() == CUDA) {
      float loss = 0;
      CUDA_CHECK(cudaMemcpy(&loss, _loss->data().data_ptr<float>(),
                            sizeof(float), cudaMemcpyDeviceToHost));
      return loss;
    }
    return _loss->data().data_ptr<float>()[0];
  };
  void backward();

private:
  autograd::VarPtr _loss;
};

class CrossEntropy {

public:
  CrossEntropy(autograd::VarPtr pred, autograd::VarPtr act);
  const float loss() const {
    if (_loss->data().device() == CUDA) {
      float loss = 0;
      CUDA_CHECK(cudaMemcpy(&loss, _loss->data().data_ptr<float>(),
                            sizeof(float), cudaMemcpyDeviceToHost));
      return loss;
    }
    return _loss->data().data_ptr<float>()[0];
  };
  void backward();

private:
  autograd::VarPtr _loss;
};

} // namespace torch

#endif
