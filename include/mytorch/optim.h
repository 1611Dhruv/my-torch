#ifndef OPTIM_H
#define OPTIM_H

#include "mytorch/autograd.h"
#include <vector>

namespace torch {
class Optim {
public:
  Optim(std::vector<std::shared_ptr<autograd::Variable>> params)
      : _params(params) {}
  ~Optim() = default;

  virtual void step() = 0;
  void zero_grad();

protected:
  std::vector<std::shared_ptr<autograd::Variable>> _params;
};

class SGD : public Optim {
public:
  SGD(std::vector<std::shared_ptr<autograd::Variable>> params, float lr)
      : Optim(params),
        _lr({1}, reinterpret_cast<std::byte *>(&lr)) {
    if (params.empty()) {
      throw std::invalid_argument("Parameters cannot be empty");
    }
  }

  void step() override;

private:
  Tensor _lr;
};
} // namespace torch

#endif
