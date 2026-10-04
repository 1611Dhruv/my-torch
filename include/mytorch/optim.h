#ifndef OPTIM_H
#define OPTIM_H

#include "mytorch/autograd.h"
#include <vector>

namespace torch {
namespace ag = autograd;
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
  SGD(std::vector<std::shared_ptr<autograd::Variable>> params, double lr)
      : Optim(params),
        _lr(lr) {
    if (params.empty()) {
      throw std::invalid_argument("Parameters cannot be empty");
    }
  }

  void step() override;

private:
  double _lr;
};

class Adam : public Optim {
public:
  Adam(std::vector<ag::VarPtr> params, double lr, double b1 = 0.9,
       double b2 = 0.999)
      : Optim(params),
        _lr(lr),
        _b1(b1),
        _b2(b2),
        _b1_running(1),
        _b2_running(1) {
    if (params.empty()) {
      throw std::invalid_argument("Parameters cannot be empty");
    }
    for (auto &p : params) {
      _m.emplace(p, Tensor::zeros_like(p->data()));
      _v.emplace(p, Tensor::zeros_like(p->data()));
    }
  }

  void step() override;

private:
  double _lr;
  double _b1;
  double _b2;
  double _b1_running;
  double _b2_running;
  std::unordered_map<ag::VarPtr, Tensor> _m;
  std::unordered_map<ag::VarPtr, Tensor> _v;
};
} // namespace torch

#endif
