#include "mytorch/optim.h"
#include <cmath>

namespace torch {

// Optim
void Optim::zero_grad() {
  for (auto &p : _params) {
    p->zero_grad();
  }
}

// SGD
void SGD::step() {
  for (auto &p : _params) {
    if (!p->grad().has_value()) {
      throw std::logic_error("Please call backward, got param without grad");
    }
    p->data() = sub(p->data(), scale(*p->grad(), _lr));
  }
}

// Adam
void Adam::step() {
  _b1_running *= _b1;
  _b2_running *= _b2;
  _step++;
  _lr = lr();
  for (auto &p : _params) {
    if (!p->grad().has_value()) {
      throw std::logic_error("Please call backward, got param without grad");
    }
    _m.at(p) = add(scale(_m.at(p), _b1), scale(*p->grad(), (1 - _b1)));
    _v.at(p) = add(scale(_v.at(p), _b2),
                   scale(mult(*p->grad(), *p->grad()), (1 - _b2)));
    p->data() = sub(
        p->data(),
        scale(div(scale(_m.at(p), 1 / (1 - _b1_running)),
                  shift(sqrt(scale(_v.at(p), 1 / (1 - _b2_running))), 1e-7)),
              _lr));
  }
}

double Adam::lr() {
  if (_step >= _max_step)
    return _lr_min;
  if (_step <= _warm_up)
    return _lr_max * _step / _warm_up;
  return _lr_min +
         0.5 * (_lr_max - _lr_min) *
             (1 + std::cos((_step - _warm_up) / (_max_step - _warm_up) * M_PI));
}

float Adam::all_grad_norm() {
  Tensor norm_g = Tensor::zeros({1}, DType::Float32, Device::CPU);
  for (auto &p : _params) {
    if (p->grad().has_value()) {
      Tensor g_sqr = torch::sum(torch::mult(*p->grad(), *p->grad()), {})
                         .to(DType::Float32, Device::CPU);
      norm_g = torch::add(norm_g, g_sqr);
    }
  }
  norm_g = torch::sqrt(norm_g);
  return norm_g.data_ptr<float>()[0];
}

void Adam::clip_grad_norm(double to_clip) {
  float norm_g = all_grad_norm();
  if (norm_g > to_clip) {
    for (auto &p : _params) {
      if (p->grad().has_value()) {
        p->grad() = torch::scale(*p->grad(), to_clip / norm_g);
      }
    }
  }
}
} // namespace torch
