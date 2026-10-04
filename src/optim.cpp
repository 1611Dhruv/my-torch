#include "mytorch/optim.h"

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
} // namespace torch
