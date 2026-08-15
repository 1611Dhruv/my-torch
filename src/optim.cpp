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
      throw std::logic_error("Please call backword, got param without grad");
    }
    p->data() = sub(p->data(), mult(*p->grad(), _lr));
  }
}
} // namespace torch
