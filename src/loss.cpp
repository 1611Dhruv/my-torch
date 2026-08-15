#include "mytorch/loss.h"

namespace torch {

MSE::MSE(autograd::VarPtr pred, autograd::VarPtr act) {
  using namespace autograd;
  VarPtr diff = sub(act, pred);
  VarPtr sq = mult(diff, diff);
  std::vector<int64_t> reduce_along(act->data().ndim());
  std::iota(reduce_along.begin(), reduce_along.end(), 0);
  for (auto &e : reduce_along) {
    e -= sq->data().ndim();
  }
  VarPtr total = sum(sq, reduce_along);
  float ndim = act->data().numel();
  VarPtr norm = Variable::leaf(Tensor({1}, reinterpret_cast<std::byte *>(&ndim)), false);

  _loss = div(total, norm);
}

void MSE::backward() { _loss->backward(); }

} // namespace torch
