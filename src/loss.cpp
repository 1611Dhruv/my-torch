#include "mytorch/loss.h"

namespace ag = torch::autograd;

namespace torch {
MSE::MSE(ag::VarPtr pred, ag::VarPtr act) {
  ag::VarPtr diff = sub(act, pred);
  ag::VarPtr sq = mult(diff, diff);
  std::vector<int64_t> reduce_along(act->data().ndim());
  std::iota(reduce_along.begin(), reduce_along.end(), 0);
  for (auto &e : reduce_along) {
    e -= sq->data().ndim();
  }
  ag::VarPtr total = sum(sq, reduce_along);
  float ndim = act->data().numel();
  _loss = scale(total, 1 / ndim);
}

void MSE::backward() {
  _loss->backward();
}

CrossEntropy::CrossEntropy(ag::VarPtr pred_nats, ag::VarPtr act_prob) {
  auto max_nat = ag::max(pred_nats, {-1}, true);
  ag::VarPtr lgsmexp = ag::add(
      max_nat,
      ag::ln(ag::sum(ag::exp(ag::sub(pred_nats, max_nat)), {-1}, true)));
  auto total =
      ag::sub(lgsmexp, ag::sum(ag::mult(pred_nats, act_prob), {-1}, true));
  float ndim = total->data().numel();
  _loss = ag::scale(ag::sum(total), 1 / ndim);
}

void CrossEntropy::backward() {
  _loss->backward();
}

} // namespace torch
