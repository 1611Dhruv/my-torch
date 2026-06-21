#include "mytorch/ops/ops.h"
#include "mytorch/storage.h"
#include <stdexcept>

namespace torch {

Tensor add(const Tensor &a, const Tensor &b) {
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("add: tensors should have the same dtype, casting not supported yet");
  }

  if (a.shape() != b.shape()) {
    throw std::invalid_argument("add: tensors must have the same shape");
  }

  if (a.device() != b.device()) {
    throw std::invalid_argument("add: tensor device mismatch"); // TODO: implement data transfer instead of throwing
  }

  if (a.device() == CPU)
    return cpu::add(a, b);
  else
    return cuda::add(a, b);
}

} // namespace torch
