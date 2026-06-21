#include "mytorch/ops.h"

namespace torch {

// Add Dispatcher (Move to macro later)
Tensor add(const Tensor &a, const Tensor &b) {
  // NOTE: Support auto type promotion later
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("Both tensors should have same datatype");
  }
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Both tensors should have same shape");
  }

  if (a.device() != b.device()) {
    throw std::invalid_argument("Both tensors should be on the same device");
  }

  switch (a.device()) {
  case torch::Device::CPU:
    return torch::cpu::add(a, b);
  default:
    throw std::logic_error("Unsupported device");
  }
}

Tensor sub(const Tensor &a, const Tensor &b) {
  // NOTE: Support auto type promotion later
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("Both tensors should have same datatype");
  }
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Both tensors should have same shape");
  }

  if (a.device() != b.device()) {
    throw std::invalid_argument("Both tensors should be on the same device");
  }

  switch (a.device()) {
  case torch::Device::CPU:
    return torch::cpu::sub(a, b);
  default:
    throw std::logic_error("Unsupported device");
  }
}

Tensor mult(const Tensor &a, const Tensor &b) {
  // NOTE: Support auto type promotion later
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("Both tensors should have same datatype");
  }
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Both tensors should have same shape");
  }

  if (a.device() != b.device()) {
    throw std::invalid_argument("Both tensors should be on the same device");
  }

  switch (a.device()) {
  case torch::Device::CPU:
    return torch::cpu::mult(a, b);
  default:
    throw std::logic_error("Unsupported device");
  }
}

// Add Dispatcher (Move to macro later)
Tensor neg(const Tensor &a) {
  switch (a.device()) {
  case torch::Device::CPU:
    return torch::cpu::neg(a);
  default:
    throw std::logic_error("Unsupported device");
  }
}

} // namespace torch
