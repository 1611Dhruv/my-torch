#include "mytorch/ops/ops.h"

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
  case torch::Device::CUDA:
    throw std::logic_error("add for CUDA not implemented");
  }
}
} // namespace torch
