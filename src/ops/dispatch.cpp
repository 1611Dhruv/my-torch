#include "mytorch/ops.h"
#include "mytorch/storage.h"
#include <stdexcept>

namespace torch {

template <typename CpuFn, typename CudaFn>
Tensor elementwise_dispatch(const Tensor &a, const Tensor &b, CpuFn cpu_op, CudaFn cuda_op) {
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
    return cpu_op(a, b);
  else
    return cuda_op(a, b);
}

Tensor add(const Tensor &a, const Tensor &b) { return elementwise_dispatch(a, b, cpu::add, cuda::add); }

Tensor sub(const Tensor &a, const Tensor &b) { return elementwise_dispatch(a, b, cpu::sub, cuda::sub); }

Tensor mult(const Tensor &a, const Tensor &b) { return elementwise_dispatch(a, b, cpu::mult, cuda::mult); }

// neg is unary and CPU-only for now (no cuda::neg yet).
Tensor neg(const Tensor &a) {
  switch (a.device()) {
  case torch::Device::CPU:
    return torch::cpu::neg(a);
  default:
    throw std::logic_error("Unsupported device");
  }
}

} // namespace torch
