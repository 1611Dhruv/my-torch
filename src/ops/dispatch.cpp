#include "mytorch/ops.h"
#include "mytorch/storage.h"
#include <stdexcept>

namespace torch {

// binary ops
template <typename CpuFn, typename CudaFn>
Tensor elementwise_binary_dispatch(const Tensor &a, const Tensor &b, CpuFn cpu_op, CudaFn cuda_op) {
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("elementwise dispatch: tensors should have the same dtype, casting not supported yet");
  }

  if (a.shape() != b.shape()) {
    throw std::invalid_argument("elementwise dispatch: tensors must have the same shape");
  }

  if (a.device() != b.device()) {
    throw std::invalid_argument(
        "elementwise dispatch: tensor device mismatch"); // TODO: implement data transfer instead of throwing
  }

  if (a.device() == CUDA)
    return cuda_op(a, b);
  else
    return cpu_op(a, b);
}

Tensor add(const Tensor &a, const Tensor &b) { return elementwise_binary_dispatch(a, b, cpu::add, cuda::add); }
Tensor sub(const Tensor &a, const Tensor &b) { return elementwise_binary_dispatch(a, b, cpu::sub, cuda::sub); }
Tensor mult(const Tensor &a, const Tensor &b) { return elementwise_binary_dispatch(a, b, cpu::mult, cuda::mult); }

// unary ops
template <typename CpuFn, typename CudaFn>
Tensor elementwise_unary_dispatch(const Tensor &a, CpuFn cpu_op, CudaFn cuda_op) {
  if (a.device() == CUDA)
    return cuda_op(a);
  else
    return cpu_op(a);
}

Tensor neg(const Tensor &a) { return elementwise_unary_dispatch(a, cpu::neg, cuda::neg); }
Tensor sin(const Tensor &a) { return elementwise_unary_dispatch(a, cpu::sin, cuda::sin); }
Tensor cos(const Tensor &a) { return elementwise_unary_dispatch(a, cpu::cos, cuda::cos); }
Tensor exp(const Tensor &a) { return elementwise_unary_dispatch(a, cpu::exp, cuda::exp); }

// matmul
Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("matmul dispatch: tensors should have the same dtype, casting not supported yet");
  }

  if (a.shape().size() != 2 || b.shape().size() != 2) {
    throw std::invalid_argument("matmul dispatch: invalid tensor shape");
  }

  if (a.shape()[1] != b.shape()[0]) {
    throw std::invalid_argument("matmul dispatch: invalid tensor shape");
  }

  if (a.device() != b.device()) {
    throw std::invalid_argument(
        "matmul dispatch: tensor device mismatch"); // TODO: implement data transfer instead of throwing
  }

  if (a.device() == CUDA)
    return cuda::matmul(a, b);
  else
    return cpu::matmul(a, b);
}

} // namespace torch
