#include "mytorch/ops/ops.h"

namespace torch {
namespace cpu {

template <typename T> static void add_kernel(const Tensor &a, const Tensor &b, Tensor &out) {
  T *a_data = a.data_ptr<T>();
}

// Add operation for a basic CPU based tensor math
// We will assume that we are working with scalar_t type
// We dont work with a and b having different tensor types (for now ig? )
Tensor add(const Tensor &a, const Tensor &b) {
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("add: tensors should have the same dtype, casting not supported yet");
  }

  if (a.shape() != b.shape()) {
    throw std::invalid_argument("add: tensors must have the same shape");
  }

  Tensor out = torch::Tensor::zeros(a.shape(), a.dtype(), a.device());
  DISPATCH_OP(a.dtype(), [&] { add_kernel<scalar_t>(a, b, out); });
  return out;
}

} // namespace cpu
} // namespace torch
