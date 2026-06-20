#include "mytorch/ops/ops.h"
#include "mytorch/tensor.h"

namespace torch {
namespace cpu {

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

  Tensor out = Tensor.like(a); // something like this
  DISPATCH_OP(a.dtype(), [&] { add_kernel<scalar_t>(a, b, out); });
  return a;
}

} // namespace cpu
} // namespace torch
