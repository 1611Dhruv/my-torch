#include "mytorch/ops.h"
#include "mytorch/tensor.h"
#include <cassert>
#include <cstdint>
#include <driver_types.h>
namespace torch {
namespace cpu {

template <typename T> Tensor cpu_matmul(const Tensor &a, const Tensor &b, Tensor &out) {
  int64_t n1 = a.shape()[0];
  int64_t n2 = a.shape()[1];
  int64_t n3 = b.shape()[1];

  const T *data_a = a.data_ptr<T>(), *data_b = b.data_ptr<T>();
  T *data_out = out.data_ptr<T>();

  for (int64_t i = 0; i < n1; i++) {
    for (int64_t j = 0; j < n3; j++) {
      for (int64_t k = 0; k < n2; k++)
        data_out[i * n3 + j] += data_a[i * n2 + k] * data_b[k * n3 + j];
    }
  }

  return out;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  assert(a.dtype() == b.dtype());
  assert(a.shape().size() == 2 && b.shape().size() == 2);
  assert(a.shape()[1] == b.shape()[0]);

  if (!a.is_contiguous() || !b.is_contiguous()) {
    throw std::invalid_argument("cpu matmul: tensors must be contiguous"); // Make contiguous() cuda aware
  }

  Tensor out = Tensor::zeros({a.shape()[0], b.shape()[1]}, a.dtype(), a.device());

  DISPATCH_OP(a.dtype(), [&] { cpu_matmul<scalar_t>(a, b, out); });
  return out;
}
} // namespace cpu
} // namespace torch
