#include "mytorch/ops.h"
#include "mytorch/tensor.h"
#include <cassert>
#include <cstdint>
namespace torch {
namespace cpu {

template <typename T>
Tensor cpu_matmul(const Tensor &a, const Tensor &b, Tensor &out, int64_t B, int64_t M, int64_t K, int64_t N) {
  const T *data_a = a.data_ptr<T>(), *data_b = b.data_ptr<T>();
  T *data_out = out.data_ptr<T>();

  for (int64_t btch = 0; btch < B; btch++) {
    for (int64_t i = 0; i < M; i++) {
      for (int64_t j = 0; j < N; j++) {
        for (int64_t k = 0; k < K; k++)
          data_out[btch * out.strides()[0] + i * out.strides()[1] + j * out.strides()[2]] +=
              data_a[btch * a.strides()[0] + i * a.strides()[1] + k * a.strides()[2]] *
              data_b[btch * b.strides()[0] + k * b.strides()[1] + j * b.strides()[2]];
      }
    }
  }

  return out;
}

Tensor matmul(const Tensor &a, const Tensor &b, int64_t B, int64_t M, int64_t K, int64_t N) {
  assert(a.dtype() == b.dtype());

  // Reshape garuantees contiguity
  Tensor T_A = a.reshape({B, M, K});
  Tensor T_B = b.reshape({B, K, N});

  Tensor out = Tensor::zeros({B, M, N}, a.dtype(), CPU);

  DISPATCH_OP(a.dtype(), [&] { cpu_matmul<scalar_t>(T_A, T_B, out, B, M, K, N); });
  return out;
}
} // namespace cpu
} // namespace torch
