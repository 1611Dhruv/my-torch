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

  for (int64_t b = 0; b < B; b++) {
    for (int64_t i = 0; i < M; i++) {
      for (int64_t j = 0; j < N; j++) {
        for (int64_t k = 0; k < K; k++)
          data_out[b * (M * N) + i * N + j] += data_a[b * (M * K) + i * K + k] * data_b[b * (K * N) + k * N + j];
      }
    }
  }

  return out;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  assert(a.dtype() == b.dtype());
  auto a_s = a.shape();
  auto b_s = b.shape();
  if (a_s.size() < 2 || b_s.size() < 2) {
    throw std::invalid_argument("Matmul must be called between a min of 2 by 2 matrices");
  }

  // Extract out a_k and a_m
  int64_t a_k = a_s.back();
  a_s.pop_back();
  int64_t a_m = a_s.back();
  a_s.pop_back();

  int64_t b_n = b_s.back();
  b_s.pop_back();
  int64_t b_k = b_s.back();
  b_s.pop_back();

  if (a_k != b_k) {
    throw std::invalid_argument("The matmul is invalid (shared dim should be the same)");
  }

  if (a_s != b_s) {
    throw std::invalid_argument("The batched dimensions should be the same");
  }

  Tensor a_l = a;
  Tensor b_l = b;
  // Figure out the Batch dimension
  if (!a.is_contiguous()) {
    a_l = a.contiguous();
  }

  if (!b.is_contiguous()) {
    b_l = b.contiguous();
  }

  auto batch = b_s;
  int64_t B = 1;
  for (int64_t i = 0; i < batch.size(); i++) {
    B *= batch[i];
  }

  a_l = a_l.reshape({B, a_m, a_k});
  b_l = b_l.reshape({B, b_k, b_n});

  Tensor out = Tensor::zeros({B, a_m, b_n}, a.dtype(), a.device());

  DISPATCH_OP(a.dtype(), [&] { cpu_matmul<scalar_t>(a_l, b_l, out, B, a_m, a_k, b_n); });
  batch.push_back(a_m);
  batch.push_back(b_n);
  return out.reshape(batch);
}
} // namespace cpu
} // namespace torch
