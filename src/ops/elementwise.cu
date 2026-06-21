#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include <cassert>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace torch {
namespace cuda {

template <typename scalar_t, typename Op>
__global__ void binary_kernel(const scalar_t *a, const scalar_t *b, scalar_t *out, int64_t n, Op op) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  out[i] = op(a[i], b[i]);
}

template <typename Op> Tensor elementwise_wrapper(const Tensor &a, const Tensor &b, Op op) {
  assert(a.dtype() == b.dtype());
  assert(a.shape() == b.shape());

  if (!a.is_contiguous() || !b.is_contiguous()) {
    throw std::invalid_argument("cuda add: tensors must be contiguous"); // Make contiguous() cuda aware
  }

  Tensor out(a.shape(), a.dtype(), a.device());

  int64_t n = a.numel();
  int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;

  DISPATCH_OP(a.dtype(), [&] {
    binary_kernel<scalar_t>
        <<<blocks, threads>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n, op);
  });

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); // TODO: Remove in prod

  return out;
}

Tensor add(const Tensor &a, const Tensor &b) {
  return elementwise_wrapper(a, b, [] __device__(auto x, auto y) { return x + y; });
}

Tensor sub(const Tensor &a, const Tensor &b) {
  return elementwise_wrapper(a, b, [] __device__(auto x, auto y) { return x - y; });
}

Tensor mult(const Tensor &a, const Tensor &b) {
  return elementwise_wrapper(a, b, [] __device__(auto x, auto y) { return x * y; });
}

} // namespace cuda
} // namespace torch
