#include "mytorch/cuda_utils.h"
#include "mytorch/ops/ops.h"
#include <cassert>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace torch {
namespace cuda {

template <typename scalar_t>
__global__ void add_kernel(const scalar_t *a, const scalar_t *b, scalar_t *out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  out[i] = a[i] + b[i];
}

Tensor add(const Tensor &a, const Tensor &b) {
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
    add_kernel<scalar_t>
        <<<blocks, threads>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
  });

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); // TODO: Remove in prod

  return out;
}

} // namespace cuda
} // namespace torch
