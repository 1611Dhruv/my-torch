#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace torch {
namespace cuda {

template <typename scalar_t>
__global__ void binary_kernel(const scalar_t *a, const scalar_t *b, scalar_t *out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  // out[i] = op(a[i], b[i]);
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  Tensor out({1, 2});
  return out;
}

} // namespace cuda
} // namespace torch
