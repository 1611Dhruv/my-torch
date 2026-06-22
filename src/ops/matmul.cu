#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include <cassert>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace torch {
namespace cuda {

template <typename scalar_t, const int TILE_WIDTH>
__global__ void matmul_kernel(const scalar_t *a, const scalar_t *b, scalar_t *out, int64_t n1, int64_t n2, int64_t n3) {
  int64_t bx = blockIdx.x, by = blockIdx.y;
  int64_t tx = threadIdx.x, ty = threadIdx.y;

  int64_t i = TILE_WIDTH * by + ty;
  int64_t j = TILE_WIDTH * bx + tx;

  __shared__ scalar_t sh_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ scalar_t sh_B[TILE_WIDTH][TILE_WIDTH];

  scalar_t partial_dot = 0;
  int64_t tiles = (n2 + TILE_WIDTH - 1) / TILE_WIDTH;
  for (int64_t k = 0; k < tiles; k++) {
    if ((i < n1) && (k * TILE_WIDTH + tx < n2)) {
      sh_A[ty][tx] = a[i * n2 + k * TILE_WIDTH + tx];
    } else {
      sh_A[ty][tx] = 0;
    }

    if ((j < n3) && (k * TILE_WIDTH + ty < n2)) {
      sh_B[ty][tx] = b[(k * TILE_WIDTH + ty) * n3 + j];
    } else {
      sh_B[ty][tx] = 0;
    }
    __syncthreads();

    for (int l = 0; l < TILE_WIDTH; l++) {
      partial_dot += sh_A[ty][l] * sh_B[l][tx];
    }
    __syncthreads();
  }

  if ((i < n1) && (j < n3))
    out[i * n3 + j] = partial_dot;
}

Tensor matmul(const Tensor &a, const Tensor &b) {
  assert(a.dtype() == b.dtype());
  assert(a.shape().size() == 2 && b.shape().size() == 2);
  assert(a.shape()[1] == b.shape()[0]);

  if (!a.is_contiguous() || !b.is_contiguous()) {
    throw std::invalid_argument("cuda matmul: tensors must be contiguous"); // Make contiguous() cuda aware
  }

  int64_t n1 = a.shape()[0];
  int64_t n2 = a.shape()[1];
  int64_t n3 = b.shape()[1];

  Tensor out({n1, n3}, a.dtype(), a.device());

  constexpr int TILE_WIDTH = 32;
  dim3 grid((n3 + TILE_WIDTH - 1) / TILE_WIDTH, (n1 + TILE_WIDTH - 1) / TILE_WIDTH);
  dim3 block(TILE_WIDTH, TILE_WIDTH);

  DISPATCH_OP(a.dtype(), [&] {
    matmul_kernel<scalar_t, TILE_WIDTH>
        <<<grid, block>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n1, n2, n3);
  });

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); // TODO: Remove in prod

  return out;
}

} // namespace cuda
} // namespace torch
