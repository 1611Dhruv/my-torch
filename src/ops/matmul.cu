#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include <cassert>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace torch {
namespace cuda {

template <typename scalar_t, const int TILE_WIDTH, const int REG_WIDTH>
__global__ void matmul_kernel(const scalar_t *a, const scalar_t *b, scalar_t *out, int64_t n1, int64_t n2, int64_t n3) {
  int64_t bx = blockIdx.x, by = blockIdx.y;
  int64_t tx = threadIdx.x, ty = threadIdx.y;

  int64_t ri = TILE_WIDTH / REG_WIDTH * by + ty;
  int64_t rj = TILE_WIDTH / REG_WIDTH * bx + tx;

  int64_t r0 = ri * REG_WIDTH;
  int64_t c0 = rj * REG_WIDTH;

  __shared__ scalar_t sh_A[TILE_WIDTH][TILE_WIDTH];
  __shared__ scalar_t sh_B[TILE_WIDTH][TILE_WIDTH];

  scalar_t accumulator[REG_WIDTH][REG_WIDTH];
#pragma unroll
  for (int i = 0; i < REG_WIDTH; i++) {
#pragma unroll
    for (int j = 0; j < REG_WIDTH; j++) {
      accumulator[i][j] = static_cast<scalar_t>(0);
    }
  }

  int tid = ty * blockDim.x + tx;
  int nthreads = blockDim.x * blockDim.y;

  int64_t tiles = (n2 + TILE_WIDTH - 1) / TILE_WIDTH;
  for (int64_t k = 0; k < tiles; k++) {
    for (int idx = tid; idx < TILE_WIDTH * TILE_WIDTH; idx += nthreads) {
      int r = idx / TILE_WIDTH;
      int c = idx % TILE_WIDTH;

      int a_row = by * TILE_WIDTH + r, a_col = k * TILE_WIDTH + c;
      if ((a_row < n1) && (a_col < n2)) {
        sh_A[r][c] = a[a_row * n2 + a_col];
      } else {
        sh_A[r][c] = 0;
      }

      int b_row = k * TILE_WIDTH + r, b_col = bx * TILE_WIDTH + c;
      if ((b_row < n2) && (b_col < n3)) {
        sh_B[r][c] = b[b_row * n3 + b_col];
      } else {
        sh_B[r][c] = 0;
      }
    }
    __syncthreads();

    for (int l = 0; l < TILE_WIDTH; l++) {
      // partial_dot += sh_A[ty][l] * sh_B[l][tx];
      scalar_t a_reg[REG_WIDTH], b_reg[REG_WIDTH];
      for (int i = 0; i < REG_WIDTH; i++)
        a_reg[i] = sh_A[i + ty * REG_WIDTH][l];
      for (int j = 0; j < REG_WIDTH; j++)
        b_reg[j] = sh_B[l][j + tx * REG_WIDTH];

      for (int i = 0; i < REG_WIDTH; i++) {
        for (int j = 0; j < REG_WIDTH; j++) {
          accumulator[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < REG_WIDTH; i++) {
    for (int j = 0; j < REG_WIDTH; j++) {
      if ((r0 + i < n1) && (c0 + j < n3))
        out[(r0 + i) * n3 + (c0 + j)] = accumulator[i][j];
    }
  }
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

  constexpr int TILE_WIDTH = 64;
  constexpr int REG_WIDTH = 4;

  dim3 grid((n3 + TILE_WIDTH - 1) / TILE_WIDTH, (n1 + TILE_WIDTH - 1) / TILE_WIDTH);
  dim3 block(TILE_WIDTH / REG_WIDTH, TILE_WIDTH / REG_WIDTH);

  DISPATCH_OP(a.dtype(), [&] {
    matmul_kernel<scalar_t, TILE_WIDTH, REG_WIDTH>
        <<<grid, block>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n1, n2, n3);
  });

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize()); // TODO: Remove in prod

  return out;
}

} // namespace cuda
} // namespace torch
