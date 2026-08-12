#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include <cassert>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string>

namespace torch {
namespace cuda {

/*
 * Double buffered, warp tiled, reg tiled Matmul
 *
 * Reduces a M * K and K * N matrix down to M * N
 * Accounts for original ones being transposed or not
 */

template <typename scalar_t, const bool a_transp, const bool b_transp, const int BM = 128, const int BN = 128,
          const int BK = 4, const int WM = 64, const int WN = 64, const int TM = 8, const int TN = 4,
          const int WN_ITER = 2>
__global__ void matmul_kernel(const scalar_t *__restrict__ A, const scalar_t *__restrict__ B, scalar_t *__restrict__ C,
                              int64_t M, int64_t K, int64_t N) {
  // +1 padding for non coalasced things
  __shared__ scalar_t As[2][BK][BM + 1];
  __shared__ scalar_t Bs[2][BK][BN + 1];

  // Some asserts for this reg + warp tiled + double buffered kernel
  constexpr int WM_ITER = WM * WN / (32 * TM * TN) / WN_ITER;
  constexpr int LANE_ROWS = WM / WM_ITER / TM;
  constexpr int LANE_COLS = WN / WN_ITER / TN;
  constexpr int WM_STMP = LANE_ROWS * TM;
  constexpr int WN_STMP = LANE_COLS * TN;

  static_assert(WM_ITER * WN_ITER * TM * TN * 32 == WM * WN, "The warp tiled configuration do not match");
  static_assert(LANE_ROWS * LANE_COLS == 32, "The warp must have only 32 lanes");

  int64_t A_batch_off = blockIdx.z * M * K;
  int64_t B_batch_off = blockIdx.z * K * N;
  int64_t C_batch_off = blockIdx.z * M * N;

  int roff = blockIdx.y * BM;
  int coff = blockIdx.x * BN;
  int tid = threadIdx.x;
  int wid = tid >> 5;
  int wroff = (wid / (BN / WN)) * WM;
  int wcoff = (wid % (BN / WN)) * WN;
  int lane = tid & 31;

  int tt = (BM * BN / WM / WN * 32);

  int pw = 0;
  int pr = 1;

  auto load_a = [&](int koff) {
    if constexpr (a_transp) {
      for (int i = tid; i < BK * BM; i += tt) {
        int lr = i / BM;
        int lc = i % BM;

        int gr = koff + lr;
        int gc = roff + lc;
        As[pw][lr][lc] = (gr < K && gc < M) ? A[A_batch_off + gr * M + gc] : scalar_t{};
      }
    } else {
      for (int i = tid; i < BK * BM; i += tt) {
        int lr = i / BK;
        int lc = i % BK;

        int gr = roff + lr;
        int gc = koff + lc;
        As[pw][lc][lr] = (gr < M && gc < K) ? A[A_batch_off + gr * K + gc] : scalar_t{};
      }
    }
  };

  auto load_b = [&](int koff) {
    if constexpr (b_transp) {
      for (int i = tid; i < BK * BN; i += tt) {
        int lr = i / BK;
        int lc = i % BK;

        int gr = coff + lr;
        int gc = koff + lc;
        Bs[pw][lc][lr] = (gr < N && gc < K) ? B[B_batch_off + gr * K + gc] : scalar_t{};
      }
    } else {
      for (int i = tid; i < BK * BN; i += tt) {
        int lr = i / BN;
        int lc = i % BN;

        int gr = koff + lr;
        int gc = coff + lc;
        Bs[pw][lr][lc] = (gr < K && gc < N) ? B[B_batch_off + gr * N + gc] : scalar_t{};
      }
    }
  };

  // Thread registers
  scalar_t a_reg[WM_ITER][TM];
  scalar_t b_reg[WN_ITER][TN];
  scalar_t acc[WM_ITER][WN_ITER][TM][TN] = {};

  auto load_a_reg = [&](int k, int wi) {
    int warp_off = wi * WM_STMP;
    for (int i = 0; i < TM; i++)
      a_reg[wi][i] = As[pr][k][wroff + warp_off + (lane / LANE_COLS) * TM + i];
  };

  auto load_b_reg = [&](int k, int wj) {
    int warp_off = wj * WN_STMP;
    for (int j = 0; j < TN; j++)
      b_reg[wj][j] = Bs[pr][k][wcoff + warp_off + (lane % LANE_COLS) * TN + j];
  };

  // Actual Matmul
  load_a(0);
  load_b(0);
  __syncthreads();
  for (int koff = 0; koff < K; koff += BK) {
    pw ^= 1;
    pr ^= 1;
    if (koff + BK < K) {
      load_a(koff + BK);
      load_b(koff + BK);
    }

    for (int k = 0; k < BK; k++) {
      // Load the warps into regs
      for (int wi = 0; wi < WM_ITER; wi++) {
        load_a_reg(k, wi);
      }

      for (int wj = 0; wj < WN_ITER; wj++) {
        load_b_reg(k, wj);
      }

      // Perform the FMA
      for (int wi = 0; wi < WM_ITER; wi++) {
        for (int wj = 0; wj < WN_ITER; wj++) {
          for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j++) {
              acc[wi][wj][i][j] += a_reg[wi][i] * b_reg[wj][j];
            }
          }
        }
      }
    }

    __syncthreads();
  }

  for (int wi = 0; wi < WM_ITER; wi++) {
    for (int wj = 0; wj < WN_ITER; wj++) {
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
          int r = roff + wroff + wi * WM_STMP + (lane / LANE_COLS) * TM + i;
          int c = coff + wcoff + wj * WN_STMP + (lane % LANE_COLS) * TN + j;

          if (r < M && c < N) {
            C[C_batch_off + r * N + c] = acc[wi][wj][i][j];
          }
        }
      }
    }
  }
}

static std::pair<bool, bool> verify_contiguity(const Tensor &a, const Tensor &b) {
  int AS = a.shape().size();
  int BS = b.shape().size();

  bool a_contig = a.is_contiguous();
  bool b_contig = b.is_contiguous();
  if (a_contig && b_contig) {
    return {false, false};
  }

  bool a_transp = (a.strides()[AS - 2] == 1 && a.strides()[AS - 1] == a.shape()[AS - 2]);
  bool b_transp = (b.strides()[BS - 2] == 1 && b.strides()[BS - 1] == b.shape()[BS - 2]);
  if ((!a_contig && !a_transp) || (!b_contig && !b_transp)) {
    throw std::invalid_argument("matmul cuda: Passed in tensors are non "
                                "contiguous and cant be treated as transposes");
  }

  return {a_transp, b_transp};
}

Tensor matmul(const Tensor &a, const Tensor &b, int64_t B, int64_t M, int64_t K, int64_t N) {
  assert(a.dtype() == b.dtype());
  assert(a.device() == Device::CUDA);

  Tensor out({B, M, N}, a.dtype(), a.device());

  dim3 grid((N + 127) / 128, (M + 127) / 128, B);
  dim3 block(128);

  // Check for contiguousness
  auto [at, bt] = verify_contiguity(a, b);

  DISPATCH_OP(a.dtype(), [&] {
    if (at && bt) {
      matmul_kernel<scalar_t, true, true>
          <<<grid, block>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), M, K, N);
    } else if (at && !bt) {
      matmul_kernel<scalar_t, true, false>
          <<<grid, block>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), M, K, N);
    } else if (!at && bt) {
      matmul_kernel<scalar_t, false, true>
          <<<grid, block>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), M, K, N);
    } else {
      matmul_kernel<scalar_t, false, false>
          <<<grid, block>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), M, K, N);
    }
  });

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  return out;
}

} // namespace cuda
} // namespace torch
