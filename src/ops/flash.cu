#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"

namespace torch {
namespace cuda {

constexpr double INF = 1e11;
constexpr int WARP_SZ = 32;
// TODO: Can support more later? but for now just use float
using scalar_t = float;
template <const int D_HEAD, const bool CAUSAL, const int BM = 32,
          const int BN = 64, const int T = 128>
__global__ void flash(const scalar_t *Q, const scalar_t *K, const scalar_t *V,
                      scalar_t *LSE, scalar_t *O, int64_t N) {
  __shared__ scalar_t Qs[BM][D_HEAD];
  // To avoid bank conflicts ugh bank aint banking
  __shared__ scalar_t Ks[BN][D_HEAD + 1];
  __shared__ scalar_t Vs[BN][D_HEAD];

  int64_t big_off = blockIdx.y * D_HEAD * N;
  Q += big_off;
  K += big_off;
  V += big_off;
  O += big_off;

  LSE += blockIdx.y * N; // LSE is just 1D

  // Normalized our thingy to account for batches
  int64_t q_off = BM * blockIdx.x;
  int tid = threadIdx.x;
  int tt = T;

  auto load = [&](auto &shared, const scalar_t *global, int sz, int off) {
    for (int i = tid; i < sz * D_HEAD; i += tt) {
      // flat idx i -> [sz][DH]
      int lr = i / D_HEAD;
      int lc = i % D_HEAD;

      int gr = off + i / D_HEAD;
      int gc = i % D_HEAD;
      shared[lr][lc] = (gr < N) ? global[gr * D_HEAD + gc] : scalar_t{};
    }
  };

  load(Qs, Q, BM, q_off);

  constexpr int NUM_WARPS = T / WARP_SZ;
  constexpr int RPW = (BM + NUM_WARPS - 1) / NUM_WARPS;
  constexpr int CPL = (BN + WARP_SZ - 1) / WARP_SZ;

  const int wid = tid / WARP_SZ;
  const int lid = tid % WARP_SZ;

  scalar_t mx_run[RPW] = {};
  scalar_t sums_run[RPW] = {};
  scalar_t out[RPW][D_HEAD / WARP_SZ] = {};

  // init mx_run to -infty
  for (int i = 0; i < RPW; i++) {
    mx_run[i] = static_cast<scalar_t>(-INF);
  }
  // For each tile
  for (int kv_off = 0; kv_off < N; kv_off += BN) {
    // Load the full K and full V
    load(Ks, K, BN, kv_off);
    load(Vs, V, BN, kv_off);

    __syncthreads();
    for (int i = 0; i < RPW; i++) {
      scalar_t P[CPL] = {};

      scalar_t mx_tile = static_cast<scalar_t>(-INF);
      scalar_t sums_tile = scalar_t{};
      // For lanes within this Warp
      for (int j = 0; j < CPL; j++) {
        int qs_off = wid * RPW + i;
        int ks_off = lid * CPL + j;
        for (int d = 0; d < D_HEAD; d++) {
          P[j] += Qs[qs_off][d] * Ks[ks_off][d];
        }
        // Finalize q.T@k / sqrt(d)
        P[j] /= sqrtf(D_HEAD); // Update maxes
        mx_tile = (mx_tile < P[j]) ? P[j] : mx_tile;
      }

      // Now that max is well defined
      for (int j = 0; j < CPL; j++) {
        // Perform e^... and update running sm
        // and also mask
        int q_i = q_off + RPW * wid + i;
        int k_j = kv_off + CPL * lid + j;
        bool keep = (k_j < N);
        if constexpr (CAUSAL) {
          keep = keep && (k_j <= q_i);
        }
        P[j] = keep ? P[j] : -INF;
        P[j] = expf(P[j] - mx_tile);
        sums_tile += P[j];
      }

      scalar_t old_mx = mx_tile;
      // Merge across lanes in this warp
      for (int off = 16; off > 0; off >>= 1) {
        scalar_t mx1 = mx_tile;
        scalar_t mx2 = __shfl_xor_sync(0xffffffff, mx1, off);
        scalar_t s1 = sums_tile;
        scalar_t s2 = __shfl_xor_sync(0xffffffff, s1, off);
        mx_tile = fmaxf(mx1, mx2);
        sums_tile = s1 * expf(mx1 - mx_tile) + s2 * expf(mx2 - mx_tile);
      }

      // Merge btw the last tile and this tile
      scalar_t mx_new = fmaxf(mx_run[i], mx_tile);
      scalar_t sum_new = sums_run[i] * expf(mx_run[i] - mx_new) +
                         sums_tile * expf(mx_tile - mx_new);
      // Also update the row of output
      for (int j = 0; j < D_HEAD / WARP_SZ; j++) {
        out[i][j] *= sums_run[i] / sum_new * expf(mx_run[i] - mx_new);
      }

      sums_run[i] = sum_new;
      mx_run[i] = mx_new;

      // Normalize P again after everything
      for (int j = 0; j < CPL; j++) {
        P[j] *= expf(old_mx - mx_new);
      }

      // Update running output
      // Go through this column again
      for (int j = 0; j < BN; j++) {
        // Get the j'th prob in this row
        auto p = __shfl_sync(0xffffffff, P[j % CPL], j / CPL);
        for (int d = 0; d < D_HEAD; d += WARP_SZ) {
          out[i][d / WARP_SZ] += p / sum_new * Vs[j][d + lid];
        }
      }
    }

    // Now what? Now we gotta do stuff... to produce output
    __syncthreads();
  }

  for (int i = 0; i < RPW; i++) {
    for (int d = 0; d < D_HEAD; d += WARP_SZ) {
      int gr = q_off + wid * RPW + i;

      if (gr < N)
        O[gr * D_HEAD + d + lid] = out[i][d / WARP_SZ];
    }
  }

  for (int i = 0; i < RPW; i++) {
    int gr = q_off + wid * RPW + i;
    if (gr < N)
      LSE[gr] = mx_run[i] + log(sums_run[i]);
  }
}

Tensor flash_atten(const Tensor &Q, const Tensor &K, const Tensor &V,
                   Tensor &LSE, Tensor &out, bool causal) {
  auto shape = Q.shape();
  int64_t shape_sz = shape.size();

  int N = shape[shape_sz - 2];
  int d_head = shape[shape_sz - 1];

  if (d_head == 32) {
    constexpr int BM = 128;
    constexpr int BN = 32;
    constexpr int T = 512;

    dim3 ng((N + BM - 1) / BM, shape[0]);
    if (causal) {
      flash<32, true, BM, BN, T>
          <<<ng, T>>>(Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                      V.data_ptr<scalar_t>(), LSE.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(), N);
    } else {
      flash<32, false, BM, BN, T>
          <<<ng, T>>>(Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                      V.data_ptr<scalar_t>(), LSE.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(), N);
    }
  } else if (d_head == 64) {
    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int T = 512;

    dim3 ng((N + BM - 1) / BM, shape[0]);
    if (causal) {
      flash<64, true, BM, BN, T>
          <<<ng, T>>>(Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                      V.data_ptr<scalar_t>(), LSE.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(), N);
    } else {
      flash<64, false, BM, BN, T>
          <<<ng, T>>>(Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                      V.data_ptr<scalar_t>(), LSE.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(), N);
    }
    /* } else if (d_head == 128) { // BUG: 128 aint fitting gang
       constexpr int BM = 32;
       constexpr int BN = 32;
       constexpr int T = 512;

       dim3 ng((N + BM - 1) / BM, shape[0]);
       if (causal) {
         flash<128, true, BM, BN, T>
             <<<ng, T>>>(Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                         V.data_ptr<scalar_t>(), LSE.data_ptr<scalar_t>(),
                         out.data_ptr<scalar_t>(), N);
       } else {
         flash<128, false, BM, BN, T>
             <<<ng, T>>>(Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                         V.data_ptr<scalar_t>(), LSE.data_ptr<scalar_t>(),
                         out.data_ptr<scalar_t>(), N);
       } */
  } else {
    throw std::invalid_argument(
        "cuda flash_atten: Expecting d_head to be 32, 64 or 128(BUG) got: " +
        std::to_string(d_head));
  }

  CUDA_CHECK(cudaGetLastError());
  return out;
}

/*
Calculates D_i = rowsum (dO ⊙ O)
*/
template <const int T, const int BS>
__global__ void helper_di_kernel(const scalar_t *O, const scalar_t *dO,
                                 scalar_t *D, int64_t D_HEAD, int64_t N) {
  constexpr int WPER = BS / (T >> 5);

  const int batch_off = blockIdx.y;
  O += batch_off * N * D_HEAD;
  dO += batch_off * N * D_HEAD;
  D += batch_off * N;

  const int row_off = blockIdx.x * BS;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lid = tid & 31;

  const int woff = wid * WPER;
  scalar_t acc{};
  for (int row = row_off + woff; row < row_off + woff + WPER && row < N;
       row++) {
    acc = 0;
    for (int c = lid; c < D_HEAD; c += 32) {
      acc += O[row * D_HEAD + c] * dO[row * D_HEAD + c];
    }

#pragma unroll
    for (int off = 16; off >= 1; off /= 2) {
      acc += __shfl_xor_sync(0xffffffff, acc, off);
    }

    if (lid == 0)
      D[row] = acc;
  }
}

// Backward
template <const int T, const int D_H, const int BS>
__global__ void flash_back_kernel(const scalar_t *O, const scalar_t *Q,
                                  const scalar_t *K, const scalar_t *V,
                                  const scalar_t *dO, const scalar_t *LSE,
                                  const scalar_t *D, scalar_t *dQ, scalar_t *dK,
                                  scalar_t *dV, int N, bool causal) {

  // BD + BD  + BD + BD + B + B^2 + B^2
  __shared__ scalar_t Qs[BS][D_H];
  __shared__ scalar_t Ks[BS][D_H];
  __shared__ scalar_t Vs[BS][D_H];
  __shared__ scalar_t dOs[BS][D_H];
  __shared__ scalar_t LSEs[BS];
  __shared__ scalar_t Ps[BS][BS];

  // Make some things reused
  auto Ds = LSEs;
  auto dSs = Ps;

  constexpr int dCNT = (BS * D_H + T - 1) / T;
  constexpr scalar_t RSQRT_DH = 1.f / csqrt(D_H);

  scalar_t dKacc[dCNT] = {};
  scalar_t dVacc[dCNT] = {};

  // incorporate batch offsets
  {
    O += (N * D_H) * blockIdx.y;
    Q += (N * D_H) * blockIdx.y;
    K += (N * D_H) * blockIdx.y;
    V += (N * D_H) * blockIdx.y;
    dO += (N * D_H) * blockIdx.y;
    LSE += (N)*blockIdx.y;
    D += (N)*blockIdx.y;
    dQ += (N * D_H) * blockIdx.y;
    dK += (N * D_H) * blockIdx.y;
    dV += (N * D_H) * blockIdx.y;
  }

  auto bid = blockIdx.x;
  auto tid = threadIdx.x;

  // Load store helpers
  auto ldst_2d = [&](int off, auto &shared, auto &global) {
    for (int i = tid; i < BS * D_H; i += T) {
      int lr = i / D_H;
      int lc = i % D_H;

      // if constexpr (transpose) {
      // shared[lc][lr] = (off + lr < N && lc < D_H)
      //                      ? global[(off + lr) * D_H + lc]
      //                     : scalar_t{};
      //} else {
      shared[lr][lc] = (off + lr < N && lc < D_H)
                           ? global[(off + lr) * D_H + lc]
                           : scalar_t{};
      // }
    }
  };

  auto ldst_1d = [&](int off, auto &shared, auto &global) {
    for (int i = tid; i < BS; i += T) {
      shared[i] = (off + i < N) ? global[off + i] : scalar_t{};
    }
  };

  int koff = bid * BS;
  ldst_2d(koff, Ks, K);
  ldst_2d(koff, Vs, V);

  for (int qoff = 0; qoff < N; qoff += BS) {
    ldst_2d(qoff, Qs, Q);
    ldst_1d(qoff, LSEs, LSE);
    __syncthreads();
    // Pre fetch next phases's dO
    ldst_2d(qoff, dOs, dO);

    // Generate P
    for (int i = tid; i < BS * BS; i += T) {
      int lc = (i % BS);
      int lr = (i / BS);

      Ps[lr][lc] = 0;
      for (int k = 0; k < D_H; k++) {
        Ps[lr][lc] += Qs[lr][k] * Ks[lc][k];
      }
      Ps[lr][lc] = __expf(Ps[lr][lc] * RSQRT_DH - LSEs[lr]);
      if (causal && (koff + lc) > (qoff + lr)) {
        Ps[lr][lc] = 0;
      }
    }

    __syncthreads();
    // Pre fetch next Phase's D
    ldst_1d(qoff, Ds, D);

    // Generate dV's shard
    for (int i = 0; i < dCNT; i++) {
      int id = i * T + tid;

      int lc = id % D_H;
      int lr = id / D_H;

      for (int k = 0; k < BS; k++) {
        dVacc[i] += Ps[k][lr] * dOs[k][lc];
      }
    }

    __syncthreads();

    // Generate dS
    for (int i = tid; i < BS * BS; i += T) {
      int lc = i % BS;
      int lr = i / BS;

      scalar_t dP = 0;
      for (int k = 0; k < D_H; k++) {
        dP += dOs[lr][k] * Vs[lc][k];
      }
      dSs[lr][lc] = dP * Ps[lr][lc] - Ps[lr][lc] * Ds[lr];
    }

    __syncthreads();

    // Generate and write back a slice of dQ
    for (int i = tid; i < BS * D_H; i += T) {
      int lr = i / D_H;
      int lc = i % D_H;

      int gr = qoff + lr;
      int gc = lc;

      if (gr < N && gc < D_H) {
        scalar_t dQe = 0;
        for (int k = 0; k < BS; k++) {
          dQe += RSQRT_DH * dSs[lr][k] * Ks[k][lc];
        }
        // Attomically add cuz other blocks also doing this same shard
        atomicAdd(&dQ[gr * D_H + gc], dQe);
      }
    }

    // Generate dK's shard
    for (int i = 0; i < dCNT; i++) {
      int id = i * T + tid;

      int lc = id % D_H;
      int lr = id / D_H;

      for (int k = 0; k < BS; k++) {
        dKacc[i] += RSQRT_DH * dSs[k][lr] * Qs[k][lc];
      }
    }
    __syncthreads();
  }

  // All are done
  for (int i = 0; i < dCNT; i++) {
    int id = i * T + tid;

    int lc = id % D_H;
    int lr = id / D_H;

    int gc = lc;
    int gr = lr + koff;

    if (gr < N && gc < D_H) {
      dK[gr * D_H + gc] = dKacc[i];
      dV[gr * D_H + gc] = dVacc[i];
    }
  }
}

/*

D = rowsum(O ⨀ dO)
dS = dP ⨀ P - P ⨀ {D}
dV = P^dO
dP = dO V^
sqrt(dk) S = QK^
P = exp(S - {LGE})
dQ = sqrt(dk) dS K
dK = sqrt(dk) dS^Q

 */
void flash_back(const Tensor &O, const Tensor &Q, const Tensor &K,
                const Tensor &V, const Tensor &dO, const Tensor &LSE,
                Tensor &dQ, Tensor &dK, Tensor &dV, bool causal) {
  const auto &shape = O.shape();
  const auto &sz = shape.size();

  int64_t D_H = shape[sz - 1];
  int64_t N = shape[sz - 2];
  int64_t B = shape[sz - 3];

  Tensor D = Tensor::zeros_like(LSE);
  // Blk for the launch so launch param names can be reused
  {
    constexpr int T = 256;
    constexpr int BS = 48;

    dim3 blk(T);
    dim3 grid((N + BS - 1) / BS, B);

    helper_di_kernel<T, BS><<<grid, blk>>>(O.data_ptr<scalar_t>(),
                                           dO.data_ptr<scalar_t>(),
                                           D.data_ptr<scalar_t>(), D_H, N);
    CUDA_CHECK(cudaGetLastError());
  }

  if (D_H == 32) {
    constexpr int T = 256;
    constexpr int BS = 48;
    constexpr int CD_H = 32;

    dim3 blk(T);
    dim3 grid((N + BS - 1) / BS, B);
    flash_back_kernel<T, CD_H, BS><<<grid, blk>>>(
        O.data_ptr<scalar_t>(), Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
        V.data_ptr<scalar_t>(), dO.data_ptr<scalar_t>(),
        LSE.data_ptr<scalar_t>(), D.data_ptr<scalar_t>(),
        dQ.data_ptr<scalar_t>(), dK.data_ptr<scalar_t>(),
        dV.data_ptr<scalar_t>(), N, causal);
    CUDA_CHECK(cudaGetLastError());
  } else if (D_H == 64) {
    constexpr int T = 256;
    constexpr int BS = 32;
    constexpr int CD_H = 64;

    dim3 blk(T);
    dim3 grid((N + BS - 1) / BS, B);
    flash_back_kernel<T, CD_H, BS><<<grid, blk>>>(
        O.data_ptr<scalar_t>(), Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
        V.data_ptr<scalar_t>(), dO.data_ptr<scalar_t>(),
        LSE.data_ptr<scalar_t>(), D.data_ptr<scalar_t>(),
        dQ.data_ptr<scalar_t>(), dK.data_ptr<scalar_t>(),
        dV.data_ptr<scalar_t>(), N, causal);
    CUDA_CHECK(cudaGetLastError());
  } else {
    throw std::invalid_argument(
        "cuda flash_atten: Expecting d_head to be 32, 64 or 128(BUG) got: " +
        std::to_string(D_H));
  }
}

} // namespace cuda
} // namespace torch
