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

// Backward

Tensor flash_back(const Tensor &Q, const Tensor &K, const Tensor &V,
                  const Tensor &dO, const Tensor &LGE, Tensor &dQ, Tensor &dK,
                  Tensor &dV) {}

} // namespace cuda
} // namespace torch
