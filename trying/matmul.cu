#include "matmul.h"

float *A_c;
float *B_c;
float *OUT_c;

#define TILE 16
#define REG 8
#define BS (TILE * REG)
#define BK 8

#define WM 16
#define WN 128
#define WNITER 2
#define TM 8
#define TN 4
#define NUM_THREADS 256
#define WMITER ((WM * WN) / (32 * TM * TN * WNITER)) // = 1
#define WSUBM (WM / WMITER)                          // = 16
#define WSUBN (WN / WNITER)                          // = 64

#define NUM_A_F4 ((BS * BK / 4) / NUM_THREADS) // float4 staged per thread for A = 1
#define NUM_B_F4 ((BK * BS / 4) / NUM_THREADS) // for B = 1

__device__ __forceinline__ void prefetch_slab(float *A, float *B, int n, int k, int m, int blockRow, int blockCol,
                                              int p, float4 *rA, float4 *rB) {
  float4 zeros{0, 0, 0, 0};
  for (int idx = 0; idx < NUM_A_F4; idx++) {
    int i = threadIdx.x + idx * NUM_THREADS;
    int s_row = i / (BK / 4);
    int s_col = (i % (BK / 4)) * 4;
    int g_row = blockRow + s_row;
    int g_col = p * BK + s_col;
    float4 t;
    if (g_row >= n || g_col >= k)
      t = zeros;
    else if (k - g_col >= 4)
      t = *reinterpret_cast<float4 *>(&A[g_row * k + g_col]);
    else {
      t.x = (g_col + 0 < k) ? A[g_row * k + g_col + 0] : 0.f;
      t.y = (g_col + 1 < k) ? A[g_row * k + g_col + 1] : 0.f;
      t.z = (g_col + 2 < k) ? A[g_row * k + g_col + 2] : 0.f;
      t.w = (g_col + 3 < k) ? A[g_row * k + g_col + 3] : 0.f;
    }
    rA[idx] = t;
  }
  for (int idx = 0; idx < NUM_B_F4; idx++) {
    int i = threadIdx.x + idx * NUM_THREADS;
    int s_row = i / (BS / 4);
    int s_col = (i % (BS / 4)) * 4;
    int g_row = p * BK + s_row;
    int g_col = blockCol + s_col;
    float4 t;
    if (g_row >= k || g_col >= m)
      t = zeros;
    else if (m - g_col >= 4)
      t = *reinterpret_cast<float4 *>(&B[g_row * m + g_col]);
    else {
      t.x = (g_col + 0 < m) ? B[g_row * m + g_col + 0] : 0.f;
      t.y = (g_col + 1 < m) ? B[g_row * m + g_col + 1] : 0.f;
      t.z = (g_col + 2 < m) ? B[g_row * m + g_col + 2] : 0.f;
      t.w = (g_col + 3 < m) ? B[g_row * m + g_col + 3] : 0.f;
    }
    rB[idx] = t;
  }
}

__device__ __forceinline__ void commit_slab(float4 *rA, float4 *rB, float As[BK][BS], float Bs[BK][BS]) {
  for (int idx = 0; idx < NUM_A_F4; idx++) {
    int i = threadIdx.x + idx * NUM_THREADS;
    int s_row = i / (BK / 4);
    int s_col = (i % (BK / 4)) * 4;
    As[s_col + 0][s_row] = rA[idx].x; // transpose into smem, same as before
    As[s_col + 1][s_row] = rA[idx].y;
    As[s_col + 2][s_row] = rA[idx].z;
    As[s_col + 3][s_row] = rA[idx].w;
  }
  for (int idx = 0; idx < NUM_B_F4; idx++) {
    int i = threadIdx.x + idx * NUM_THREADS;
    int s_row = i / (BS / 4);
    int s_col = (i % (BS / 4)) * 4;
    float4 *Bs4 = reinterpret_cast<float4 *>(&Bs[s_row][s_col]);
    *Bs4 = rB[idx];
  }
}

__global__ void kernel_warptiled_doublebuf(float *A, float *B, float *OUT, int n, int k, int m) {
  __shared__ float As[2][BK][BS]; // two buffers, ping-pong
  __shared__ float Bs[2][BK][BS];

  int blockRow = blockIdx.y * BS;
  int blockCol = blockIdx.x * BS;

  int warpIdx = threadIdx.x / 32;
  int warpRow = warpIdx / (BS / WN);
  int warpCol = warpIdx % (BS / WN);
  int laneIdx = threadIdx.x % 32;
  int threadRowInWarp = laneIdx / (WSUBN / TN);
  int threadColInWarp = laneIdx % (WSUBN / TN);

  __align__(16) float a[WMITER * TM] = {};
  __align__(16) float b[WNITER * TN] = {};
  __align__(16) float acc[WMITER * TM][WNITER * TN] = {};

  float4 rA[NUM_A_F4]; // register stage for the next slab
  float4 rB[NUM_B_F4];

  int numPhases = (k + BK - 1) / BK;

  // prologue: load slab 0 and commit it
  prefetch_slab(A, B, n, k, m, blockRow, blockCol, 0, rA, rB);
  commit_slab(rA, rB, As[0], Bs[0]);
  __syncthreads();

  for (int p = 0; p < numPhases; p++) {
    int cur = p & 1;
    int nxt = (p + 1) & 1;

    // issue next slab's global loads NOW (latency hides behind the kk loop)
    if (p + 1 < numPhases)
      prefetch_slab(A, B, n, k, m, blockRow, blockCol, p + 1, rA, rB);

    // compute on the current buffer
    for (int kk = 0; kk < BK; kk++) {
      for (int wm = 0; wm < WMITER; wm++)
        for (int i = 0; i < TM; i += 4) {
          int row = warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i;
          float4 *a4 = reinterpret_cast<float4 *>(&a[wm * TM + i]);
          float4 *As4 = reinterpret_cast<float4 *>(&As[cur][kk][row]);
          *a4 = *As4;
        }
      for (int wn = 0; wn < WNITER; wn++)
        for (int j = 0; j < TN; j += 4) {
          int col = warpCol * WN + wn * WSUBN + threadColInWarp * TN + j;
          float4 *b4 = reinterpret_cast<float4 *>(&b[wn * TN + j]);
          float4 *Bs4 = reinterpret_cast<float4 *>(&Bs[cur][kk][col]);
          *b4 = *Bs4;
        }
      for (int wm = 0; wm < WMITER; wm++)
        for (int wn = 0; wn < WNITER; wn++)
          for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
              acc[wm * TM + i][wn * TN + j] += a[wm * TM + i] * b[wn * TN + j];
    }

    // commit the prefetched registers into the OTHER buffer
    if (p + 1 < numPhases)
      commit_slab(rA, rB, As[nxt], Bs[nxt]);

    __syncthreads();
  }

  // writeback (unchanged)
  for (int wm = 0; wm < WMITER; wm++)
    for (int wn = 0; wn < WNITER; wn++)
      for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
          int outRow = blockRow + warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i;
          int outCol = blockCol + warpCol * WN + wn * WSUBN + threadColInWarp * TN + j;
          if (outRow < n && outCol + 3 < m) {
            float4 *acc4 = reinterpret_cast<float4 *>(&acc[wm * TM + i][wn * TN + j]);
            float4 *out4 = reinterpret_cast<float4 *>(&OUT[outRow * m + outCol]);
            *out4 = *acc4;
          } else if (outRow < n) {
            for (int c = 0; c < 4 && outCol + c < m; c++)
              OUT[outRow * m + outCol + c] = acc[wm * TM + i][wn * TN + j + c];
          }
        }
}

__global__ void kernel_warptiled_ashared_T(float *A, float *B, float *OUT, int n, int k, int m) {
  __shared__ float As[BK][BS]; // transposed: k-major, was As[BS][BK]
  __shared__ float Bs[BK][BS];
  float4 zeros{0, 0, 0, 0};

  int blockRow = blockIdx.y * BS;
  int blockCol = blockIdx.x * BS;

  int warpIdx = threadIdx.x / 32;
  int warpRow = warpIdx / (BS / WN);
  int warpCol = warpIdx % (BS / WN);

  int laneIdx = threadIdx.x % 32;
  int threadRowInWarp = laneIdx / (WSUBN / TN);
  int threadColInWarp = laneIdx % (WSUBN / TN);

  float a[WMITER * TM] = {};
  float b[WNITER * TN] = {};
  float acc[WMITER * TM][WNITER * TN] = {};

  for (int p = 0; p <= k / BK; p++) {
    // ---- load A block: coalesced float4 read, transposed scalar store ----
    for (int i = threadIdx.x; i < BS * BK / 4; i += NUM_THREADS) {
      int s_row = i / (BK / 4);       // m-index within block
      int s_col = (i % (BK / 4)) * 4; // k-index within block
      int g_row = blockRow + s_row;
      int g_col = p * BK + s_col;

      float4 tmp;
      if (g_row >= n || g_col >= k) {
        tmp = zeros;
      } else if (k - g_col >= 4) {
        tmp = *reinterpret_cast<float4 *>(&A[g_row * k + g_col]); // contiguous read kept
      } else {
        tmp.x = (g_col + 0 < k) ? A[g_row * k + g_col + 0] : 0.f;
        tmp.y = (g_col + 1 < k) ? A[g_row * k + g_col + 1] : 0.f;
        tmp.z = (g_col + 2 < k) ? A[g_row * k + g_col + 2] : 0.f;
        tmp.w = (g_col + 3 < k) ? A[g_row * k + g_col + 3] : 0.f;
      }
      // transpose on the way into smem: 4 k-values go to 4 different rows
      As[s_col + 0][s_row] = tmp.x;
      As[s_col + 1][s_row] = tmp.y;
      As[s_col + 2][s_row] = tmp.z;
      As[s_col + 3][s_row] = tmp.w;
    }

    // ---- load B block: unchanged (float4 along m) ----
    for (int i = threadIdx.x; i < BK * BS / 4; i += NUM_THREADS) {
      int s_row = i / (BS / 4);
      int s_col = (i % (BS / 4)) * 4;
      int g_row = p * BK + s_row;
      int g_col = blockCol + s_col;
      float4 *Bs4 = reinterpret_cast<float4 *>(&Bs[s_row][s_col]);
      float4 *B4 = reinterpret_cast<float4 *>(&B[g_row * m + g_col]);
      if (g_row >= k || g_col >= m) {
        *Bs4 = zeros;
      } else if (m - g_col >= 4) {
        *Bs4 = *B4;
      } else {
        for (int c = 0; c < 4; c++)
          Bs[s_row][s_col + c] = (g_col + c < m) ? B[g_row * m + g_col + c] : 0;
      }
    }
    __syncthreads();

    for (int kk = 0; kk < BK; kk++) {
      // ---- A read: now contiguous, vectorized like B (conflict-free) ----
      for (int wm = 0; wm < WMITER; wm++)
        for (int i = 0; i < TM; i += 4) {
          int row = warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i;
          float4 *a4 = reinterpret_cast<float4 *>(&a[wm * TM + i]);
          float4 *As4 = reinterpret_cast<float4 *>(&As[kk][row]);
          *a4 = *As4;
        }
      // ---- B read: unchanged ----
      for (int wn = 0; wn < WNITER; wn++)
        for (int j = 0; j < TN; j += 4) {
          int col = warpCol * WN + wn * WSUBN + threadColInWarp * TN + j;
          float4 *b4 = reinterpret_cast<float4 *>(&b[wn * TN + j]);
          float4 *Bs4 = reinterpret_cast<float4 *>(&Bs[kk][col]);
          *b4 = *Bs4;
        }
      for (int wm = 0; wm < WMITER; wm++)
        for (int wn = 0; wn < WNITER; wn++)
          for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
              acc[wm * TM + i][wn * TN + j] += a[wm * TM + i] * b[wn * TN + j];
    }
    __syncthreads();
  }

  // ---- writeback: unchanged ----
  for (int wm = 0; wm < WMITER; wm++)
    for (int wn = 0; wn < WNITER; wn++)
      for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
          int outRow = blockRow + warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i;
          int outCol = blockCol + warpCol * WN + wn * WSUBN + threadColInWarp * TN + j;
          if (outRow < n && outCol + 3 < m) {
            float4 *acc4 = reinterpret_cast<float4 *>(&acc[wm * TM + i][wn * TN + j]);
            float4 *out4 = reinterpret_cast<float4 *>(&OUT[outRow * m + outCol]);
            *out4 = *acc4;
          } else if (outRow < n) {
            for (int c = 0; c < 4 && outCol + c < m; c++)
              OUT[outRow * m + outCol + c] = acc[wm * TM + i][wn * TN + j + c];
          }
        }
}

__global__ void kernel_warptile(float *A, float *B, float *OUT, int n, int k, int m) {
  __shared__ float As[BS][BK];
  __shared__ float Bs[BK][BS];
  float4 zeros{0, 0, 0, 0};

  int blockRow = blockIdx.y * BS;
  int blockCol = blockIdx.x * BS;

  // warp placement in the block tile
  int warpIdx = threadIdx.x / 32;
  int warpRow = warpIdx / (BS / WN);
  int warpCol = warpIdx % (BS / WN);

  // lane placement inside one warp subtile
  int laneIdx = threadIdx.x % 32;
  int threadRowInWarp = laneIdx / (WSUBN / TN);
  int threadColInWarp = laneIdx % (WSUBN / TN);

  float a[WMITER * TM] = {};
  float b[WNITER * TN] = {};
  float acc[WMITER * TM][WNITER * TN] = {};

  for (int p = 0; p <= k / BK; p++) {
    // load A block (BS x BK), float4 along k
    for (int i = threadIdx.x; i < BS * BK / 4; i += NUM_THREADS) {
      int s_row = i / (BK / 4);
      int s_col = (i % (BK / 4)) * 4;
      int g_row = blockRow + s_row;
      int g_col = p * BK + s_col;
      float4 *As4 = reinterpret_cast<float4 *>(&As[s_row][s_col]);
      float4 *A4 = reinterpret_cast<float4 *>(&A[g_row * k + g_col]);
      if (g_row >= n || g_col >= k) {
        *As4 = zeros;
      } else if (k - g_col >= 4) {
        *As4 = *A4;
      } else {
        for (int c = 0; c < 4; c++)
          As[s_row][s_col + c] = (g_col + c < k) ? A[g_row * k + g_col + c] : 0;
      }
    }
    // load B block (BK x BS), float4 along m
    for (int i = threadIdx.x; i < BK * BS / 4; i += NUM_THREADS) {
      int s_row = i / (BS / 4);
      int s_col = (i % (BS / 4)) * 4;
      int g_row = p * BK + s_row;
      int g_col = blockCol + s_col;
      float4 *Bs4 = reinterpret_cast<float4 *>(&Bs[s_row][s_col]);
      float4 *B4 = reinterpret_cast<float4 *>(&B[g_row * m + g_col]);
      if (g_row >= k || g_col >= m) {
        *Bs4 = zeros;
      } else if (m - g_col >= 4) {
        *Bs4 = *B4;
      } else {
        for (int c = 0; c < 4; c++)
          Bs[s_row][s_col + c] = (g_col + c < m) ? B[g_row * m + g_col + c] : 0;
      }
    }
    __syncthreads();

    for (int kk = 0; kk < BK; kk++) {
      // regM: scalar, strided column read of As (still bank-conflicted)
      for (int wm = 0; wm < WMITER; wm++)
        for (int i = 0; i < TM; i++) {
          int row = warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i;
          a[wm * TM + i] = As[row][kk];
        }
      // regN: float4 contiguous read of Bs row
      for (int wn = 0; wn < WNITER; wn++)
        for (int j = 0; j < TN; j += 4) {
          int col = warpCol * WN + wn * WSUBN + threadColInWarp * TN + j;
          float4 *b4 = reinterpret_cast<float4 *>(&b[wn * TN + j]);
          float4 *Bs4 = reinterpret_cast<float4 *>(&Bs[kk][col]);
          *b4 = *Bs4;
        }
      // outer products into per-thread accumulators
      for (int wm = 0; wm < WMITER; wm++)
        for (int wn = 0; wn < WNITER; wn++)
          for (int i = 0; i < TM; i++)
            for (int j = 0; j < TN; j++)
              acc[wm * TM + i][wn * TN + j] += a[wm * TM + i] * b[wn * TN + j];
    }
    __syncthreads();
  }

  // writeback
  for (int wm = 0; wm < WMITER; wm++)
    for (int wn = 0; wn < WNITER; wn++)
      for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
          int outRow = blockRow + warpRow * WM + wm * WSUBM + threadRowInWarp * TM + i;
          int outCol = blockCol + warpCol * WN + wn * WSUBN + threadColInWarp * TN + j;
          if (outRow < n && outCol + 3 < m) {
            float4 *acc4 = reinterpret_cast<float4 *>(&acc[wm * TM + i][wn * TN + j]);
            float4 *out4 = reinterpret_cast<float4 *>(&OUT[outRow * m + outCol]);
            *out4 = *acc4;
          } else if (outRow < n) {
            for (int c = 0; c < 4 && outCol + c < m; c++)
              OUT[outRow * m + outCol + c] = acc[wm * TM + i][wn * TN + j + c];
          }
        }
}
__global__ void kernel_reg_float4(float *A, float *B, float *OUT, int n, int k, int m) {
  __shared__ float As[BS][BK];
  __shared__ float Bs[BK][BS];

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  float4 zeros{0, 0, 0, 0};

  int blockRow = blockIdx.y * BS;
  int blockCol = blockIdx.x * BS;

  float a[REG] = {};
  float b[REG] = {};
  float acc[REG][REG] = {};

  // phases: load (|) compute
  // Finally writeback
  // For each phase of Block K size
  for (int p = 0; p <= k / BK; p++) {
    // Load into S
    for (int i = tid; i < BS * BK / 4; i += TILE * TILE) {
      /*
         What should thread 0 load, 1 load?
         Well in phase p we'd want BS * BK elems
         We can make each thread load 4 elms // BK = 8, BK/4 = 2
         t0: A[blockRow][p * BK ... p*BK + 3] into As[0][0...3]
         t1: A[blockRow][p * BK +4 ... p*BK + 7] into As[0][0...8]
         t2: A[blockRow+1][p * BK ... p*BK + 3] into
         t3: A[blockRow+1][...]
         t4: A[blockRow+2][...]

         ti: A[bloackRow][p * BS + 4 * i... p * BS + 4 * i + 1]

         When can I do this? whenever k-(...) >= 4
         so when k - g_col >= 4
       */
      int s_row = i / (BK / 4);
      int s_col = (i % (BK / 4)) * 4;

      int g_row = blockRow + s_row;
      int g_col = p * BK + s_col;
      float4 *As4 = reinterpret_cast<float4 *>(&As[s_row][s_col]);
      float4 *A4 = reinterpret_cast<float4 *>(&A[g_row * k + g_col]);

      if (g_row >= n || g_col >= k) {
        *As4 = zeros;
      } else if (k - g_col >= 4) {
        *As4 = *A4;
      } else {
        for (int col = s_col; col < 4 + s_col; col++) {
          g_col = p * BK + col;
          if (g_col < k) {
            As[s_row][col] = A[g_row * k + g_col];
          } else {
            As[s_row][col] = 0;
          }
        }
      }
    }

    for (int i = tid; i < BS * BK / 4; i += TILE * TILE) {
      /*
      Now from b:

      t0: B[p*BK][blockCol ... blockCol + 3] into Bs[0][0..3]
      t1: B[p*BK][blockCol + 4 ... blockCol + 7] into Bs[0][4..7]
      t2: B[p*BK][blockCol + 8 ... blockCol + 11] into Bs[0][...]
      ....
      t3: B[p*BK][blockCol + 4 ... blockCol + 7] into Bs[0][...]
      */

      int s_row = i / (BS / 4);
      int s_col = i % (BS / 4) * 4;

      int g_row = p * BK + s_row;
      int g_col = blockCol + s_col;

      float4 *Bs4 = reinterpret_cast<float4 *>(&Bs[s_row][s_col]);
      float4 *B4 = reinterpret_cast<float4 *>(&B[g_row * m + g_col]);

      if (g_row >= k || g_col >= m) {
        *Bs4 = zeros;
      } else if (m - g_col >= 4) {
        *Bs4 = *B4;
      } else {
        for (int col = s_col; col < 4 + s_col; col++) {
          g_col = blockCol + col;
          if (g_col < m) {
            Bs[s_row][col] = B[g_row * k + g_col];
          } else {
            Bs[s_row][col] = 0;
          }
        }
      }
    }
    __syncthreads();

    // Now that things are loaded we update each kk
    // What we want is the reg by reg grid to accumulate values
    // What should be accumulated in (tx,ty)?
    // if we do the 8 by 8 cont pattern rn it'll be
    // (tx,ty) computes: [blockRow + REG * ty][blockCol + REG * tx]
    for (int kk = 0; kk < BK; kk++) {
      for (int i = 0; i < REG; i++) {
        a[i] = As[threadIdx.y * REG + i][kk];
      }
      for (int j = 0; j < REG; j += 4) {
        // b[j] = Bs[kk][threadIdx.x * REG + j];
        // when t(0,0) and t(0,1) both in same warp access BANK0
        float4 *b4 = reinterpret_cast<float4 *>(&b[j]);
        float4 *Bs4 = reinterpret_cast<float4 *>(&Bs[kk][threadIdx.x * REG + j]);
        *b4 = *Bs4;
      }

      for (int i = 0; i < REG; i++) {
        for (int j = 0; j < REG; j++) {
          acc[i][j] += a[i] * b[j];
        }
      }
    }

    __syncthreads();
  }

  // Done computing now we writeback
  for (int i = 0; i < REG; i++) {
    for (int j = 0; j < REG; j += 4) {
      int outRow = blockRow + threadIdx.y * REG + i;
      int outCol = blockCol + threadIdx.x * REG + j;

      float4 *acc4 = reinterpret_cast<float4 *>(&acc[i][j]);
      float4 *out4 = reinterpret_cast<float4 *>(&OUT[outRow * m + outCol]);
      *out4 = *acc4;
    }
  }
}

__global__ void kernel_shared_reg_tiled(float *A, float *B, float *OUT, int n, int k, int m) {
  // 128 by 8
  __shared__ float As[BS][BK];
  // 8 by 128
  __shared__ float Bs[BK][BS];

  // Now each thread is responsible for 8 times its input
  float a_reg[REG] = {};
  float b_reg[REG] = {};

  float acc[REG][REG] = {};

  // Block (x,y) is responsible for computing REG * TILE
  // y first cuz memory is layed out column wise and cols will share

  int blockRow = blockIdx.y * (TILE * REG);
  int blockCol = blockIdx.x * (TILE * REG);

  int tid = threadIdx.x + blockDim.x * threadIdx.y;

  // Perform tile
  // during phase "p" what should be loaded in the shared memory?
  // We have a 256 KB cache so what is the max we can store
  //
  for (int p = 0; p <= k / BK; p++) {

    // Your goal is to fill only one row col skip rest
    // Remaining TILE * TILE will fill rest :P
    // 256 threads, need to load up 128 by 8
    for (int i = tid; i < BS * BK; i += TILE * TILE) {
      int r = i / BK;
      int c = i % BK;
      // in phase p
      // which row col should be gotten
      // We will want row r and column "c"+ p
      if ((r + blockRow) >= n || (p * BK + c) >= k) {
        As[r][c] = 0;
      } else {
        As[r][c] = A[(r + blockRow) * k + p * BK + c];
      }
    }

    for (int i = tid; i < BS * BK; i += TILE * TILE) {
      int r = i / BS;
      int c = i % BS;

      if ((p * BK + r) >= k || (c + blockCol) >= m) {
        Bs[r][c] = 0;
      } else {
        Bs[r][c] = B[(p * BK + r) * m + c + blockCol];
      }
    }
    __syncthreads();

    for (int kk = 0; kk < BK; kk++) {

      for (int i = 0; i < REG; i++) {
        a_reg[i] = As[threadIdx.y * REG + i][kk];
      }

      for (int j = 0; j < REG; j++) {
        b_reg[j] = Bs[kk][threadIdx.x * REG + j];
      }

      // Store in a_reg
      for (int i = 0; i < REG; i++) {
        for (int j = 0; j < REG; j++) {
          acc[i][j] += a_reg[i] * b_reg[j];
        }
      }
    }
    // Wait for accumulation to happen
    __syncthreads();
  }

  // Write out later
  for (int i = 0; i < REG; i++) {
    for (int j = 0; j < REG; j++) {
      int r = blockRow + threadIdx.y * REG + i;
      int c = blockCol + threadIdx.x * REG + j;
      if (r < n && c < m)
        OUT[r * m + c] = acc[i][j];
    }
  }
}

__global__ void kernel_shared_memory_tiled(float *A, float *B, float *OUT, int n, int k, int m) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  // go from 0 through k-1 in sizes of TILE
  // 0 1 2 3 (k = 4)
  float acc = 0;
  for (int p = 0; p <= k / TILE; p++) {
    if (i >= n || (p * TILE + threadIdx.x) >= k) {
      As[threadIdx.y][threadIdx.x] = 0;
    } else {
      As[threadIdx.y][threadIdx.x] = A[i * k + p * TILE + threadIdx.x];
    }

    if (j >= m || (p * TILE + threadIdx.y) >= k) {
      Bs[threadIdx.y][threadIdx.x] = 0;
    } else {
      Bs[threadIdx.y][threadIdx.x] = B[p * TILE * m + threadIdx.y * m + j];
    }

    __syncthreads();
    for (int kk = 0; kk < TILE; kk++) {
      acc += As[threadIdx.y][kk] * Bs[kk][threadIdx.x];
    }
    __syncthreads();
  }
  if (i < n && j < m)
    OUT[i * m + j] = acc;
}

__global__ void kernel_naive(float *A, float *B, float *OUT, int n, int k, int m) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n || j >= m)
    return;

  float s = 0;
  for (int l = 0; l < k; l++) {
    s += A[i * k + l] * B[m * l + j];
  }
  OUT[i * m + j] = s;
}

__global__ void kernel_naive_very(float *A, float *B, float *OUT, int n, int k, int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= n || j >= m)
    return;

  float s = 0;
  for (int l = 0; l < k; l++) {
    s += A[i * k + l] * B[m * l + j];
  }
  OUT[i * m + j] = s;
}

void matmul_alloc(float *A, float *B, int n, int k, int m) {
  cudaMalloc(&A_c, n * k * sizeof(float));
  cudaMalloc(&B_c, k * m * sizeof(float));
  cudaMalloc(&OUT_c, n * m * sizeof(float));
  // Copy data
  cudaMemcpy(A_c, A, n * k * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_c, B, k * m * sizeof(float), cudaMemcpyHostToDevice);
}

void matmul_gpu(int n, int k, int m) {
  dim3 block(NUM_THREADS);
  dim3 grid((m + BS - 1) / BS, (n + BS - 1) / BS);
  kernel_warptiled_doublebuf<<<grid, block>>>(A_c, B_c, OUT_c, n, k, m);
  cudaDeviceSynchronize();
}

void matmul_get(float *OUT, int n, int m) {
  // Copy data
  cudaMemcpy(OUT, OUT_c, n * m * sizeof(float), cudaMemcpyDeviceToHost);

  // Free up
  cudaFree(A_c);
  cudaFree(B_c);
  cudaFree(OUT_c);
}

// Final CUBLAS
#include <cublas_v2.h>

static cublasHandle_t g_handle = nullptr;

void matmul_gpu_cublas(int n, int k, int m) {
  if (!g_handle)
    cublasCreate(&g_handle);

  const float alpha = 1.0f, beta = 0.0f;
  // row-major C(n x m) = A(n x k) . B(k x m)
  // computed as column-major C^T(m x n) = B^T . A^T  -> pass B then A, both OP_N
  cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, // M, N, K in cuBLAS's (column-major) frame
              &alpha, B_c, m,                              // first operand = B, leading dim = m
              A_c, k,                                      // second operand = A, leading dim = k
              &beta, OUT_c, m);                            // result = OUT, leading dim = m
  cudaDeviceSynchronize();
}
