#include "matmul.h"

float *A_c;
float *B_c;
float *OUT_c;

#define TILE 16
#define REG 8
#define BS (TILE * REG)
#define BK 8

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
  dim3 block(TILE, TILE);
  dim3 grid((m + BS - 1) / BS, (n + BS - 1) / BS);
  kernel_reg_float4<<<grid, block>>>(A_c, B_c, OUT_c, n, k, m);
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
