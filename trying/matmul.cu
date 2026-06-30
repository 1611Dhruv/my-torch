#include "matmul.h"

float *A_c;
float *B_c;
float *OUT_c;

#define TILE 16
#define REG 8
#define BLOCK_SZ (TILE * REG)
#define BK 8

__global__ void kernel_shared_reg_tiled(float *A, float *B, float *OUT, int n, int k, int m) {
  // 128 by 8
  __shared__ float As[BLOCK_SZ][BK];
  // 8 by 128
  __shared__ float Bs[BK][BLOCK_SZ];

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
    for (int i = tid; i < BLOCK_SZ * BK; i += TILE * TILE) {
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

      if ((p * BK + c) >= k || (r + blockCol) >= m) {
        Bs[c][r] = 0;
      } else {
        Bs[c][r] = B[(p * BK + c) * m + r + blockCol];
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
  dim3 grid((m + BLOCK_SZ - 1) / BLOCK_SZ, (n + BLOCK_SZ - 1) / BLOCK_SZ);
  kernel_shared_reg_tiled<<<grid, block>>>(A_c, B_c, OUT_c, n, k, m);
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
