#include "matmul.h"

float *A_c;
float *B_c;
float *OUT_c;

#define TILE 16

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
  dim3 grid((m + TILE - 1) / TILE, (n + TILE - 1) / TILE);
  kernel_shared_memory_tiled<<<grid, block>>>(A_c, B_c, OUT_c, n, k, m);
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
