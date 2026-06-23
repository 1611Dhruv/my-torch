#include "matmul.h"

float *A_c;
float *B_c;
float *OUT_c;
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
  dim3 block(16, 16);
  dim3 grid((m + 15) / 16, (n + 15) / 16);
  kernel_naive<<<grid, block>>>(A_c, B_c, OUT_c, n, k, m);
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
