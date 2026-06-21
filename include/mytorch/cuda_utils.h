#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime_api.h>
#include <stdexcept>

#define CUDA_CHECK(expr)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = (expr);                                                                                          \
    if (err != cudaSuccess) {                                                                                          \
      throw std::runtime_error(cudaGetErrorString(err));                                                               \
    }                                                                                                                  \
  } while (0)

#endif // CUDA_UTILS_H
