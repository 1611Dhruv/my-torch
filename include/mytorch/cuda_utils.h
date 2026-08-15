#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime_api.h>
#include <stdexcept>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)
namespace torch {
constexpr int64_t MAX_DIM = 8;

struct BinaryStridedDims {
  int ndim;
  int64_t shape[MAX_DIM];
  int64_t a_strides[MAX_DIM];
  int64_t b_strides[MAX_DIM];
};

struct UnaryStridedDims {
  int ndim;
  int64_t shape[MAX_DIM];
  int64_t strides[MAX_DIM];
};

struct ReduceDims {
  int64_t nreduce;
  int64_t reduce_shape[MAX_DIM];
  int64_t reduce_stride[MAX_DIM];

  int64_t nkeep;
  int64_t keep_shape[MAX_DIM];
  int64_t keep_stride[MAX_DIM];
};

} // namespace torch

#endif // CUDA_UTILS_H
