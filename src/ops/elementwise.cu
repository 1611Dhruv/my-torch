#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <sstream>
#include <vector>

namespace torch {
namespace cuda {

template <typename scalar_t, typename Op>
__global__ void binary_kernel(const scalar_t *a, const scalar_t *b,
                              scalar_t *out, int64_t n, Op op) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  out[i] = op(a[i], b[i]);
}

template <typename scalar_t, typename Op>
__global__ void unary_kernel(const scalar_t *a, scalar_t *out, int64_t n,
                             Op op) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  out[i] = op(a[i]);
}

// Added support for non contiguous element wise ops
// Dont care about internal offsets, can assume data_ptr returns start of this
// tensor

template <typename scalar_t, typename Op>
__global__ void binary_kernel_strided(const scalar_t *a, const scalar_t *b,
                                      scalar_t *out, int64_t n, Op op,
                                      BinaryStridedDims strides) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  int64_t a_i = 0;
  int64_t b_i = 0;

  // Idea is to convert i -> [] [] [] <-- coords and use these coords to
  // generate a and b's offsets
  int64_t curr = i;
  for (int64_t j = strides.ndim - 1; j >= 0; j--) {
    int64_t coord = curr % strides.shape[j];
    curr = curr / strides.shape[j];
    a_i += coord * strides.a_strides[j];
    b_i += coord * strides.b_strides[j];
  }

  // Works but will be slow? cuz HBM will have random access pattern and
  // we waste some ops calculating up well not really cuz GPU  go brrr
  out[i] = op(a[a_i], b[b_i]);
}

template <typename scalar_t, typename Op>
__global__ void unary_kernel_strided(const scalar_t *a, scalar_t *out,
                                     int64_t n, Op op,
                                     UnaryStridedDims strides) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= n)
    return;

  int64_t curr = i;
  int64_t a_i = 0;
  for (int64_t j = strides.ndim - 1; j >= 0; j--) {
    int64_t coord = curr % strides.shape[j];
    curr = curr / strides.shape[j];
    a_i += coord * strides.strides[j];
  }

  out[i] = op(a[a_i]);
}

template <typename Op>
Tensor elementwise_binary_wrapper(const Tensor &a, const Tensor &b, Tensor &out,
                                  Op op) {
  assert(a.dtype() == b.dtype());

  int64_t n = a.numel();
  const auto &shape = a.shape();
  int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;

  if (!a.is_contiguous() || !b.is_contiguous()) {
    const auto &stride_a = a.strides();
    const auto &stride_b = b.strides();

    // Max we support is 8 dims for now, if you need more you seem to have
    // issues....
    if (shape.size() > MAX_DIM) {
      std::ostringstream oss;
      oss << "cuda binary: support only maximum of " << MAX_DIM
          << " dim shapes";
      throw std::invalid_argument(oss.str());
    }

    // Populate the Stride Op
    BinaryStridedDims stride;
    stride.ndim = shape.size();
    for (int64_t i = 0; i < stride.ndim; i++) {
      stride.shape[i] = shape[i];
      stride.a_strides[i] = stride_a[i];
      stride.b_strides[i] = stride_b[i];
    }
    DISPATCH_OP(a.dtype(), [&] {
      binary_kernel_strided<scalar_t>
          <<<blocks, threads>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                                out.data_ptr<scalar_t>(), n, op, stride);
    });
  } else {
    // Both are contiguous so just directly use quick kernel
    DISPATCH_OP(a.dtype(), [&] {
      binary_kernel<scalar_t>
          <<<blocks, threads>>>(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                                out.data_ptr<scalar_t>(), n, op);
    });
  }

  CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor add(const Tensor &a, const Tensor &b, Tensor &out) {
  return elementwise_binary_wrapper(
      a, b, out, [] __device__(auto x, auto y) { return x + y; });
}

Tensor sub(const Tensor &a, const Tensor &b, Tensor &out) {
  return elementwise_binary_wrapper(
      a, b, out, [] __device__(auto x, auto y) { return x - y; });
}

Tensor mult(const Tensor &a, const Tensor &b, Tensor &out) {
  return elementwise_binary_wrapper(
      a, b, out, [] __device__(auto x, auto y) { return x * y; });
}
Tensor div(const Tensor &a, const Tensor &b, Tensor &out) {
  return elementwise_binary_wrapper(
      a, b, out, [] __device__(auto x, auto y) { return x / y; });
}

template <typename Op>
Tensor elementwise_unary_wrapper(const Tensor &a, Tensor &out, Op op) {
  int64_t n = a.numel();
  int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;

  if (!a.is_contiguous()) {
    if (a.shape().size() > MAX_DIM) {
      std::ostringstream oss;
      oss << "cuda binary: support only maximum of " << MAX_DIM
          << " dim shapes";
      throw std::invalid_argument(oss.str());
      ;
    }
    UnaryStridedDims stride;
    stride.ndim = a.shape().size();
    for (int64_t i = 0; i < stride.ndim; i++) {
      stride.shape[i] = a.shape()[i];
      stride.strides[i] = a.strides()[i];
    }

    DISPATCH_OP(a.dtype(), [&] {
      unary_kernel_strided<scalar_t><<<blocks, threads>>>(
          a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n, op, stride);
    });
  } else {
    DISPATCH_OP(a.dtype(), [&] {
      unary_kernel<scalar_t><<<blocks, threads>>>(
          a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n, op);
    });
  }

  CUDA_CHECK(cudaGetLastError());

  return out;
}

Tensor neg(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper(a, out,
                                   [] __device__(auto x) { return -x; });
}

Tensor sin(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper(a, out,
                                   [] __device__(auto x) { return sinf(x); });
}

Tensor cos(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper(a, out,
                                   [] __device__(auto x) { return cosf(x); });
}

Tensor exp(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper(a, out,
                                   [] __device__(auto x) { return expf(x); });
}

Tensor contiguous(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper(a, out, [] __device__(auto x) { return x; });
}

template <typename scalar_t>
Tensor fill(const Tensor &a, Tensor &out, scalar_t t) {
  return elementwise_unary_wrapper(a, out,
                                   [t] __device__(auto x) { return t; });
}

// Special Kernels
Tensor relu(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper(a, out, [] __device__(auto x) {
    // max(a,b) =
    // (a + b) / 2 + abs(a - b) / 2
    // a/2 + b/2 + a/2 - b/2
    return (x + abs(x)) / 2;
  });
}
Tensor relu_back(const Tensor &a, const Tensor &g, Tensor &out) {
  return elementwise_binary_wrapper(
      a, g, out, [] __device__(auto x, auto y) { return (x > 0) ? y : 0; });
}

} // namespace cuda
} // namespace torch
