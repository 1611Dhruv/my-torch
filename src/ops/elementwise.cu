#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
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

template <bool float_only = false, typename Op>
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
      if constexpr (float_only && !std::is_floating_point_v<scalar_t>) {
        throw std::invalid_argument(
            "Called elementwise_binary_wrapper which is "
            "float only on a non float type");
      } else {
        binary_kernel_strided<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), n, op, stride);
      }
    });
  } else {
    // Both are contiguous so just directly use quick kernel
    DISPATCH_OP(a.dtype(), [&] {
      if constexpr (float_only && !std::is_floating_point_v<scalar_t>) {
        throw std::invalid_argument(
            "Called elementwise_binary_wrapper which is "
            "float only on a non float type");
      } else {
        binary_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), n, op);
      }
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

template <const bool float_only = false, typename Op>
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
      if constexpr (float_only && !std::is_floating_point_v<scalar_t>) {
        throw std::invalid_argument("Called elementwise_unary_wrapper which is "
                                    "float only on a non float type");
      } else {
        unary_kernel_strided<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n, op, stride);
      }
    });
  } else {
    DISPATCH_OP(a.dtype(), [&] {
      if constexpr (float_only && !std::is_floating_point_v<scalar_t>) {
        throw std::invalid_argument("Called elementwise_unary_wrapper which is "
                                    "float only on a non float type");
      } else {
        unary_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n, op);
      }
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
  return elementwise_unary_wrapper<true>(
      a, out, [] __device__(auto x) { return std::sin(x); });
}

Tensor cos(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper<true>(
      a, out, [] __device__(auto x) { return std::cos(x); });
}

Tensor exp(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper<true>(
      a, out, [] __device__(auto x) { return std::exp(x); });
}

Tensor ln(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper<true>(
      a, out, [] __device__(auto x) { return std::log(x); });
}

Tensor sqrt(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper<true>(
      a, out, [] __device__(auto x) { return std::sqrt(x); });
}

Tensor scale(const Tensor &a, Tensor &out, double s) {
  return elementwise_unary_wrapper(a, out,
                                   [s] __device__(auto x) { return x * s; });
}

Tensor shift(const Tensor &a, Tensor &out, double s) {
  return elementwise_unary_wrapper(a, out,
                                   [s] __device__(auto x) { return x + s; });
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
    return x > 0 ? x : 0;
  });
}
Tensor relu_back(const Tensor &a, const Tensor &g, Tensor &out) {
  return elementwise_binary_wrapper(
      a, g, out, [] __device__(auto x, auto y) { return (x > 0) ? y : 0; });
}

Tensor silu(const Tensor &a, Tensor &out) {
  return elementwise_unary_wrapper<true>(a, out, [] __device__(auto x) {
    auto sigmoid = [] __device__(auto x) {
      auto one = decltype(x)(1);
      return (one / (one + ::exp(-x)));
    };
    return x * sigmoid(x);
  });
}

Tensor silu_back(const Tensor &a, const Tensor &g, Tensor &out) {
  return elementwise_binary_wrapper<true>(
      a, g, out, [] __device__(auto x, auto y) {
        auto sigmoid = [] __device__(auto x) {
          auto one = decltype(x)(1);
          return (one / (one + ::exp(-x)));
        };
        auto sig = sigmoid(x);
        return y * (sig + x * sig * (1 - sig));
      });
}

template <typename src_t, typename dest_t, const int BS = 128>
__global__ void cast_kernal(const src_t *A, dest_t *out, int64_t N) {
  int tid = threadIdx.x;
  int boff = BS * blockIdx.x;
  const int tt = blockDim.x;

  for (int i = tid; i < BS; i += tt) {
    int id = boff + i;
    if (id < N) {
      out[id] = static_cast<dest_t>(A[id]);
    }
  }
}

template <typename src_t, typename dest_t, const int BS = 128>
__global__ void cast_kernal_contig(const src_t *A, dest_t *out, int64_t N,
                                   UnaryStridedDims dim) {
  int tid = threadIdx.x;
  int boff = BS * blockIdx.x;
  const int tt = blockDim.x;

  for (int i = tid; i < BS; i += tt) {
    if (boff + i < N) {
      int id = boff + i;
      int i_off = 0;
      for (int j = dim.ndim - 1; j >= 0; j--) {
        i_off += (id % dim.shape[j]) * dim.strides[j];
        id /= dim.shape[j];
      }
      out[boff + i] = static_cast<dest_t>(A[i_off]);
    }
  }
}

Tensor cast(const Tensor &a, Tensor &out) {
  const bool use_stride = !a.is_contiguous();
  UnaryStridedDims stride;
  if (!a.is_contiguous()) {
    if (a.shape().size() > MAX_DIM) {
      std::ostringstream oss;
      oss << "cuda binary: support only maximum of " << MAX_DIM
          << " dim shapes";
      throw std::invalid_argument(oss.str());
      ;
    }

    stride.ndim = a.shape().size();
    for (int64_t i = 0; i < stride.ndim; i++) {
      stride.shape[i] = a.shape()[i];
      stride.strides[i] = a.strides()[i];
    }
  }

  DISPATCH_OP_AS(a.dtype(), src_t, [&]() {
    DISPATCH_OP_AS(out.dtype(), dest_t, [&] {
      int N = a.numel();
      constexpr int T = 256;
      constexpr int BS = T << 2;
      int ng = (N + BS - 1) / BS;
      if (use_stride) {
        cast_kernal_contig<src_t, dest_t, BS>
            <<<ng, T>>>(a.data_ptr<src_t>(), out.data_ptr<dest_t>(), N, stride);
      } else {
        cast_kernal<src_t, dest_t, BS>
            <<<ng, T>>>(a.data_ptr<src_t>(), out.data_ptr<dest_t>(), N);
      }
      CUDA_CHECK(cudaGetLastError());
    });
  });
  return out;
}

template <typename scalar_t, const int T, const bool add = false>
__global__ void index_select_kernel(const int *idx, const scalar_t *src,
                                    scalar_t *dest, int N, int R, int M) {
  __shared__ int idx_s[T];
  int tid = threadIdx.x;
  int boff = blockIdx.x * T;
  int i = tid + boff;

  if (i < N) {
    idx_s[tid] = idx[i];
    assert(idx_s[tid] >= 0 && idx_s[tid] < M);
  }
  __syncthreads();

  int wid = tid / WARP_SIZE;
  int lid = tid % WARP_SIZE;

  constexpr int NUM_WARPS = (T + WARP_SIZE - 1) / WARP_SIZE;
  for (int woff = wid; woff < T; woff += NUM_WARPS) {
    if (woff + boff < N) {
      for (int j = lid; j < R; j += WARP_SIZE) {
        if constexpr (add) {
          atomicAdd(&dest[idx_s[woff] * R + j], src[(woff + boff) * R + j]);
        } else {
          dest[(woff + boff) * R + j] = src[idx_s[woff] * R + j];
        }
      }
    }
  }
}

Tensor index_select(const Tensor &a, const Tensor &idx, Tensor &out) {
  int N = idx.numel();
  auto a_contig = a.contiguous();
  constexpr int T = 256;
  dim3 grid((N + T - 1) / T);

  int R = a_contig.numel() / a_contig.shape()[0];
  int M = a_contig.shape()[0];

  DISPATCH_OP(a.dtype(), [&] {
    if (idx.dtype() != DType::Int32) {
      throw std::invalid_argument("Called index_select where idx is not int32");
    }
    index_select_kernel<scalar_t, T>
        <<<grid, T>>>(idx.data_ptr<int>(), a_contig.data_ptr<scalar_t>(),
                      out.data_ptr<scalar_t>(), N, R, M);
  });
  CUDA_CHECK(cudaGetLastError());
  return out;
}

Tensor index_select_back(const Tensor &g, Tensor idx, Tensor &out) {
  int N = idx.numel();
  constexpr int T = 256;
  dim3 grid((N + T - 1) / T);

  int R = out.numel() / out.shape()[0];
  int M = out.shape()[0];

  DISPATCH_OP(out.dtype(), [&] {
    if (idx.dtype() != DType::Int32) {
      throw std::invalid_argument("Called index_select where idx is not int32");
    }
    if constexpr (!std::is_same<scalar_t, uint8_t>() &&
                  !std::is_same<scalar_t, int32_t>()) {
      index_select_kernel<scalar_t, T, true>
          <<<grid, T>>>(idx.data_ptr<int>(), g.data_ptr<scalar_t>(),
                        out.data_ptr<scalar_t>(), N, R, M);
    }
  });
  CUDA_CHECK(cudaGetLastError());
  return out;
}

} // namespace cuda
} // namespace torch
