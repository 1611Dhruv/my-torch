#include "mytorch/cuda_utils.h"
#include "mytorch/tensor.h"
#include <vector>

namespace torch {
namespace cuda {

template <typename scalar_t, typename ReduceOp>
__global__ void reduce(const scalar_t *inp, scalar_t *out, ReduceDims dim,
                       ReduceOp op, scalar_t ident) {
  int tid = threadIdx.x;
  int blk = blockIdx.x * blockDim.x;
  int64_t koff = 0;

  int64_t x = blk + tid;
  for (int i = dim.nkeep - 1; i >= 0; i--) {
    koff += dim.keep_stride[i] * (x % dim.keep_shape[i]);
    x /= dim.keep_shape[i];
  }
  // Out of bounds as couldn't completely normalize
  if (x)
    return;

  scalar_t acc = ident;
  int64_t idx[MAX_DIM] = {0};
  int64_t roff = 0;
  bool done = false;
  while (!done) {
    scalar_t elm = inp[koff + roff];
    acc = op(acc, elm);
    int d = dim.nreduce - 1;

    idx[d]++;
    roff += dim.reduce_stride[d];

    while (idx[d] == dim.reduce_shape[d]) {
      roff -= dim.reduce_stride[d] * dim.reduce_shape[d];
      idx[d] = 0;
      d--;

      if (d < 0) {
        done = true;
        break;
      }
      idx[d]++;
      roff += dim.reduce_stride[d];
    }
  }

  // Assuming out is contiguous, (just means our tid + blk)
  out[tid + blk] = acc;
}

template <typename scalar_t, typename ReduceOp>
void reduce_helper(const Tensor &a, Tensor &out,
                   const std::vector<int64_t> &dims, scalar_t ident,
                   ReduceOp op) {
  if (dims.size() > MAX_DIM) {
    throw std::invalid_argument("too many reduce dims on GPU, you have issues");
  }
  const auto &a_shape = a.shape();
  const auto &a_stride = a.strides();

  if (a.ndim() - dims.size() > MAX_DIM) {
    throw std::invalid_argument("too many keep dims on GPU, you have issues");
  }

  ReduceDims red_dim;
  red_dim.nreduce = dims.size();
  red_dim.nkeep = a.ndim() - dims.size();

  int j = 0;
  int k = 0;
  for (int i = 0; i < a.ndim(); i++) {
    if (j < dims.size() && i == dims[j]) {
      red_dim.reduce_shape[j] = a_shape[i];
      red_dim.reduce_stride[j] = a_stride[i];
      j++;
    } else {
      red_dim.keep_shape[k] = a_shape[i];
      red_dim.keep_stride[k] = a_stride[i];
      k++;
    }
  }
  int64_t outs = out.numel();
  constexpr int T = 256;
  const int64_t grid = (outs + T - 1) / T;
  reduce<<<grid, T>>>(a.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), red_dim,
                      op, ident);
  CUDA_CHECK(cudaGetLastError());
}

Tensor sum(const Tensor &a, Tensor &out, const std::vector<int64_t> &dims) {
  DISPATCH_OP(a.dtype(), [&]() {
    reduce_helper(a, out, dims, static_cast<scalar_t>(0),
                  [] __device__(scalar_t x, scalar_t y) { return x + y; });
  });
  return out;
}
} // namespace cuda

} // namespace torch
