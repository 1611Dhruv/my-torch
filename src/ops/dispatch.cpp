#include "mytorch/ops.h"
#include "mytorch/storage.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace torch {

// binary ops
template <typename CpuFn, typename CudaFn>
Tensor elementwise_binary_dispatch(const Tensor &a, const Tensor &b,
                                   CpuFn cpu_op, CudaFn cuda_op) {
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("elementwise dispatch: tensors should have the "
                                "same dtype, casting not supported yet");
  }

  if (a.device() != b.device()) {
    throw std::invalid_argument(
        "elementwise dispatch: tensor device mismatch"); // TODO: implement data
                                                         // transfer instead of
                                                         // throwing
  }

  int64_t adim = a.shape().size();
  int64_t bdim = b.shape().size();
  int64_t ndim = std::max(adim, bdim);
  std::vector<int64_t> target_shape(ndim, 1);

  int64_t i = adim - 1;
  int64_t j = bdim - 1;
  int64_t k = std::max(i, j);

  while (i >= 0 && j >= 0) {
    target_shape[k--] = std::max(a.shape()[i--], b.shape()[j--]);
  }

  while (i >= 0)
    target_shape[k--] = a.shape()[i--];

  while (j >= 0)
    target_shape[k--] = b.shape()[j--];

  Tensor _a = a.broadcast_to(target_shape);
  Tensor _b = b.broadcast_to(target_shape);

  // Dispatch owns the shape, so it owns the allocation. Uninitialized: every
  // element is written by both backends.
  Tensor out(target_shape, a.dtype(), a.device());

  if (a.device() == CUDA)
    return cuda_op(_a, _b, out);
  else
    return cpu_op(_a, _b, out);
}

Tensor add(const Tensor &a, const Tensor &b) {
  return elementwise_binary_dispatch(a, b, cpu::add, cuda::add);
}
Tensor sub(const Tensor &a, const Tensor &b) {
  return elementwise_binary_dispatch(a, b, cpu::sub, cuda::sub);
}
Tensor mult(const Tensor &a, const Tensor &b) {
  return elementwise_binary_dispatch(a, b, cpu::mult, cuda::mult);
}
Tensor div(const Tensor &a, const Tensor &b) {
  return elementwise_binary_dispatch(a, b, cpu::div, cuda::div);
}

// unary ops
template <typename CpuFn, typename CudaFn>
Tensor elementwise_unary_dispatch(const Tensor &a, CpuFn cpu_op,
                                  CudaFn cuda_op) {
  Tensor out(a.shape(), a.dtype(), a.device());

  if (a.device() == CUDA)
    return cuda_op(a, out);
  else
    return cpu_op(a, out);
}

Tensor neg(const Tensor &a) {
  return elementwise_unary_dispatch(a, cpu::neg, cuda::neg);
}
Tensor sin(const Tensor &a) {
  return elementwise_unary_dispatch(a, cpu::sin, cuda::sin);
}
Tensor cos(const Tensor &a) {
  return elementwise_unary_dispatch(a, cpu::cos, cuda::cos);
}
Tensor exp(const Tensor &a) {
  return elementwise_unary_dispatch(a, cpu::exp, cuda::exp);
}

// The matmul backends can consume a tensor whose last two dims are swapped on
// an otherwise contiguous buffer (CUDA templates the kernel on it, CPU indexes
// by stride). Anything else has to be materialized, and that decision belongs
// here so both devices see the same contract.
static Tensor matmul_operand(const Tensor &t) {
  if (t.is_contiguous())
    return t;

  int64_t n = t.shape().size();
  bool transposed =
      (t.strides()[n - 2] == 1 && t.strides()[n - 1] == t.shape()[n - 2]);
  if (transposed)
    return t;

  return t.contiguous();
}

// matmul
Tensor matmul(const Tensor &a, const Tensor &b) {
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("matmul dispatch: tensors should have the same "
                                "dtype, casting not supported yet");
  }

  if (a.shape().size() < 2 || b.shape().size() < 2) {
    throw std::invalid_argument("matmul dispatch: invalid tensor shape");
  }

  if (a.device() != b.device()) {
    throw std::invalid_argument(
        "matmul dispatch: tensor device mismatch"); // TODO: implement data
                                                    // transfer instead of
                                                    // throwing
  }

  auto AS = a.shape();
  auto BS = b.shape();

  int64_t K = AS.back();
  AS.pop_back();
  int64_t M = AS.back();
  AS.pop_back();

  int64_t N = BS.back();
  BS.pop_back();

  if (BS.back() != K) {
    throw std::invalid_argument(
        "matmul dispatch: the reducing dimension K is not the same");
  } else {
    BS.pop_back();
  }

  int64_t asz = AS.size();
  int64_t bsz = BS.size();
  std::vector<int64_t> BATCH(std::max(asz, bsz), 1);
  asz--;
  bsz--;

  int64_t t = BATCH.size() - 1;
  while (asz >= 0 && bsz >= 0) {
    BATCH[t--] = std::max(AS[asz], BS[bsz]);
    asz--;
    bsz--;
  }

  while (asz >= 0) {
    BATCH[t--] = AS[asz--];
  }
  while (bsz >= 0) {
    BATCH[t--] = BS[bsz--];
  }

  auto BATCH_A = BATCH;
  BATCH_A.push_back(M);
  BATCH_A.push_back(K);

  auto BATCH_B = BATCH;
  BATCH_B.push_back(K);
  BATCH_B.push_back(N);

  Tensor _a = a.broadcast_to(BATCH_A);
  Tensor _b = b.broadcast_to(BATCH_B);

  // Otherwise we find our batch
  int64_t B = 1;
  for (auto b : BATCH)
    B *= b;

  auto out_shape = BATCH;
  out_shape.push_back(M);
  out_shape.push_back(N);

  // Both backends are documented to accept only contiguous or 2-D-transposed
  // inputs, so normalize here rather than letting each device decide for itself
  // (CPU used to silently copy via reshape(), CUDA used to throw).
  Tensor A = matmul_operand(_a);
  Tensor Bm = matmul_operand(_b);

  Tensor output({B, M, N}, _a.dtype(), _a.device());
  if (_a.device() == CUDA)
    cuda::matmul(A, Bm, output, B, M, K, N);
  else
    cpu::matmul(A, Bm, output, B, M, K, N);

  return output.reshape(out_shape);
}

// reductions
template <typename CpuFn, typename CudaFn>
Tensor reduce(const Tensor &a, std::vector<int64_t> &dims, bool keep_dim,
              CpuFn cpu_op, CudaFn cuda_op) {
  // CPU OP and Cuda op always keeps dims

  std::vector<int64_t> target_shape = a.shape();
  if (dims.empty()) {
    dims.assign(a.ndim(), 0);
    std::iota(dims.begin(), dims.end(), 0);
  }
  for (auto &dim : dims) {
    if (dim < 0)
      dim += a.ndim();
    if (dim < 0) {
      throw std::invalid_argument("reduce: Dimension passed in is still < 0");
    }
    if (dim >= a.ndim()) {
      throw std::invalid_argument(
          "reduce: Dimension passed in is still >= maxdim");
    }
    target_shape[dim] = 1;
  }

  // Need this for the subsequent things
  std::sort(dims.begin(), dims.end());
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());

  Tensor out(target_shape, a.dtype(), a.device());

  Tensor result =
      (a.device() == CPU) ? cpu_op(a, out, dims) : cuda_op(a, out, dims);
  if (!keep_dim) {
    return result.squeeze(dims);
  }
  return result;
}

Tensor sum(const Tensor &a, std::vector<int64_t> dims, bool keep_dim) {
  return reduce(a, dims, keep_dim, cpu::sum, cuda::sum);
}

// Special kernels
Tensor relu(const Tensor &a) {
  return elementwise_unary_dispatch(a, cpu::relu, cuda::relu);
}
Tensor relu_back(const Tensor &a, const Tensor &g) {
  return elementwise_binary_dispatch(a, g, cpu::relu_back, cuda::relu_back);
}

} // namespace torch
