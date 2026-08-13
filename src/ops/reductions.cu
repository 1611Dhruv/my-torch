#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"

namespace torch {
namespace cuda {

Tensor sum(const Tensor &a, Tensor &out, const std::vector<int64_t> &dims) { return out; }
} // namespace cuda

} // namespace torch
