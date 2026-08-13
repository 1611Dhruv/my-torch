#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"

namespace torch {
namespace cuda {

Tensor sum(Tensor &a, const std::vector<int64_t> &dims, bool keep_dim) { return a; }
} // namespace cuda

} // namespace torch
