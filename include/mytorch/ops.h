#ifndef OPS_H
#define OPS_H

#include "mytorch/tensor.h"

namespace torch {
// Carries the device agnostic dispatchers
// Will perform type checks device checks and all
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mult(const Tensor &a, const Tensor &b);
Tensor div(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);

Tensor neg(const Tensor &a);
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor exp(const Tensor &a);
Tensor ln(const Tensor &a);

// Special Fused kernels
Tensor relu(const Tensor &a);
Tensor relu_back(const Tensor &a, const Tensor &g);

// Reductions
Tensor sum(const Tensor &a, std::vector<int64_t> dims, bool keep_dim = true);
Tensor max(const Tensor &a, std::vector<int64_t> dims, bool keep_dim = true);

Tensor cast(const Tensor &a, DType dtype);
/*
 NOTE: Future

Tensor mean(const Tensor &a, std::vector<int64_t> dims, bool keep_dim = true);
*/

namespace cpu {

// CPU specific dispatchers
// NOTE: `out` is allocated (uninitialized) by the dispatcher, which owns all
// shape math. A backend that needs a zeroed buffer must zero it itself.
Tensor add(const Tensor &a, const Tensor &b, Tensor &out);
Tensor sub(const Tensor &a, const Tensor &b, Tensor &out);
Tensor mult(const Tensor &a, const Tensor &b, Tensor &out);
Tensor div(const Tensor &a, const Tensor &b, Tensor &out);
Tensor matmul(const Tensor &a, const Tensor &b, Tensor &out, int64_t B,
              int64_t M, int64_t K, int64_t N);

Tensor neg(const Tensor &a, Tensor &out);
Tensor sin(const Tensor &a, Tensor &out);
Tensor cos(const Tensor &a, Tensor &out);
Tensor exp(const Tensor &a, Tensor &out);
Tensor ln(const Tensor &a, Tensor &out);

// Reductions
// NOTE: Always keeps dim = true
Tensor sum(const Tensor &a, Tensor &out, const std::vector<int64_t> &dims);
Tensor max(const Tensor &a, Tensor &out, const std::vector<int64_t> &dims);

// Special fused kernels
Tensor relu(const Tensor &a, Tensor &out);
Tensor relu_back(const Tensor &a, const Tensor &g, Tensor &out);

Tensor cast(const Tensor &a, Tensor &out);
/*
 NOTE: Future

Tensor mean(const Tensor &a, Tensor &out, std::vector<int64_t> dims);
*/

} // namespace cpu

namespace cuda {

// CUDA specific dispatchers
// NOTE: `out` is allocated (uninitialized) by the dispatcher, which owns all
// shape math. A backend that needs a zeroed buffer must zero it itself.
Tensor add(const Tensor &a, const Tensor &b, Tensor &out);
Tensor sub(const Tensor &a, const Tensor &b, Tensor &out);
Tensor mult(const Tensor &a, const Tensor &b, Tensor &out);
Tensor div(const Tensor &a, const Tensor &b, Tensor &out);
Tensor matmul(const Tensor &a, const Tensor &b, Tensor &out, int64_t B,
              int64_t M, int64_t K, int64_t N);

Tensor neg(const Tensor &a, Tensor &out);
Tensor sin(const Tensor &a, Tensor &out);
Tensor cos(const Tensor &a, Tensor &out);
Tensor exp(const Tensor &a, Tensor &out);
Tensor ln(const Tensor &a, Tensor &out);

Tensor contiguous(const Tensor &a, Tensor &out);
template <typename scalar_t>
Tensor fill(const Tensor &a, Tensor &out, scalar_t t);

// Reductions
// NOTE: Always keeps dim = true
Tensor sum(const Tensor &a, Tensor &out, const std::vector<int64_t> &dims);
Tensor max(const Tensor &a, Tensor &out, const std::vector<int64_t> &dims);

// Special fused kernels
Tensor relu(const Tensor &a, Tensor &out);
Tensor relu_back(const Tensor &a, const Tensor &g, Tensor &out);

Tensor cast(const Tensor &a, Tensor &out);

/*
 NOTE: Future

Tensor mean(const Tensor &a, Tensor &out, std::vector<int64_t> dims);
*/

} // namespace cuda

} // namespace torch

#endif
