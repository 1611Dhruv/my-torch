#ifndef OPS_H
#define OPS_H

#include "mytorch/tensor.h"

namespace torch {
// Device-agnostic dispatchers: perform type/device checks, then route to cpu/cuda.
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mult(const Tensor &a, const Tensor &b);
Tensor neg(const Tensor &a);

/*
 NOTE: Future
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor exp(const Tensor &a);
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor ln(Tensor &a);
*/

namespace cpu {

// CPU specific dispatchers
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mult(const Tensor &a, const Tensor &b);
Tensor neg(const Tensor &a);
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor exp(const Tensor &a);

/*
 NOTE: Future
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor ln(Tensor &a);
*/

} // namespace cpu

namespace cuda {

// CUDA specific dispatchers
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mult(const Tensor &a, const Tensor &b);

/*
 NOTE: Future
Tensor neg(const Tensor &a);
Tensor matmul(const Tensor &a, const Tensor &b);
Tensor sin(Tensor &a);
Tensor cos(Tensor &a);
Tensor exp(Tensor &a);
Tensor ln(Tensor &a);
*/

} // namespace cuda

} // namespace torch

#endif
