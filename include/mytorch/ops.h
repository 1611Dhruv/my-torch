#ifndef OPS_H
#define OPS_H

#include "mytorch/tensor.h"

namespace torch {
// Carries the device agnostic dispatchers
// Will perform type checks device checks and all
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mult(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);

Tensor neg(const Tensor &a);
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor exp(const Tensor &a);

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
Tensor matmul(const Tensor &a, const Tensor &b);

Tensor neg(const Tensor &a);
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor exp(const Tensor &a);

/*
 NOTE: Future
Tensor ln(Tensor &a);
*/

} // namespace cpu

namespace cuda {

// CUDA specific dispatchers
Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mult(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);

Tensor neg(const Tensor &a);
Tensor sin(const Tensor &a);
Tensor cos(const Tensor &a);
Tensor exp(const Tensor &a);

/*
 NOTE: Future
<<<<<<< HEAD
=======
Tensor neg(const Tensor &a);
Tensor matmul(const Tensor &a, const Tensor &b);
>>>>>>> upstream/main
Tensor sin(Tensor &a);
Tensor cos(Tensor &a);
Tensor exp(Tensor &a);
Tensor ln(Tensor &a);
*/

} // namespace cuda

} // namespace torch

#endif
