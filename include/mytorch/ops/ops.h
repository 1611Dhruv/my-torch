#include "mytorch/tensor.h"

namespace torch {

namespace cpu {

Tensor add(const Tensor &a, const Tensor &b);
Tensor sub(const Tensor &a, const Tensor &b);
Tensor mult(const Tensor &a, const Tensor &b);
Tensor matmul(const Tensor &a, const Tensor &b);

/*
 NOTE: Future
Tensor sin(Tensor &a);
Tensor cos(Tensor &a);
Tensor exp(Tensor &a);
Tensor ln(Tensor &a);
*/

} // namespace cpu

} // namespace torch
