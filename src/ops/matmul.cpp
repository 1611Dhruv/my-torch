#include "mytorch/ops.h"
#include <driver_types.h>
namespace torch {
namespace cpu {
Tensor matmul(const Tensor &a, const Tensor &b) {
  Tensor out({1, 2});
  return out;
}
} // namespace cpu
} // namespace torch
