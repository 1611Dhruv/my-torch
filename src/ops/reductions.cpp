#include "mytorch/ops.h"
#include "mytorch/tensor.h"
#include <vector>

namespace torch {
namespace cpu {

template <typename scalar_t, typename ReduceFn>
Tensor reduce(const Tensor &a, Tensor &out, const std::vector<int64_t> reduce_dim, ReduceFn reduce_op, scalar_t init) {
  /*
   * Iterate over all the elements in the frame :)
   */
  std::vector<int64_t> keep_dim;

  int64_t j = 0;
  int64_t reduce_dim_sz = reduce_dim.size();
  for (int64_t i = 0; i < a.ndim(); i++) {
    if (j < reduce_dim_sz && i == reduce_dim[j]) {
      j++;
      continue;
    }
    keep_dim.push_back(i);
  }

  std::vector<int64_t> k_idx(keep_dim.size(), 0);

  int64_t out_off = 0;
  auto incr = [&a, &out, &out_off](const std::vector<int64_t> &dim, std::vector<int64_t> &idx, int64_t &off, bool &done,
                                   bool update_out = false) {
    int d = dim.size() - 1;
    idx[d]++;
    off += a.strides()[dim[d]];
    if (update_out) {
      out_off += out.strides()[dim[d]];
    }
    while (idx[d] == a.shape()[dim[d]]) {
      idx[d] = 0;
      off -= a.strides()[dim[d]] * a.shape()[dim[d]];
      if (update_out) {
        out_off -= out.shape()[dim[d]] * out.strides()[dim[d]];
      }
      d--;
      if (d < 0) {
        done = true;
        break;
      }
      idx[d]++;
      off += a.strides()[dim[d]];
      if (update_out) {
        out_off += out.strides()[dim[d]];
      }
    }
  };

  int64_t k_off = 0; // Offset to keep track of things
  bool k_done = false;
  while (!k_done) {
    std::vector<int64_t> r_idx(reduce_dim.size(), 0);
    int64_t r_off = 0;
    bool r_done = false;
    // Iterate over this indice
    scalar_t reduced = init;
    while (!r_done) {
      reduced = reduce_op(reduced, a.data_ptr<scalar_t>()[r_off + k_off]);
      incr(reduce_dim, r_idx, r_off, r_done, false);
    }
    out.data_ptr<scalar_t>()[out_off] = reduced;
    incr(keep_dim, k_idx, k_off, k_done, true);
  }

  return out;
}

Tensor sum(const Tensor &a, Tensor &out, const std::vector<int64_t> dims) {
  DISPATCH_OP(a.dtype(),
              [&]() { reduce(a, out, dims, [&](scalar_t x, scalar_t y) { return x + y; }, static_cast<scalar_t>(0)); });
  return out;
}

} // namespace cpu
}; // namespace torch
