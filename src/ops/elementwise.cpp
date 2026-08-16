#include "mytorch/ops.h"
#include <cassert>
#include <cmath>
#include <stack>

namespace torch {
namespace cpu {

// Helper which will walk through one tensor applying op and saving in out
template <typename T, typename Op>
static void unary_elementwise(const Tensor &a, Tensor &out, Op op) {
  assert(out.is_contiguous() && out.shape() == a.shape());

  const T *a_data = a.data_ptr<T>();
  T *o_data = out.data_ptr<T>();

  // if both are contiguous that means its a simple op
  // Go through the entire frame and just add
  if (a.is_contiguous()) {
    auto N = a.numel();
    for (int64_t i = 0; i < N; i++) {
      o_data[i] = op(a_data[i]);
    }
    return;
  }

  const auto shape = a.shape();
  int64_t N_DIM = shape.size();
  struct Frame {
    int64_t dim;
    int64_t a_off;
    int64_t o_off;
  };

  std::stack<Frame> frames;
  frames.push({0, 0, 0});

  while (!frames.empty()) {
    auto f = frames.top();
    frames.pop();
    if (f.dim == N_DIM) {
      o_data[f.o_off] = op(a_data[f.a_off]);
    } else {
      // We go add this dims kids
      for (int i = 0; i < shape[f.dim]; i++) {
        frames.push({
            f.dim + 1,
            f.a_off + i * a.strides()[f.dim],
            f.o_off + i * out.strides()[f.dim],
        });
      }
    }
  }
}

// Helper which will walk through the two tensors applying op and saving  in out
template <typename T, typename Op>
static void binary_elementwise(const Tensor &a, const Tensor &b, Tensor &out,
                               Op op) {
  assert(out.is_contiguous() && out.shape() == a.shape());

  const T *a_data = a.data_ptr<T>();
  const T *b_data = b.data_ptr<T>();
  T *o_data = out.data_ptr<T>();

  // if both are contiguous that means its a simple op
  // Go through the entire frame and just add
  if (a.is_contiguous() && b.is_contiguous()) {
    auto N = a.numel();
    for (int64_t i = 0; i < N; i++) {
      o_data[i] = op(a_data[i], b_data[i]);
    }
    return;
  }

  const auto shape = a.shape();
  int64_t N_DIM = shape.size();
  struct Frame {
    int64_t dim;
    int64_t a_off;
    int64_t b_off;
    int64_t o_off;
  };

  std::stack<Frame> frames;
  frames.push({0, 0, 0, 0});

  while (!frames.empty()) {
    auto f = frames.top();
    frames.pop();
    if (f.dim == N_DIM) {
      o_data[f.o_off] = op(a_data[f.a_off], b_data[f.b_off]);
    } else {
      // We go add this dims kids
      for (int i = 0; i < shape[f.dim]; i++) {
        frames.push({
            f.dim + 1,
            f.a_off + i * a.strides()[f.dim],
            f.b_off + i * b.strides()[f.dim],
            f.o_off + i * out.strides()[f.dim],
        });
      }
    }
  }
}

// Add operation for a basic CPU based tensor math
// We will assume that we are working with scalar_t type
// We dont work with a and b having different tensor types (for now ig? )
Tensor add(const Tensor &a, const Tensor &b, Tensor &out) {
  assert(a.dtype() == b.dtype());
  assert(a.shape() == b.shape());

  DISPATCH_OP(a.dtype(), [&] {
    binary_elementwise<scalar_t>(a, b, out,
                                 [](scalar_t x, scalar_t y) { return x + y; });
  });
  return out;
}

Tensor sub(const Tensor &a, const Tensor &b, Tensor &out) {
  assert(a.dtype() == b.dtype());
  assert(a.shape() == b.shape());

  DISPATCH_OP(a.dtype(), [&] {
    binary_elementwise<scalar_t>(a, b, out,
                                 [](scalar_t x, scalar_t y) { return x - y; });
  });
  return out;
}

Tensor mult(const Tensor &a, const Tensor &b, Tensor &out) {
  assert(a.dtype() == b.dtype());
  assert(a.shape() == b.shape());

  DISPATCH_OP(a.dtype(), [&] {
    binary_elementwise<scalar_t>(a, b, out,
                                 [](scalar_t x, scalar_t y) { return x * y; });
  });
  return out;
}

Tensor div(const Tensor &a, const Tensor &b, Tensor &out) {
  assert(a.dtype() == b.dtype());
  assert(a.shape() == b.shape());

  DISPATCH_OP(a.dtype(), [&] {
    binary_elementwise<scalar_t>(a, b, out, [](scalar_t x, scalar_t y) {
      if (y == 0) {
        throw std::invalid_argument("div called with y = 0");
      }
      return x / y;
    });
  });
  return out;
}

Tensor exp(const Tensor &a, Tensor &out) {
  DISPATCH_OP(a.dtype(), [&] {
    if constexpr (!std::is_floating_point_v<scalar_t>) {
      throw std::invalid_argument("Called elementwise_unary_wrapper which is "
                                  "float only on a non float type");
    } else
      unary_elementwise<scalar_t>(a, out,
                                  [](scalar_t x) { return std::exp(x); });
  });
  return out;
}

Tensor ln(const Tensor &a, Tensor &out) {
  DISPATCH_OP(a.dtype(), [&] {
    if constexpr (!std::is_floating_point_v<scalar_t>) {
      throw std::invalid_argument("Called elementwise_unary_wrapper which is "
                                  "float only on a non float type");
    } else {
      unary_elementwise<scalar_t>(a, out,
                                  [](scalar_t x) { return std::log(x); });
    }
  });
  return out;
}

Tensor sqrt(const Tensor &a, Tensor &out) {
  DISPATCH_OP(a.dtype(), [&] {
    if constexpr (!std::is_floating_point_v<scalar_t>) {
      throw std::invalid_argument("Called elementwise_unary_wrapper which is "
                                  "float only on a non float type");
    } else {
      unary_elementwise<scalar_t>(a, out,
                                  [](scalar_t x) { return std::sqrt(x); });
    }
  });
  return out;
}

Tensor scale(const Tensor &a, Tensor &out, double s) {
  DISPATCH_OP(a.dtype(), [&] {
    unary_elementwise<scalar_t>(
        a, out, [s](scalar_t x) { return x * static_cast<scalar_t>(s); });
  });
  return out;
}
Tensor sin(const Tensor &a, Tensor &out) {
  DISPATCH_OP(a.dtype(), [&] {
    if constexpr (!std::is_floating_point_v<scalar_t>) {
      throw std::invalid_argument("Called elementwise_unary_wrapper which is "
                                  "float only on a non float type");
    } else
      unary_elementwise<scalar_t>(a, out,
                                  [](scalar_t x) { return std::sin(x); });
  });
  return out;
}

Tensor cos(const Tensor &a, Tensor &out) {
  DISPATCH_OP(a.dtype(), [&] {
    if constexpr (!std::is_floating_point_v<scalar_t>) {
      throw std::invalid_argument("Called elementwise_unary_wrapper which is "
                                  "float only on a non float type");
    } else
      unary_elementwise<scalar_t>(a, out,
                                  [](scalar_t x) { return std::cos(x); });
  });
  return out;
}

Tensor neg(const Tensor &a, Tensor &out) {
  DISPATCH_OP(a.dtype(), [&] {
    unary_elementwise<scalar_t>(a, out, [](scalar_t x) { return -x; });
  })
  return out;
}

// Special Ops
Tensor relu(const Tensor &a, Tensor &out) {
  DISPATCH_OP(a.dtype(), [&] {
    unary_elementwise<scalar_t>(a, out,
                                [](auto x) { return (x + std::abs(x)) / 2; });
  });
  return out;
}

Tensor relu_back(const Tensor &a, const Tensor &g, Tensor &out) {
  DISPATCH_OP(a.dtype(), [&] {
    binary_elementwise<scalar_t>(a, g, out, [](auto x, auto b) {
      return x > 0 ? b : static_cast<scalar_t>(0);
    });
  });
  return out;
}

Tensor cast(const Tensor &a, Tensor &out) {
  DISPATCH_OP_AS(a.dtype(), src_t, [&]() {
    DISPATCH_OP_AS(out.dtype(), dest_t, [&] {
      const src_t *a_ptr = a.data_ptr<src_t>();
      dest_t *out_ptr = out.data_ptr<dest_t>();
      int64_t N = a.numel();
      for (int64_t i = 0; i < N; i++) {
        int i_off = 0;
        if (!a.is_contiguous()) {
          int id = i;
          for (int j = a.ndim() - 1; j >= 0; j--) {
            i_off += (id % a.shape()[j]) * a.strides()[j];
            id /= a.shape()[j];
          }
        } else {
          i_off = i;
        }
        out_ptr[i] = static_cast<dest_t>(a_ptr[i_off]);
      }
    });
  });
  return out;
}

} // namespace cpu
} // namespace torch
