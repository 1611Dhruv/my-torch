#include "mytorch/ops/ops.h"
#include <stack>

namespace torch {
namespace cpu {

// Helper which will walk through one tensor applying op and saving in out
template <typename T, typename Op> static void unary_elementwise(const Tensor &a, Tensor &out, Op op) {
  if (!out.is_contiguous() || out.shape() != a.shape()) {
    throw std::runtime_error("out wasn't contiguous or had a different shape");
  }

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
  frames.push({0, 0, 0, 0});

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
static void binary_elementwise(const Tensor &a, const Tensor &b, Tensor &out, Op op) {
  if (!out.is_contiguous() || out.shape() != a.shape()) {
    throw std::runtime_error("out wasn't contiguous or had a different shape");
  }

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
Tensor add(const Tensor &a, const Tensor &b) {
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("add: tensors should have the same dtype, casting not supported yet");
  }

  if (a.shape() != b.shape()) {
    throw std::invalid_argument("add: tensors must have the same shape");
  }

  Tensor out = torch::Tensor::zeros(a.shape(), a.dtype(), a.device());
  DISPATCH_OP(a.dtype(),
              [&] { binary_elementwise<scalar_t>(a, b, out, [](scalar_t x, scalar_t y) { return x + y; }); });
  return out;
}

} // namespace cpu
} // namespace torch
