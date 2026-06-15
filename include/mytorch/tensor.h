#ifndef TENSOR_H
#define TENSOR_H

#include "mytorch/storage.h"
#include <cstdint>
#include <vector>

namespace torch {

enum class DType { Float32, Int32, UInt8 };
size_t itemsize(DType type);

class Tensor {
public:
  Tensor(std::vector<int64_t> shape, DType dtype, Device device);

  // View Ops
  Tensor reshape() const;
  Tensor transpose(int64_t dim1, int64_t dim2) const;

  // Access Ops
  Tensor operator[](int64_t i) const;
  template <typename T> T &at(std::initializer_list<int64_t> idx);
  template <typename T> T &item();
  template <typename T> T *data_ptr();

  // Metadata Accessors
  int64_t ndim() const;
  const std::vector<int64_t> shape() const;
  const std::vector<int64_t> strides() const;
  int64_t offset() const;
  DType dtype() const;
  Device device() const;

  // static factories
  static Tensor &zeros(std::vector<int64_t> shape, DType dtype, Device device);
  static Tensor &ones(std::vector<int64_t> shape, DType dtype, Device device);
  static Tensor &rand(std::vector<int64_t> shape, DType dtype, Device device);

  /*
   NOTE: Future maybe
  Tensor permute(std::vector<int64_t> dim_order) const;
  Tensor slice(int64_t dim, int64_t start, int64_t end) const;
  Tensor squeeze() const;
  Tensor unsqueeze() const;
  Tensor broadcast_to(std::vector<int64_t> shape) const;

  bool is_contiguous() const;
  int64_t nbytes() const;
 */
private:
  Storage _storage;
  std::vector<int64_t> _shape;
  std::vector<int64_t> _strides;
  int64_t _offset;
  DType _dtype;
  Device _device;
};
} // namespace torch
#endif
