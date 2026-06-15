#ifndef TENSOR_H
#define TENSOR_H

#include "mytorch/storage.h"
#include <cstdint>
#include <iostream>
#include <vector>

namespace torch {

enum class DType { Float32, Int32, UInt8 };
size_t itemsize(DType type);

class Tensor {
public:
  Tensor(std::vector<int64_t> shape, DType dtype = DType::Float32, Device device = CPU);

  // View Ops
  Tensor reshape(std::vector<int64_t> new_shape) const;
  Tensor transpose(int64_t dim1, int64_t dim2) const;

  // Access Ops
  Tensor operator[](int64_t i) const;
  template <typename T> T &at(std::initializer_list<int64_t> idx);
  template <typename T> T &item();
  template <typename T> T *data_ptr();

  // Metadata Accessors
  int64_t ndim() const { return static_cast<int64_t>(_shape.size()); };
  int64_t numel() const;
  inline const std::vector<int64_t> &shape() const { return _shape; };
  inline const std::vector<int64_t> &strides() const { return _strides; };
  inline int64_t offset() const { return _offset; };
  inline DType dtype() const { return _dtype; };
  inline Device device() const { return _storage.device(); };

  // static factories
  static Tensor zeros(std::vector<int64_t> shape, DType dtype, Device device);
  static Tensor ones(std::vector<int64_t> shape, DType dtype, Device device);
  static Tensor rand(std::vector<int64_t> shape, Device device);

  /*
   NOTE: Future maybe
  bool is_contiguous() const;
  Tensor contiguous() const;

  Tensor permute(std::vector<int64_t> dim_order) const;
  Tensor slice(int64_t dim, int64_t start, int64_t end) const;
  Tensor squeeze() const;
  Tensor unsqueeze() const;
  Tensor broadcast_to(std::vector<int64_t> shape) const;

  int64_t nbytes() const;
 */
private:
  Tensor(Storage storage, std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, DType dtype);

  std::vector<int64_t> _shape;
  DType _dtype;
  std::vector<int64_t> _strides;
  Storage _storage;
  int64_t _offset;
};

std::ostream &operator<<(std::ostream &os, const Tensor &t);

} // namespace torch
#endif
