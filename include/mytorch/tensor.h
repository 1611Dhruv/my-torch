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
  Tensor contiguous() const;

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
  bool is_contiguous() const;

  // static factories
  static Tensor zeros(std::vector<int64_t> shape, DType dtype, Device device);
  static Tensor ones(std::vector<int64_t> shape, DType dtype, Device device);
  static Tensor rand(std::vector<int64_t> shape, Device device);

  /*
   NOTE: Future maybe
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

template <typename T> constexpr DType dtype_of();
template <> constexpr DType dtype_of<float>() { return DType::Float32; }
template <> constexpr DType dtype_of<int32_t>() { return DType::Int32; }
template <> constexpr DType dtype_of<uint8_t>() { return DType::UInt8; }

template <typename T> T *typeptr(DType type, std::byte *buff) {
  if (dtype_of<T>() != type) {
    throw std::invalid_argument("Type cannot be casted");
  }
  return reinterpret_cast<T *>(buff);
}

template <typename T> T &Tensor::at(std::initializer_list<int64_t> idx) {
  int64_t N = static_cast<int64_t>(_shape.size());
  int64_t curr = 0;

  int elm_off = _offset;
  for (auto it = idx.begin(); it != idx.end(); it++) {
    if (curr >= N) {
      throw std::invalid_argument("Too many indexes provide, dim doesn't match");
    }
    elm_off += _strides[curr];
    curr++;
  }
  if (curr != N) {
    throw std::invalid_argument("Please index all the way to one value");
  }

  return *typeptr<T>(_dtype, _storage.get() + elm_off);
}

template <typename T> T &Tensor::item() {
  if (_shape.size() != 0) {
    throw std::invalid_argument("Please call item on a singleton tensor");
  }
  return *typeptr<T>(_dtype, _storage.get() + _offset);
}

template <typename T> T *Tensor::data_ptr() { return typeptr<T>(_dtype, _storage.get() + _offset); }

} // namespace torch
#endif
