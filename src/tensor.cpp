#include "mytorch/tensor.h"
#include "mytorch/storage.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

namespace torch {
size_t itemsize(DType type) {
  switch (type) {
  case DType::Float32:
    return 4;
  case DType::Int32:
    return 4;
  case DType::UInt8:
    return 1;
  }
  throw std::invalid_argument("unknown DType");
}

static int64_t numel_of(const std::vector<int64_t> &shape) {
  int64_t count = 1;
  for (auto &dim : shape)
    count *= dim;
  return count;
}

std::vector<int64_t> strides_for(const std::vector<int64_t> &shape) {
  int64_t N = static_cast<int64_t>(shape.size());
  std::vector<int64_t> strides(N, 1);
  for (int64_t i = N - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Tensor Constructor

// Public allocating tensor
Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : _shape(shape),
      _dtype(dtype),
      _strides(strides_for(shape)),
      _storage(numel_of(shape) * itemsize(dtype), device),
      _offset(0) {}

// Private View only
Tensor::Tensor(Storage storage, std::vector<int64_t> shape, std::vector<int64_t> strides, int64_t offset, DType dtype)
    : _shape(shape),
      _dtype(dtype),
      _strides(strides),
      _offset(offset),
      _storage(storage) {}

// View Ops
Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
  if (numel_of(_shape) != numel_of(new_shape)) {
    throw std::invalid_argument("Reshape can't work as number of elements are different");
  }
  return Tensor(_storage, new_shape, strides_for(new_shape), 0, _dtype);
}

// Allows negative indexing too
Tensor Tensor::transpose(int64_t dim1, int64_t dim2) const {
  int ndim = _shape.size();
  if (dim1 < 0)
    dim1 = ndim - dim1;
  if (dim2 < 0)
    dim2 = ndim - dim2;

  if (dim1 < 0 || dim2 < 0 || dim1 >= ndim || dim2 >= ndim) {
    throw std::invalid_argument("The provided dimensions are not transposable");
  }

  if (dim1 == dim2)
    return *this;

  std::vector<int64_t> new_shape(_shape);
  std::vector<int64_t> new_strides(_strides);

  std::swap(new_shape[dim1], new_shape[dim2]);
  std::swap(new_strides[dim1], new_strides[dim2]);
  return Tensor(_storage, new_shape, new_strides, 0, _dtype);
}

static bool next_index(std::vector<int64_t> &idx, const std::vector<int64_t> &shape) {
  for (int64_t d = static_cast<int64_t>(idx.size()) - 1; d >= 0; --d) {
    if (++idx[d] < shape[d])
      return true;
    idx[d] = 0;
  }
  return false;
}

Tensor Tensor::contiguous() const {
  if (is_contiguous())
    return *this;

  Tensor out(_shape, _dtype, device());
  std::vector<int64_t> idx(_shape.size(), 0);
  size_t size = itemsize(_dtype);
  std::byte *dst = out._storage.get();
  const std::byte *src = _storage.get();

  for (int64_t d = 0; d < numel(); d++) {
    int64_t src_flat = _offset;
    for (int64_t k = 0; k < static_cast<int64_t>(_shape.size()); k++)
      src_flat += idx[k] * _strides[k];

    std::memcpy(dst + d * size, src + src_flat * size, size);
    next_index(idx, _shape);
  }

  return out;
}

// Access Ops
Tensor Tensor::operator[](int64_t i) const {
  int64_t N = static_cast<int64_t>(_shape.size());
  if (_shape.size() == 0) {
    throw std::invalid_argument("Cannot index further into a singleton Tensor");
  }

  std::vector<int64_t> new_shape(N - 1);
  std::vector<int64_t> new_strides(N - 1);
  for (int64_t i = 0; i < N; i++) {
    new_shape[i] = _shape[i + 1];
    new_strides[i] = _strides[i + 1];
  }

  return Tensor(_storage, new_shape, new_strides, i * _strides[0] * itemsize(_dtype), _dtype);
}

// Metadata Accessors
int64_t Tensor::numel() const { return numel_of(_shape); }

bool Tensor::is_contiguous() const {
  int64_t N = static_cast<int64_t>(_shape.size());
  int64_t expected = 1;
  for (int64_t i = N - 1; i >= 0; i--) {
    if (_shape[i] != 1 && expected != _strides[i])
      return false;
    expected *= _shape[i];
  }
  return true;
}

// Static factory
Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype, Device device) {
  Tensor t(shape, dtype, device);
  std::memset(t._storage.get(), 0, t._storage.size());
  return t;
}

Tensor Tensor::ones(std::vector<int64_t> shape, DType dtype, Device device) {
  Tensor t(shape, dtype, device);
  int64_t n = t.numel();
  switch (dtype) {
  case torch::DType::Float32: {
    float *p = t.data_ptr<float>();
    std::fill(p, p + n, 1.0f);
    break;
  }
  case torch::DType::Int32: {
    int32_t *p = t.data_ptr<int32_t>();
    std::fill(p, p + n, 1);
    break;
  }
  case torch::DType::UInt8: {
    uint8_t *p = t.data_ptr<uint8_t>();
    std::fill(p, p + n, uint8_t(1));
    break;
  }
  }
  return t;
}

Tensor Tensor::rand(std::vector<int64_t> shape, Device device) {
  Tensor t(shape, torch::DType::Float32, device);
  int64_t n = t.numel();
  float *p = t.data_ptr<float>();
  std::mt19937 gen(random());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int64_t i = 0; i < n; i++) {
    p[i] = dist(gen);
  }

  return t;
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  os << t.numel();
  return os;
}
}; // namespace torch
