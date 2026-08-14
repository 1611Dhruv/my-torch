#include "mytorch/tensor.h"
#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include "mytorch/storage.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
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

  // Non contiguous needs a new buffer before reshaping
  if (!is_contiguous())
    return this->contiguous().reshape(new_shape);

  return Tensor(_storage, new_shape, strides_for(new_shape), _offset, _dtype);
}

// Allows negative indexing too
Tensor Tensor::transpose(int64_t dim1, int64_t dim2) const {
  int64_t ndim = _shape.size();
  if (dim1 < 0)
    dim1 = ndim + dim1;
  if (dim2 < 0)
    dim2 = ndim + dim2;

  if (dim1 < 0 || dim2 < 0 || dim1 >= ndim || dim2 >= ndim) {
    throw std::invalid_argument("The provided dimensions are not transposable");
  }

  if (dim1 == dim2)
    return *this;

  std::vector<int64_t> new_shape(_shape);
  std::vector<int64_t> new_strides(_strides);

  std::swap(new_shape[dim1], new_shape[dim2]);
  std::swap(new_strides[dim1], new_strides[dim2]);
  return Tensor(_storage, new_shape, new_strides, _offset, _dtype);
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

  if (device() == CUDA) {
    Tensor gathered(_shape, _dtype, device());
    return cuda::contiguous(*this, gathered);
  }

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

Tensor Tensor::squeeze(std::vector<int64_t> dims) const {
  bool all = false;
  if (dims.empty()) {
    all = true;
  }
  std::sort(dims.begin(), dims.end());
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());

  std::vector<int64_t> new_shape;
  std::vector<int64_t> new_stride;

  int k = 0;
  for (int64_t i = 0; i < ndim(); i++) {
    if (_shape[i] == 1 && (all || k < dims.size() && i == dims[k])) {
      k++;
      continue;
    }
    new_shape.push_back(_shape[i]);
    new_stride.push_back(_strides[i]);
  }
  // Still make 1 by 1
  if (new_shape.empty()) {
    new_shape.push_back(1);
    new_stride.push_back(1);
  }

  return Tensor(_storage, new_shape, new_stride, _offset, _dtype);
}

Tensor Tensor::broadcast_to(std::vector<int64_t> target) const {
  int64_t target_dim = target.size();
  int64_t src_dim = _shape.size();

  if (target_dim < src_dim) {
    throw std::invalid_argument("Cannot broadcast from a higher to lower dim");
  }

  int i = src_dim - 1;
  int j = target_dim - 1;

  std::vector<int64_t> new_strides(target_dim, 0);
  while (i >= 0 && j >= 0) {
    if (target[j] == -1) {
      target[j] = _shape[i];
      new_strides[j] = _strides[i];
    } else if (target[j] < 1) {
      throw std::invalid_argument("Could not broadcast as target dimension is < 1 & not -1");
    } else if (_shape[i] == target[j]) {
      new_strides[j] = _strides[i];
    } else if (_shape[i] == 1) {
      new_strides[j] = 0;
    } else {
      throw std::invalid_argument("Could not broadcast as the dimensions dont match");
    }
    i--;
    j--;
  }

  while (j >= 0) {
    if (target[j--] <= 0) {
      throw std::invalid_argument("Could not broad cast with starting target");
    }
  }

  return Tensor(_storage, target, new_strides, _offset, _dtype);
}

// Access Ops
Tensor Tensor::operator[](int64_t i) const {
  int64_t N = static_cast<int64_t>(_shape.size());
  if (_shape.size() == 0) {
    throw std::invalid_argument("Cannot index further into a singleton Tensor");
  }
  if (i < 0 || i >= _shape[0]) {
    throw std::invalid_argument("Index out of range");
  }

  std::vector<int64_t> new_shape(N - 1);
  std::vector<int64_t> new_strides(N - 1);
  for (int64_t j = 0; j < N - 1; j++) {
    new_shape[j] = _shape[j + 1];
    new_strides[j] = _strides[j + 1];
  }

  return Tensor(_storage, new_shape, new_strides, _offset + i * _strides[0], _dtype);
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
  if (device == CUDA) {
    CUDA_CHECK(cudaMemset(t._storage.get(), 0, t._storage.size()));
  } else {
    std::memset(t._storage.get(), 0, t._storage.size());
  }
  return t;
}

Tensor Tensor::zeros_like(const Tensor &other) {
  Tensor t(other.shape(), other.dtype(), other.device());
  if (other.device() == CUDA) {
    CUDA_CHECK(cudaMemset(t._storage.get(), 0, t._storage.size()));
  } else {
    std::memset(t._storage.get(), 0, t._storage.size());
  }
  return t;
}
Tensor Tensor::ones_like(const Tensor &other) { return Tensor::ones(other.shape(), other.dtype(), other.device()); }

Tensor Tensor::ones(std::vector<int64_t> shape, DType dtype, Device device) {
  Tensor t(shape, dtype, device);
  int64_t n = t.numel();

  int64_t nb = t._storage.size();
  void *host_cp = (device == CPU) ? t._storage.get() : malloc(nb);

  switch (dtype) {
  case torch::DType::Float32: {
    float *p = (float *)host_cp;
    std::fill(p, p + n, 1.0f);
    break;
  }
  case torch::DType::Int32: {
    int32_t *p = (int32_t *)host_cp;
    std::fill(p, p + n, 1);
    break;
  }
  case torch::DType::UInt8: {
    uint8_t *p = (uint8_t *)host_cp;
    std::fill(p, p + n, uint8_t(1));
    break;
  }
  }

  if (device == CUDA) {
    CUDA_CHECK(cudaMemcpy(t._storage.get(), host_cp, nb, cudaMemcpyHostToDevice));
    free(host_cp);
  }
  return t;
}

static std::mt19937 &rand_generator() {
  static std::mt19937 gen(std::random_device{}());
  return gen;
}

void manual_seed(uint64_t seed) { rand_generator().seed(seed); }

Tensor Tensor::rand(std::vector<int64_t> shape, Device device) {
  Tensor t(shape, torch::DType::Float32, device);
  int64_t n = t.numel();
  int64_t nb = t._storage.size();
  void *host = (device == CPU) ? t._storage.get() : malloc(nb);

  float *p = static_cast<float *>(host);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int64_t i = 0; i < n; i++) {
    p[i] = dist(rand_generator());
  }

  if (device == CUDA) {
    CUDA_CHECK(cudaMemcpy(t._storage.get(), host, nb, cudaMemcpyHostToDevice));
    free(host);
  }
  return t;
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  os << t.numel();
  return os;
}
} // namespace torch
