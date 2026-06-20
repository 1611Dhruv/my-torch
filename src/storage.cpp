#include "mytorch/storage.h"

namespace torch {

Storage::Storage(size_t num_bytes, Device device)
    : _buffer(new std::byte[num_bytes]),
      _refcount(new size_t(1)),
      _size(num_bytes),
      _device(device) {}

std::byte *Storage::get() { return _buffer; }
const std::byte *Storage::get() const { return _buffer; }

size_t Storage::size() const { return _size; }
Device Storage::device() const { return _device; }
size_t Storage::use_count() const {
  if (!_refcount)
    return 0;
  return *_refcount;
}

void Storage::release() {
  if (_refcount) {
    *_refcount -= 1;
    if (*_refcount == 0) {
      delete[] _buffer;
      delete _refcount;
    }
  }
}

Storage::~Storage() { release(); }

// Copy shares the buffer so a Tensor's views point at the same bytes.
Storage::Storage(const Storage &other)
    : _buffer(other._buffer),
      _refcount(other._refcount),
      _size(other._size),
      _device(other._device) {
  *_refcount += 1;
}

Storage &Storage::operator=(const Storage &other) {
  if (this != &other) {
    release();

    _buffer = other._buffer;

    _size = other._size;
    _device = other._device;

    _refcount = other._refcount;
    *_refcount += 1;
  }
  return *this;
}

Storage::Storage(Storage &&other) noexcept
    : _buffer(other._buffer),
      _refcount(other._refcount),
      _size(other._size),
      _device(other._device) {
  other._buffer = nullptr;
  other._refcount = nullptr;
}

Storage &Storage::operator=(Storage &&other) noexcept {
  if (this != &other) {
    release();

    _buffer = other._buffer;
    _refcount = other._refcount;

    _size = other._size;
    _device = other._device;

    other._buffer = nullptr;
    other._refcount = nullptr;
  }
  return *this;
}

} // namespace torch
