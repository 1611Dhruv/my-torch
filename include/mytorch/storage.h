#ifndef STORAGE_H
#define STORAGE_H

#include <cstddef>
#include <cstdio>

namespace torch {
enum Device { CPU, CUDA };

// Storage handle
class Storage {
public:
  explicit Storage(size_t num_bytes, Device device = CPU);
  const std::byte *get() const;
  size_t size() const;
  Device device() const;

  // NOTE: Move is noexcept because move shouldn't throw error
  ~Storage(); // destructor should decrement refcount and free buffer if 0
  Storage(const Storage &other); // copy constructor increment refcount
  Storage &operator=(const Storage &other); // copy op
  Storage(const Storage &&other) noexcept;  // move constructor change ownership
                                            // and clear the other
  Storage &operator=(const Storage &&other) noexcept; // move op

private:
  // Shared buffer and refcounter for this storage handle
  std::byte *_buffer;
  std::size_t *_refcount;
  size_t _size;
  Device _device;
};

} // namespace torch

#endif // STORAGE_H
