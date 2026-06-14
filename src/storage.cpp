#include "mytorch/storage.h"

namespace torch {

Storage::Storage(size_t num_bytes, Device device) {}
std::byte *Storage::get() { return nullptr; }
const std::byte *Storage::get() const { return nullptr; }
size_t Storage::size() const { return 0; }
Device Storage::device() const { return Device::CPU; }
size_t Storage::use_count() const { return 0; }

Storage::~Storage() = default;
Storage::Storage(const Storage &other) = default;
Storage &Storage::operator=(const Storage &other) = default;
Storage::Storage(Storage &&other) noexcept {};
Storage &Storage::operator=(Storage &&other) noexcept { return *this; };

} // namespace torch
