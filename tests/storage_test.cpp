// Spec-by-tests for torch::Storage.
//
// These tests ARE the spec for Storage's ownership semantics. They will not
// link until storage.cpp implements the constructor + rule-of-five -- that
// "undefined reference to torch::Storage::..." link error is your TODO list.
//
// Run:  cmake -S . -B build && cmake --build build && ctest --test-dir build

#include "mytorch/storage.h"
#include <cstring>
#include <gtest/gtest.h>

using torch::Device;
using torch::Storage;

// --- construction -----------------------------------------------------------

TEST(StorageTest, ConstructSetsSizeDeviceAndBuffer) {
  Storage s(128);
  EXPECT_EQ(s.size(), 128u);
  EXPECT_EQ(s.device(), Device::CPU);
  EXPECT_NE(s.get(), nullptr);
  EXPECT_EQ(s.use_count(), 1u); // sole owner
}

TEST(StorageTest, DeviceIsRemembered) {
  Storage s(8, Device::CUDA);
  EXPECT_EQ(s.device(), Device::CUDA);
}

// --- data access ------------------------------------------------------------

TEST(StorageTest, ReadWriteRoundTrip) {
  Storage s(4);
  std::byte *p = s.get();
  for (int i = 0; i < 4; ++i)
    p[i] = static_cast<std::byte>(i + 1);
  const std::byte *r = s.get();
  for (int i = 0; i < 4; ++i)
    EXPECT_EQ(r[i], static_cast<std::byte>(i + 1));
}

// --- copy semantics: copies SHARE, they do not duplicate --------------------

TEST(StorageTest, CopySharesBufferAndBumpsRefcount) {
  Storage a(64);
  Storage b = a;                // copy ctor
  EXPECT_EQ(a.get(), b.get());  // same buffer, not a deep copy
  EXPECT_EQ(a.use_count(), 2u); // both report the shared count
  EXPECT_EQ(b.use_count(), 2u);
}

TEST(StorageTest, CopiesSeeEachOthersWrites) {
  Storage a(4);
  Storage b = a;
  a.get()[0] = static_cast<std::byte>(42);
  EXPECT_EQ(b.get()[0], static_cast<std::byte>(42)); // shared bytes
}

TEST(StorageTest, RefcountDropsWhenCopyDies) {
  Storage a(16);
  {
    Storage b = a;
    EXPECT_EQ(a.use_count(), 2u);
  } // b destructs here -> must decrement, must NOT free a's buffer
  EXPECT_EQ(a.use_count(), 1u);
  EXPECT_NE(a.get(), nullptr); // still valid after b is gone
}

// --- move semantics: transfer ownership, leave a valid husk -----------------

TEST(StorageTest, MoveTransfersOwnership) {
  Storage a(32);
  std::byte *buf = a.get();
  Storage b = std::move(a); // move ctor
  EXPECT_EQ(b.get(), buf);  // b now owns the original buffer
  EXPECT_EQ(b.use_count(), 1u);
  // moved-from 'a' must be a safe, destructible husk:
  EXPECT_EQ(a.get(), nullptr);
}

// --- self-assignment must not corrupt or free -------------------------------

TEST(StorageTest, SelfCopyAssignIsSafe) {
  Storage a(8);
  std::byte *buf = a.get();
  a = a; // must be a no-op, must NOT free buf
  EXPECT_EQ(a.get(), buf);
  EXPECT_EQ(a.use_count(), 1u);
}

// Assigning over a Storage should release its previous buffer's ownership:
// make a(=buf1) and b(=buf2), copy a third owner c=b so buf2.use_count==2,
// then do b = a; expect b shares buf1 and the old buf2 count drops to 1 (c).
TEST(StorageTest, CopyAssignReleasesOldBuffer) {
  Storage a(128);
  Storage b(128);
  Storage c = b;
  EXPECT_EQ(b.use_count(), 2u);
  b = a;
  EXPECT_EQ(c.use_count(), 1u);
}

// Self move-assign (a = std::move(a)) must leave 'a' valid and not double-free.
TEST(StorageTest, SelfMoveAssignIsSafe) {
  Storage a(128);
  a = std::move(a);
}

// Move-assign: b = std::move(a) transfers a's buffer to b, releases b's old
// buffer, and leaves a as a null husk.
TEST(StorageTest, MoveAssignTransfersAndReleases) {
  Storage a(128);
  std::byte *buf = a.get();
  Storage b = std::move(a);
  EXPECT_EQ(a.get(), nullptr);
  EXPECT_EQ(a.use_count(), 0u);
  EXPECT_EQ(b.get(), buf);
  EXPECT_EQ(b.use_count(), 1u);
}
