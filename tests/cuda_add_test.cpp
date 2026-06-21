// Spec-by-tests for torch::cuda::add (the GPU elementwise add kernel).
//
// These actually launch the kernel, so they require a CUDA device at runtime.
// Data is staged host->device with cudaMemcpy (there's no .to(device) yet),
// the kernel runs, and the result is copied back for comparison.
//
// Run:  ctest --test-dir build -R Cuda

#include "mytorch/cuda_utils.h"
#include "mytorch/ops/ops.h"
#include "mytorch/tensor.h"
#include <gtest/gtest.h>
#include <vector>

using torch::DType;
using torch::Tensor;
using Device = torch::Device;

// n is intentionally NOT a multiple of the 256-thread block size, so the last
// block is ragged and the kernel's `i >= n` bounds guard is exercised.
TEST(CudaAddTest, AddsElementwise) {
  constexpr int64_t n = 1000;

  std::vector<float> ha(n), hb(n);
  for (int64_t i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i + 1);
  }

  Tensor a({n}, DType::Float32, Device::CUDA);
  Tensor b({n}, DType::Float32, Device::CUDA);
  CUDA_CHECK(cudaMemcpy(a.data_ptr<float>(), ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.data_ptr<float>(), hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  Tensor c = torch::cuda::add(a, b);

  std::vector<float> hc(n);
  CUDA_CHECK(cudaMemcpy(hc.data(), c.data_ptr<float>(), n * sizeof(float), cudaMemcpyDeviceToHost));

  for (int64_t i = 0; i < n; ++i)
    EXPECT_FLOAT_EQ(hc[i], ha[i] + hb[i]);
}

// Mismatched shapes must throw, not launch.
TEST(CudaAddTest, ThrowsOnShapeMismatch) {
  Tensor a({4}, DType::Float32, Device::CUDA);
  Tensor b({5}, DType::Float32, Device::CUDA);
  EXPECT_THROW(torch::cuda::add(a, b), std::invalid_argument);
}
