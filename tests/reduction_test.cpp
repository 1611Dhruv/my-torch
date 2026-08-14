// Spec-by-tests for torch::sum (and the generic reduce underneath it), CPU and CUDA.
//
// The reduction walks two nested odometers -- one over the kept dims, one over
// the reduced dims -- so the cases that matter are the ones with MORE THAN ONE
// digit in either odometer. A {2,3} tensor reducing one dim only ever exercises
// a single-digit walk and will pass even when the carry logic is broken; {2,3,4}
// reducing one dim is the smallest shape that actually carries.
//
// Oracle: reference_sum() below re-implements the reduction on a flat host
// buffer with no Tensor machinery involved, so it cannot share a bug with either
// backend. Both devices are checked against that same oracle, which also forces
// CPU and CUDA to agree with each other.
//
// Run:  ctest --test-dir build -R Reduction
//       CUDA_LAUNCH_BLOCKING=1 ./build/tests/reduction_test   (to localize faults)
//       compute-sanitizer ./build/tests/reduction_test        (to catch OOB)

#include "mytorch/cuda_utils.h"
#include "mytorch/ops.h"
#include "mytorch/tensor.h"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <vector>

using torch::CPU;
using torch::CUDA;
using torch::Device;
using torch::DType;
using torch::Tensor;
using Shape = std::vector<int64_t>;

// --- helpers ---------------------------------------------------------------

static Shape contiguous_strides(const Shape &s) {
  Shape st(s.size(), 1);
  for (int64_t i = static_cast<int64_t>(s.size()) - 2; i >= 0; --i)
    st[i] = st[i + 1] * s[i + 1];
  return st;
}

static int64_t product(const Shape &s) {
  return std::accumulate(s.begin(), s.end(), int64_t{1}, std::multiplies<int64_t>());
}

// Independent reference implementation: sum over `dims` with keepdim=true,
// operating on a flat row-major buffer. Deliberately shares no code with
// src/ops/reductions.cpp or reductions.cu.
static std::vector<float> reference_sum(const Shape &shape, const std::vector<float> &vals, const Shape &dims) {
  int64_t nd = shape.size();
  Shape oshape = shape;
  for (int64_t d : dims)
    oshape[d] = 1;

  Shape ostr = contiguous_strides(oshape);
  std::vector<float> out(product(oshape), 0.0f);

  Shape idx(nd, 0);
  int64_t n = product(shape);
  for (int64_t flat = 0; flat < n; ++flat) {
    int64_t oo = 0;
    for (int64_t k = 0; k < nd; ++k)
      oo += (oshape[k] == 1 ? 0 : idx[k]) * ostr[k];
    out[oo] += vals[flat];

    for (int64_t k = nd - 1; k >= 0; --k) {
      if (++idx[k] < shape[k])
        break;
      idx[k] = 0;
    }
  }
  return out;
}

// Ascending values, so any index mix-up changes the result (a constant fill
// would let a wrong traversal still produce the right totals).
static std::vector<float> seq(int64_t n) {
  std::vector<float> v(n);
  std::iota(v.begin(), v.end(), 0.0f);
  return v;
}

static Tensor make(const Shape &shape, const std::vector<float> &vals, Device dev) {
  Tensor t(shape, DType::Float32, dev);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  if (dev == CUDA)
    CUDA_CHECK(cudaMemcpy(t.data_ptr<float>(), vals.data(), vals.size() * sizeof(float), cudaMemcpyHostToDevice));
  else
    std::copy(vals.begin(), vals.end(), t.data_ptr<float>());
  return t;
}

// Flat contents of a contiguous tensor, pulled back to host if it's on device.
static std::vector<float> host(const Tensor &t) {
  EXPECT_TRUE(t.is_contiguous());
  std::vector<float> h(t.numel());
  if (t.device() == CUDA)
    CUDA_CHECK(cudaMemcpy(h.data(), t.data_ptr<float>(), h.size() * sizeof(float), cudaMemcpyDeviceToHost));
  else
    std::copy(t.data_ptr<float>(), t.data_ptr<float>() + t.numel(), h.begin());
  return h;
}

static bool cuda_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// --- device-parameterized suite ---------------------------------------------
// Every case below runs on BOTH backends against the same oracle, so the two
// implementations are also forced to agree with each other.

class ReductionTest : public ::testing::TestWithParam<Device> {
protected:
  void SetUp() override {
    if (GetParam() == CUDA && !cuda_available())
      GTEST_SKIP() << "no CUDA device available";
  }
  Device dev() const { return GetParam(); }

  void expect_matches_reference(const Shape &shape, const Shape &dims) {
    std::vector<float> vals = seq(product(shape));
    Tensor a = make(shape, vals, dev());
    Tensor got = torch::sum(a, dims);

    Shape norm;
    for (int64_t d : dims)
      norm.push_back(d < 0 ? d + static_cast<int64_t>(shape.size()) : d);

    Shape want_shape = shape;
    for (int64_t d : norm)
      want_shape[d] = 1;
    ASSERT_EQ(got.shape(), want_shape);

    std::vector<float> want = reference_sum(shape, vals, norm);
    std::vector<float> g = host(got);
    ASSERT_EQ(g.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i)
      EXPECT_FLOAT_EQ(g[i], want[i]) << "at flat index " << i;
  }
};

INSTANTIATE_TEST_SUITE_P(Devices, ReductionTest, ::testing::Values(CPU, CUDA),
                         [](const ::testing::TestParamInfo<Device> &info) {
                           return info.param == CPU ? "Cpu" : "Cuda";
                         });

// --- one dim at a time, rank 3 ---------------------------------------------
// {2,3,4} is the smallest shape where reducing a single dim leaves a TWO-digit
// keep odometer, which is what forces a carry.

TEST_P(ReductionTest, SumDim0OfRank3) { expect_matches_reference({2, 3, 4}, {0}); }
TEST_P(ReductionTest, SumDim1OfRank3) { expect_matches_reference({2, 3, 4}, {1}); }
TEST_P(ReductionTest, SumDim2OfRank3) { expect_matches_reference({2, 3, 4}, {2}); }

// Non-divisible, non-power-of-two extents so nothing accidentally lines up, and
// the element count straddles the 256-thread block size.
TEST_P(ReductionTest, SumMiddleDimOfRank4) { expect_matches_reference({3, 5, 7, 2}, {1}); }

// More outputs than one block's worth -> exercises blockIdx in the CUDA path.
TEST_P(ReductionTest, SumWithMoreOutputsThanOneBlock) { expect_matches_reference({40, 30, 3}, {2}); }

// --- several dims at once ---------------------------------------------------
// Two reduce digits AND two keep digits: both odometers have to carry.

TEST_P(ReductionTest, SumTwoNonAdjacentDims) { expect_matches_reference({2, 3, 4, 5}, {0, 2}); }
TEST_P(ReductionTest, SumTwoAdjacentDims) { expect_matches_reference({2, 3, 4, 5}, {1, 2}); }
TEST_P(ReductionTest, SumThreeOfFourDims) { expect_matches_reference({2, 3, 4, 5}, {0, 1, 3}); }

// --- dim-list normalization -------------------------------------------------

TEST_P(ReductionTest, NegativeDimMatchesPositive) {
  std::vector<float> vals = seq(24);
  Tensor a = make({2, 3, 4}, vals, dev());
  Tensor b = make({2, 3, 4}, vals, dev());
  EXPECT_EQ(host(torch::sum(a, {-1})), host(torch::sum(b, {2})));
}

TEST_P(ReductionTest, DuplicateDimsAreIdempotent) {
  // {1,1} must not double-count, and must not divide by the wrong count later
  // when mean() lands.
  std::vector<float> vals = seq(24);
  Tensor a = make({2, 3, 4}, vals, dev());
  Tensor b = make({2, 3, 4}, vals, dev());
  EXPECT_EQ(host(torch::sum(a, {1, 1})), host(torch::sum(b, {1})));
}

TEST_P(ReductionTest, UnsortedDimsMatchSorted) {
  // Both reduce implementations derive their keep dims by walking i against
  // reduce_dim in order, so an unsorted list silently keeps the wrong axes if
  // normalization doesn't sort before handing off.
  std::vector<float> vals = seq(120);
  Tensor a = make({2, 3, 4, 5}, vals, dev());
  Tensor b = make({2, 3, 4, 5}, vals, dev());
  EXPECT_EQ(host(torch::sum(a, {2, 0})), host(torch::sum(b, {0, 2})));
}

TEST_P(ReductionTest, MixedNegativeAndPositiveDims) {
  // -1 normalizes to 3, which must sort BEFORE being handed to the backend.
  std::vector<float> vals = seq(120);
  Tensor a = make({2, 3, 4, 5}, vals, dev());
  Tensor b = make({2, 3, 4, 5}, vals, dev());
  EXPECT_EQ(host(torch::sum(a, {-1, 0})), host(torch::sum(b, {0, 3})));
}

// --- shape of the result ----------------------------------------------------

TEST_P(ReductionTest, KeepDimLeavesSizeOneAxis) {
  Tensor a = make({2, 3, 4}, seq(24), dev());
  EXPECT_EQ(torch::sum(a, {1}).shape(), Shape({2, 1, 4}));
  EXPECT_EQ(torch::sum(a, {0, 2}).shape(), Shape({1, 3, 1}));
}

TEST_P(ReductionTest, OutputIsFreshAndContiguous) {
  Tensor a = make({2, 3, 4}, seq(24), dev());
  Tensor out = torch::sum(a, {1});
  EXPECT_TRUE(out.is_contiguous());
  EXPECT_NE(out.data_ptr<float>(), a.data_ptr<float>());
}

TEST_P(ReductionTest, SumDoesNotMutateInput) {
  std::vector<float> vals = seq(24);
  Tensor a = make({2, 3, 4}, vals, dev());
  torch::sum(a, {1});
  EXPECT_EQ(host(a), vals);
}

// --- strided / offset inputs ------------------------------------------------
// Both reduces index `a` by its own strides, so a view must give the same answer
// as the materialized copy.

TEST_P(ReductionTest, SumOfTransposedViewMatchesContiguous) {
  Tensor a = make({2, 3, 4}, seq(24), dev());
  Tensor view = a.transpose(0, 2); // {4,3,2}, non-contiguous
  ASSERT_FALSE(view.is_contiguous());

  Tensor materialized = view.contiguous();
  for (int64_t d = 0; d < 3; ++d)
    EXPECT_EQ(host(torch::sum(view, {d})), host(torch::sum(materialized, {d}))) << "reducing dim " << d;
}

TEST_P(ReductionTest, SumOfOffsetViewMatchesContiguous) {
  // a[1] moves _offset; data_ptr() folds it in, so the reduce must not add it
  // a second time.
  Tensor a = make({3, 4, 5}, seq(60), dev());
  Tensor view = a[1].transpose(0, 1); // {5,4}, offset 20, non-contiguous
  ASSERT_FALSE(view.is_contiguous());

  Tensor materialized = view.contiguous();
  EXPECT_EQ(host(torch::sum(view, {0})), host(torch::sum(materialized, {0})));
  EXPECT_EQ(host(torch::sum(view, {1})), host(torch::sum(materialized, {1})));
}

// --- the duality with broadcast --------------------------------------------

TEST_P(ReductionTest, BroadcastThenSumScalesByExpansionFactor) {
  // broadcast_to stride-0-expands an axis; summing that axis back must return
  // the original values times the expansion factor. If broadcast and reduce
  // disagree about which axis is which, this fails loudly. Note the stride-0
  // dim means many threads read the SAME address -- worth confirming on GPU.
  std::vector<float> vals = seq(12);
  Tensor a = make({1, 3, 4}, vals, dev());
  Tensor wide = a.broadcast_to({5, 3, 4});
  ASSERT_FALSE(wide.is_contiguous());

  Tensor out = torch::sum(wide, {0});
  ASSERT_EQ(out.shape(), Shape({1, 3, 4}));
  std::vector<float> got = host(out);
  for (size_t i = 0; i < vals.size(); ++i)
    EXPECT_FLOAT_EQ(got[i], vals[i] * 5.0f);
}

// --- reducing everything ----------------------------------------------------
// Both leave the keep odometer empty, which used to walk off the front of the
// index array on CPU. The CUDA kernel handles it via its bounds guard: nkeep==0
// means the unravel loop never runs, so only thread 0 survives with koff 0.

TEST_P(ReductionTest, ReduceEveryDimProducesScalar) {
  Tensor a = make({2, 3}, seq(6), dev());
  Tensor out = torch::sum(a, {0, 1});
  EXPECT_EQ(out.shape(), Shape({1, 1}));
  EXPECT_FLOAT_EQ(host(out)[0], 15.0f); // 0+1+2+3+4+5
}

TEST_P(ReductionTest, EmptyDimListReducesEverything) {
  // Pins the semantics chosen for `sum(x)` with no dims: reduce all axes, same
  // as naming every axis explicitly. (PyTorch made the opposite call for a
  // while and it was a well-known footgun, so this is worth stating loudly.)
  Tensor a = make({2, 3}, seq(6), dev());
  Tensor out = torch::sum(a, {});
  EXPECT_EQ(out.shape(), Shape({1, 1}));
  EXPECT_FLOAT_EQ(host(out)[0], 15.0f);

  Tensor b = make({2, 3, 4}, seq(24), dev());
  EXPECT_EQ(host(torch::sum(b, {})), host(torch::sum(b, {0, 1, 2})));
}

// --- CPU-only: things that don't vary by device -----------------------------

TEST_P(ReductionTest, KeepDimFalseDropsTheReducedAxes) {
  Tensor a = make({2, 3, 4}, seq(24), dev());

  Tensor one = torch::sum(a, {1}, false);
  EXPECT_EQ(one.shape(), Shape({2, 4}));
  // Same numbers as the keepdim version, just a different shape.
  EXPECT_EQ(host(one), host(torch::sum(a, {1}, true)));

  // Dropping two axes must not shift the index of the second one mid-squeeze.
  Tensor two = torch::sum(a, {0, 2}, false);
  EXPECT_EQ(two.shape(), Shape({3}));
  EXPECT_EQ(host(two), host(torch::sum(a, {0, 2}, true)));
}

TEST(ReductionCpuOnlyTest, DimListArgumentIsNotMutated) {
  // torch::sum takes dims by value and normalizes/sorts in place; the caller's
  // vector must be untouched.
  Tensor a = make({2, 3, 4}, seq(24), CPU);
  std::vector<int64_t> dims = {-1, 0};
  torch::sum(a, dims);
  EXPECT_EQ(dims, std::vector<int64_t>({-1, 0}));
}

TEST(ReductionCpuOnlyTest, SumOfInt32Tensor) {
  // Float32 and Int32 take different DISPATCH_OP branches. A `scalar_t` that
  // gets deduced from an int literal rather than the dtype passes the Int32 case
  // and throws on the Float32 one, so both need covering.
  Tensor a({2, 3}, DType::Int32, CPU);
  for (int i = 0; i < 6; ++i)
    a.data_ptr<int32_t>()[i] = i;

  Tensor out = torch::sum(a, {1});
  ASSERT_EQ(out.shape(), Shape({2, 1}));
  EXPECT_EQ(out.data_ptr<int32_t>()[0], 0 + 1 + 2);
  EXPECT_EQ(out.data_ptr<int32_t>()[1], 3 + 4 + 5);
}

TEST(ReductionCpuOnlyTest, OutOfRangeDimThrows) {
  // Without an upper-bound check, target_shape[dim] writes past the end.
  Tensor a = make({2, 3}, seq(6), CPU);
  EXPECT_THROW(torch::sum(a, {7}), std::invalid_argument);
}
