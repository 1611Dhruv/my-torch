// Milestone test: the shift task.
//
// The task: input is {B,T,D} of random floats; target[b,t,:] == input[b,t-1,:],
// with position 0 targeting zeros. Train on one fixed batch until the loss is
// ~0. This is "overfit a single batch" -- we are not testing generalization,
// we are testing that the machinery can express and learn the function at all.
// If a model cannot fit a handful of examples it has seen thousands of times,
// the bug is in the implementation, not in the learning.
//
// Why THIS task: a position-wise network (Linear -> ReLU -> Linear applied
// independently at each t) provably cannot solve it, because position t's
// target is position t-1's *input*, which a position-wise function never sees.
// Only attention can route information across positions. So convergence is
// direct evidence that attention works -- the same logic XOR gave us for the
// nonlinearity. ControlPositionwiseModelPlateaus is what proves the task
// actually discriminates; without it, easy convergence might just mean the
// target leaked into the input.
//
// Shift RIGHT (copy the previous token), not left: the needed information is
// in the past, which a causal mask permits. Shifting left would be impossible
// by construction and would look identical to a broken model.
//
// NOT YET REGISTERED. Add to tests/CMakeLists.txt once nn::Transformer exists:
//     add_executable(transformer_test transformer_test.cpp)
//     target_link_libraries(transformer_test PRIVATE mytorch GTest::gtest_main)
//     gtest_discover_tests(transformer_test)
//
// Run:  ctest --test-dir build -R Shift

#include "mytorch/autograd.h"
#include "mytorch/loss.h"
#include "mytorch/nn/module.h"
#include "mytorch/optim.h"
#include "mytorch/tensor.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using torch::CPU;
using torch::DType;
using torch::MSE;
using torch::SGD;
using torch::Tensor;
using torch::autograd::VarPtr;
using torch::autograd::Variable;
namespace nn = torch::nn;

// --- task configuration -----------------------------------------------------
// Deliberately tiny: this runs on every `ctest`, and a slow milestone test is
// one people stop running. B != T != D so a transposed or collapsed axis can't
// accidentally line up.
static constexpr int64_t SB = 4;   // batch
static constexpr int64_t ST = 6;   // sequence length
static constexpr int64_t SD = 32;  // d_model
static constexpr int64_t SH = 4;   // heads  (SD % SH == 0)
static constexpr int64_t SL = 2;   // layers
static constexpr int64_t SFF = 64; // d_ff

// --- helpers ----------------------------------------------------------------

static std::vector<float> noise(int64_t n, uint32_t salt = 0) {
  std::vector<float> v(n);
  for (int64_t i = 0; i < n; ++i) {
    uint32_t h = (static_cast<uint32_t>(i) + salt * 0x9E3779B9u) * 2654435761u;
    v[i] = static_cast<float>(static_cast<int32_t>(h % 21u) - 10) * 0.1f;
  }
  return v;
}

static VarPtr leaf3d(int64_t B, int64_t T, int64_t D,
                     const std::vector<float> &vals) {
  Tensor t({B, T, D}, DType::Float32, CPU);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  float *p = t.data_ptr<float>();
  for (size_t i = 0; i < vals.size(); ++i)
    p[i] = vals[i];
  return Variable::leaf(t, false);
}

// Flat row-major target for the shift task: zeros at t == 0, otherwise the
// previous position's input vector.
static std::vector<float> shifted(const std::vector<float> &in) {
  std::vector<float> out(in.size(), 0.0f);
  for (int64_t b = 0; b < SB; ++b)
    for (int64_t t = 1; t < ST; ++t)
      for (int64_t d = 0; d < SD; ++d)
        out[(b * ST + t) * SD + d] = in[(b * ST + (t - 1)) * SD + d];
  return out;
}

struct Losses {
  float first;
  float last;
};

static Losses train(nn::Module &model, SGD &opt, const VarPtr &x,
                    const VarPtr &y, int steps) {
  Losses l{0.0f, 0.0f};
  for (int i = 0; i < steps; ++i) {
    opt.zero_grad();
    MSE loss(model(x), y);
    float cur = loss.loss();
    if (i == 0)
      l.first = cur;
    l.last = cur;
    loss.backward();
    opt.step();
  }
  return l;
}

// ===========================================================================
// Control: the task must be unsolvable without attention
// ===========================================================================

TEST(ShiftTaskTest, ControlPositionwiseModelPlateaus) {
  // A position-wise MLP has no path from position t-1 to position t, so the
  // best it can do is predict the per-position mean. If this ever CONVERGES,
  // the task is broken (target leaking into the input) and every other test in
  // this file becomes meaningless -- so it runs first.
  torch::manual_seed(0);
  auto in = noise(SB * ST * SD);
  auto x = leaf3d(SB, ST, SD, in);
  auto y = leaf3d(SB, ST, SD, shifted(in));

  nn::Sequential mlp({std::make_shared<nn::Linear>(SD, SFF),
                      std::make_shared<nn::ReLU>(),
                      std::make_shared<nn::Linear>(SFF, SD)});
  SGD opt(mlp.params(), 0.01f);

  auto l = train(mlp, opt, x, y, 500);
  // TODO: tune this floor once you see the real plateau value. It should sit
  // comfortably ABOVE where the Transformer lands and BELOW the initial loss.
  EXPECT_GT(l.last, 0.05f)
      << "a position-wise model solved a task that requires cross-position "
         "information -- check that the target is not leaking into the input";
}

// ===========================================================================
// The milestone
// ===========================================================================

TEST(ShiftTaskTest, TransformerOverfitsTheShiftTask) {
  torch::manual_seed(0);
  auto in = noise(SB * ST * SD);
  auto x = leaf3d(SB, ST, SD, in);
  auto y = leaf3d(SB, ST, SD, shifted(in));

  // TODO: construct the model, e.g.
  //   nn::Transformer model(SD, SH, SL, SFF, /*max_context=*/ST);
  // It must map {B,T,D} -> {B,T,D}: positional parameter added to the input,
  // SL blocks, final norm. No embedding and no output head -- inputs are
  // already vectors and the loss is MSE.
  //
  // TODO: SGD opt(model.params(), lr);
  //       auto l = train(model, opt, x, y, steps);
  //
  // Start with lr=0.01 and 2000 steps. If it plateaus rather than diverging,
  // that is an optimization problem (raise lr, or add more steps), not a
  // correctness one -- the control test above is what tells you which.

  // EXPECT_LT(l.last, l.first) << "loss never moved";
  // EXPECT_LT(l.last, 0.01f) << "did not overfit; final loss " << l.last;
  GTEST_SKIP() << "fill in once nn::Transformer exists";
}

TEST(ShiftTaskTest, LearnedFunctionIsActuallyTheShift) {
  // A small loss scalar can be small for the wrong reasons. This checks the
  // function itself, position by position -- the same reason MlpLearnsXor
  // inspects all four predictions instead of trusting the loss.
  torch::manual_seed(0);
  auto in = noise(SB * ST * SD);
  auto x = leaf3d(SB, ST, SD, in);
  auto y = leaf3d(SB, ST, SD, shifted(in));
  auto want = shifted(in);

  // TODO: build + train the same model as above, then:
  //
  //   auto out = model(x);                       // hold the VarPtr! binding
  //   const float *p = out->data().data_ptr<float>();  // only data_ptr dangles
  //   for (int64_t i = 0; i < SB * ST * SD; ++i)
  //     EXPECT_NEAR(p[i], want[i], 0.05f) << "position " << (i / SD % ST);
  //
  // Position 0 is worth watching specifically: it has no predecessor and must
  // emit zeros, and the ONLY way the model can tell position 0 apart is the
  // positional parameter. If every position but 0 is right, positions aren't
  // wired up.
  GTEST_SKIP() << "fill in once nn::Transformer exists";
}

TEST(ShiftTaskTest, EveryParameterIsReachableFromTheTopLevel) {
  // register_module has to be called at every level of nesting -- Transformer
  // -> block -> {attention, norms, feed-forward}. Miss one and params()
  // silently returns a subset: the model still trains, just not all of it, and
  // nothing raises. Hand-count and compare.
  //
  // TODO: nn::Transformer model(SD, SH, SL, SFF, ST);
  //
  // Per block: 4 (attention Wq/Wk/Wv/Wo)
  //          + 2 or 4 (two norms: 1 param each for RMSNorm, 2 for LayerNorm)
  //          + 4 (feed-forward: 2 Linears x weight+bias)
  // Plus 1 positional parameter and the final norm's parameters.
  //
  // EXPECT_EQ(model.params().size(), <the number you computed>);
  //
  // Also assert no duplicate names, which catches a prefix that isn't applied.
  GTEST_SKIP() << "fill in once nn::Transformer exists";
}

TEST(ShiftTaskTest, WholeModelIsStillCausal) {
  // MhaTest.IsCausal proves the attention layer masks correctly. This proves
  // the property survives the full stack -- a residual connection wired to the
  // wrong tensor, or a norm that mixes across positions, would break causality
  // while leaving every shape and every gradient intact.
  //
  // TODO: build the model, run it on `in`, save the output. Then perturb ONLY
  // the last position of the input, run again, and assert every output
  // position < ST-1 is bit-identical (EXPECT_FLOAT_EQ, not EXPECT_NEAR -- an
  // exactly-causal model gives exactly the same bits).
  GTEST_SKIP() << "fill in once nn::Transformer exists";
}
