// Milestone test: the shift task, on token ids.
//
// The task: input is {B,T} of token ids drawn from 1..V-1; the target at
// position t is the one-hot of the id at position t-1, and position 0 targets
// the one-hot of a reserved BOS id (0). The model emits {B,T,V} logits and is
// trained with MSE against the one-hots on ONE fixed batch until the loss is
// ~0. This is "overfit a single batch": we are not testing generalization, we
// are testing that the machinery can express and learn the function at all.
//
// Why THIS task: a position-wise network (Embedding -> Linear -> ReLU ->
// Linear, applied independently at each t) only ever sees the CURRENT id, and
// the batch is built so the same id shows up with several different
// predecessors. The best a position-wise model can do is predict the average
// one-hot of those predecessors, which leaves an irreducible MSE we compute
// directly from the data (positionwise_optimum). Only attention can route
// position t-1's id into position t. So landing well BELOW that optimum is
// direct evidence that attention works, and ControlPositionwiseModelPlateaus
// proves the task actually discriminates.
//
// Shift RIGHT (copy the previous token), not left: the needed information is
// in the past, which a causal mask permits.
//
// Position 0 is the positional-encoding check. With no positional signal, a
// model cannot tell "I am at position 0, emit BOS" from "I am somewhere else,
// copy my predecessor": attention weights depend only on content. If every
// position but 0 is right, positions aren't wired up.
//
// Run:  ctest --test-dir build -R Shift

#include "mytorch/autograd.h"
#include "mytorch/loss.h"
#include "mytorch/nn/module.h"
#include "mytorch/optim.h"
#include "mytorch/tensor.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <set>
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
// Deliberately tiny: this runs on every `ctest`. B != T so a transposed axis
// can't line up by accident. V is small enough that 24 positions over 7
// usable ids guarantees repeated ids with different predecessors.
static constexpr int64_t SB = 4;   // batch
static constexpr int64_t ST = 6;   // sequence length
static constexpr int64_t SV = 8;   // vocab; id 0 is BOS and never an input
static constexpr int64_t SD = 32;  // d_model
static constexpr int64_t SH = 4;   // heads  (SD % SH == 0)
static constexpr int64_t SL = 2;   // layers
static constexpr int64_t SFF = 64; // d_ff
static constexpr int32_t BOS = 0;

// --- helpers ----------------------------------------------------------------

// Deterministic "random" ids in 1..SV-1.
static std::vector<int32_t> token_ids(uint32_t salt = 0) {
  std::vector<int32_t> v(SB * ST);
  for (int64_t i = 0; i < SB * ST; ++i) {
    uint32_t h = (static_cast<uint32_t>(i) + salt * 0x9E3779B9u) * 2654435761u;
    v[i] = 1 + static_cast<int32_t>((h >> 8) % (SV - 1));
  }
  return v;
}

static VarPtr id_leaf(const std::vector<int32_t> &vals) {
  Tensor t({SB, ST}, DType::Int32, CPU);
  EXPECT_EQ(static_cast<int64_t>(vals.size()), t.numel());
  int32_t *p = t.data_ptr<int32_t>();
  for (size_t i = 0; i < vals.size(); ++i)
    p[i] = vals[i];
  return Variable::leaf(t, false);
}

// The id each position should predict: BOS at t == 0, else the previous id.
static std::vector<int32_t> shifted_ids(const std::vector<int32_t> &in) {
  std::vector<int32_t> out(in.size(), BOS);
  for (int64_t b = 0; b < SB; ++b)
    for (int64_t t = 1; t < ST; ++t)
      out[b * ST + t] = in[b * ST + (t - 1)];
  return out;
}

// {B,T,V} one-hot of `ids`, as a Float32 leaf.
static VarPtr onehot_leaf(const std::vector<int32_t> &ids) {
  Tensor t({SB, ST, SV}, DType::Float32, CPU);
  float *p = t.data_ptr<float>();
  std::fill(p, p + t.numel(), 0.0f);
  for (int64_t i = 0; i < SB * ST; ++i)
    p[i * SV + ids[i]] = 1.0f;
  return Variable::leaf(t, false);
}

// The lowest MSE any function of the CURRENT id alone can reach: for each id,
// predict the mean one-hot of its targets. This is the plateau a position-wise
// model converges to, computed from the data rather than guessed.
static float positionwise_optimum(const std::vector<int32_t> &in,
                                  const std::vector<int32_t> &want) {
  double total = 0.0;
  for (int32_t v = 1; v < SV; ++v) {
    std::vector<double> mean(SV, 0.0);
    int count = 0;
    for (int64_t i = 0; i < SB * ST; ++i)
      if (in[i] == v) {
        mean[want[i]] += 1.0;
        ++count;
      }
    if (count == 0)
      continue;
    for (auto &m : mean)
      m /= count;
    for (int64_t i = 0; i < SB * ST; ++i)
      if (in[i] == v)
        for (int32_t k = 0; k < SV; ++k) {
          double target = (k == want[i]) ? 1.0 : 0.0;
          total += (target - mean[k]) * (target - mean[k]);
        }
  }
  return static_cast<float>(total / (SB * ST * SV));
}

static std::vector<float> to_vec(const Tensor &t) {
  Tensor c = t.contiguous();
  const float *p = c.data_ptr<float>();
  return std::vector<float>(p, p + c.numel());
}

static std::vector<int32_t> argmax_rows(const std::vector<float> &logits) {
  std::vector<int32_t> out(SB * ST);
  for (int64_t i = 0; i < SB * ST; ++i) {
    auto row = logits.begin() + i * SV;
    out[i] = static_cast<int32_t>(std::max_element(row, row + SV) - row);
  }
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

// The batch must actually contain ids with more than one distinct predecessor,
// or the task degenerates into a lookup table that a position-wise model can
// solve. Checked once here so every other test can rely on it.
TEST(ShiftTaskTest, TaskIsNotAPositionwiseLookup) {
  auto in = token_ids();
  auto want = shifted_ids(in);
  int ambiguous_ids = 0;
  for (int32_t v = 1; v < SV; ++v) {
    std::set<int32_t> preds;
    for (int64_t i = 0; i < SB * ST; ++i)
      if (in[i] == v)
        preds.insert(want[i]);
    if (preds.size() > 1)
      ++ambiguous_ids;
  }
  EXPECT_GE(ambiguous_ids, 3) << "too few ids have multiple predecessors";
  EXPECT_GT(positionwise_optimum(in, want), 0.01f);
}

// ===========================================================================
// Control: the task must be unsolvable without attention
// ===========================================================================

TEST(ShiftTaskTest, ControlPositionwiseModelPlateaus) {
  // Embedding -> MLP has no path from position t-1 to position t, so it can't
  // beat positionwise_optimum. If it does, the target is leaking into the
  // input and every other test in this file is meaningless.
  torch::manual_seed(0);
  auto in = token_ids();
  auto want = shifted_ids(in);
  auto x = id_leaf(in);
  auto y = onehot_leaf(want);
  const float opt_mse = positionwise_optimum(in, want);

  nn::Sequential mlp({std::make_shared<nn::Embedding>(SV, SD),
                      std::make_shared<nn::Linear>(SD, SFF),
                      std::make_shared<nn::ReLU>(),
                      std::make_shared<nn::Linear>(SFF, SV)});
  SGD opt(mlp.params(), 0.05f);

  auto l = train(mlp, opt, x, y, 1000);
  EXPECT_LT(l.last, l.first) << "control never learned anything";
  EXPECT_GT(l.last, 0.8f * opt_mse)
      << "a position-wise model beat the position-wise optimum (" << opt_mse
      << ") -- the target is leaking into the input";
}

// ===========================================================================
// The milestone
// ===========================================================================

TEST(ShiftTaskTest, TransformerOverfitsTheShiftTask) {
  torch::manual_seed(0);
  auto in = token_ids();
  auto want = shifted_ids(in);
  auto x = id_leaf(in);
  auto y = onehot_leaf(want);
  const float opt_mse = positionwise_optimum(in, want);

  nn::Transformer model(SV, SD, SFF, SL, SH, /*max_context=*/ST);
  SGD opt(model.params(), 0.01f);
  auto l = train(model, opt, x, y, 2000);

  EXPECT_LT(l.last, l.first) << "loss never moved";
  EXPECT_LT(l.last, 0.25f * opt_mse)
      << "did not beat the position-wise optimum (" << opt_mse
      << ") by a clear margin; final loss " << l.last
      << ". If it plateaus near the optimum, attention isn't routing "
         "information across positions, or there is no positional signal.";
}

TEST(ShiftTaskTest, LearnedFunctionIsActuallyTheShift) {
  // A small loss can be small for the wrong reasons. Check the function
  // itself: the argmax at every position must be the previous id (BOS at 0).
  torch::manual_seed(0);
  auto in = token_ids();
  auto want = shifted_ids(in);
  auto x = id_leaf(in);
  auto y = onehot_leaf(want);

  nn::Transformer model(SV, SD, SFF, SL, SH, ST);
  SGD opt(model.params(), 0.01f);
  train(model, opt, x, y, 2000);

  auto out = model(x); // hold the VarPtr; only the data_ptr would dangle
  auto got = argmax_rows(to_vec(out->data()));
  for (int64_t i = 0; i < SB * ST; ++i)
    EXPECT_EQ(got[i], want[i])
        << "batch " << (i / ST) << " position " << (i % ST)
        << (i % ST == 0 ? "  (position 0 must emit BOS: positional signal?)"
                        : "");
}

TEST(ShiftTaskTest, EveryParameterIsReachableFromTheTopLevel) {
  // register_module has to be called at every level of nesting. Miss one and
  // params() silently returns a subset: the model still runs, just part of it
  // never trains. Hand-count and compare.
  torch::manual_seed(0);
  nn::Transformer model(SV, SD, SFF, SL, SH, ST);

  // Per block: 4 attention (Wq, Wk, Wv, Wo) + 2 RMSNorm gains + 2 FFN weights.
  // Top level: 1 embedding table + 2 unembed (weight, bias).
  // Update this when a positional parameter or a final norm is added.
  const size_t per_block = 4 + 2 + 2;
  const size_t top_level = 1 + 2;
  EXPECT_EQ(model.params().size(), SL * per_block + top_level);

  std::set<std::string> names;
  for (auto &[name, p] : model.named_params()) {
    EXPECT_TRUE(names.insert(name).second) << "duplicate parameter name " << name;
  }
}

TEST(ShiftTaskTest, WholeModelIsStillCausal) {
  // MhaTest.IsCausal proves the attention layer masks correctly. This proves
  // the property survives the full stack: embedding, residuals, norms, unembed.
  torch::manual_seed(0);
  auto in = token_ids();
  nn::Transformer model(SV, SD, SFF, SL, SH, ST);

  auto l0 = to_vec(model(id_leaf(in))->data());
  auto bumped = in;
  for (int64_t b = 0; b < SB; ++b)
    bumped[b * ST + (ST - 1)] = 1 + (bumped[b * ST + (ST - 1)] % (SV - 1));
  auto l1 = to_vec(model(id_leaf(bumped))->data());

  for (int64_t b = 0; b < SB; ++b)
    for (int64_t t = 0; t + 1 < ST; ++t)
      for (int64_t k = 0; k < SV; ++k) {
        int64_t i = (b * ST + t) * SV + k;
        EXPECT_FLOAT_EQ(l0[i], l1[i])
            << "batch " << b << " position " << t
            << " changed when only the last position's id changed";
      }
}
