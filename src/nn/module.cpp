#include "mytorch/autograd.h"
#include "mytorch/storage.h"
#include <cmath>
#include <mytorch/nn/module.h>
#include <string>

static constexpr double eps = 1e-9;

namespace torch {
namespace nn {

// Module
void Module::_collect(
    const std::string &prefix,
    std::vector<std::pair<std::string, ag::VarPtr>> &out) const {
  for (const auto &[param_name, pptr] : _params) {
    out.emplace_back(std::make_pair(prefix + param_name, pptr));
  }
  for (const auto &[module_name, mptr] : _modules) {
    mptr->_collect(prefix + module_name + ".", out);
  }
}

std::vector<std::pair<std::string, ag::VarPtr>> Module::named_params() const {
  std::vector<std::pair<std::string, ag::VarPtr>> out;
  this->_collect("", out);
  return out;
}

std::vector<ag::VarPtr> Module::params() const {
  auto npms = named_params();
  std::vector<ag::VarPtr> out;
  out.reserve(npms.size());
  for (auto &[_, v] : npms) {
    out.push_back(v);
  }
  return out;
}

void Module::zero_grad() {
  for (auto &v : params()) {
    v->zero_grad();
  }
}

void Module::to(DType dtype, Device dev) {
  for (auto &p : params()) {
    p->data() = p->data().to(dtype, dev);
  }
}

void Module::register_module(const std::string &name, Module *module) {
  _modules.emplace_back(std::make_pair(name, module));
}
ag::VarPtr Module::register_param(const std::string &name, ag::VarPtr param) {
  _params.emplace_back(std::make_pair(name, param));
  return param;
}

// Linear layer
Linear::Linear(int64_t in_dim, int64_t out_dim, DType dtype, Device dev) {
  // Construct a _in by _out
  // so that forward is just: x = [Batch, _in] @ weight
  _weight = register_param(
      "weight", ag::Variable::leaf(Tensor::randn({in_dim, out_dim}, dev, 0,
                                                 std::sqrt(2.0 / in_dim))));
  _bias = register_param(
      "bias", ag::Variable::leaf(Tensor::zeros({out_dim}, dtype, dev)));
}

ag::VarPtr Linear::forward(ag::VarPtr inp) {
  return ag::add(ag::matmul(inp, _weight), _bias);
}

// Sequential
Sequential::Sequential(std::initializer_list<std::shared_ptr<Module>> modules)
    : _layers(modules) {
  for (int64_t i = 0; i < _layers.size(); i++) {
    register_module("module_" + std::to_string(i), _layers[i].get());
  }
}

ag::VarPtr Sequential::forward(ag::VarPtr inp) {
  auto out = inp;
  for (auto &m : _layers) {
    out = m->forward(out);
  }
  return out;
}

// Layer Norm
LayerNorm::LayerNorm(int64_t in_dim, DType dtype, Device dev) {
  _gain = register_param(
      "gain", ag::Variable::leaf(Tensor::ones({in_dim}, dtype, dev)));
  _bias = register_param(
      "bias", ag::Variable::leaf(Tensor::zeros({in_dim}, dtype, dev)));
}

ag::VarPtr LayerNorm::forward(ag::VarPtr x) {
  double H = x->data().shape().back();
  auto x_mean = ag::scale(ag::sum(x, {-1}, true), 1 / H);
  auto x_cent = ag::sub(x, x_mean);
  auto x_sd = ag::sqrt(ag::shift(
      ag::scale(ag::sum(ag::mult(x_cent, x_cent), {-1}, true), 1.0 / H), eps));

  auto x_norm = ag::div(x_cent, x_sd);
  return ag::add(ag::mult(_gain, x_norm), _bias);
}

// RMS Norm
RMSNorm::RMSNorm(int64_t in_dim, DType dtype, Device dev) {
  _gain = register_param(
      "gain", ag::Variable::leaf(Tensor::ones({in_dim}, dtype, dev)));
}

ag::VarPtr RMSNorm::forward(ag::VarPtr x) {
  double H = x->data().shape().back();
  auto rms = ag::sqrt(
      ag::shift(ag::scale(ag::sum(ag::mult(x, x), {-1}, true), 1 / H), eps));
  auto norm = ag::div(x, rms);
  return ag::mult(_gain, norm);
}

// MultiHeadSelfAttention
MultiHeadSelfAttention::MultiHeadSelfAttention(int64_t d_model, int64_t n_heads,
                                               int64_t max_context, DType dtype,
                                               Device dev)
    : _causal_mask({max_context, max_context}, DType::Float32, CPU),
      _d_model(d_model),
      _n_heads(n_heads),
      _max_context(max_context) {
  if (d_model % n_heads) {
    throw std::invalid_argument(
        "The Attention model heads must divide model dim");
  }
  _Wq = register_param(
      "Wq", ag::Variable::leaf(
                torch::Tensor::randn({d_model, d_model}).to(dtype, dev)));
  _Wk = register_param(
      "Wk", ag::Variable::leaf(
                torch::Tensor::randn({d_model, d_model}).to(dtype, dev)));
  _Wv = register_param(
      "Wv", ag::Variable::leaf(
                torch::Tensor::randn({d_model, d_model}).to(dtype, dev)));
  _Wo = register_param(
      "Wo", ag::Variable::leaf(
                torch::Tensor::randn({d_model, d_model}).to(dtype, dev)));
  for (int64_t i = 0; i < max_context; i++) {
    for (int64_t j = 0; j <= i; j++) {
      _causal_mask[i][j].item<double>() = -1e11;
    }
  }
  _causal_mask = _causal_mask.to(dtype, dev);
}

// Assume we got {T, N}
ag::VarPtr MultiHeadSelfAttention::forward(ag::VarPtr inp) {
  auto inp_shape = inp->data().shape();
  if (inp_shape.size() < 2) {
    // TODO: Could be changed to warning? but nah throw is right
    throw std::logic_error("MHA called with only one dim");
  }

  int N = inp_shape.size();
  int64_t d_model = inp_shape[N - 1];
  int64_t T = inp_shape[N - 2];

  if (d_model != _d_model) {
    throw std::logic_error("MHA called with diff model dim");
  }

  if (T > _max_context) {
    throw std::logic_error("MHA called with longer context than supported one");
  }

  int64_t B = 1;
  for (int i = N - 3; i >= 0; i--) {
    B *= inp_shape[i];
  }
  auto d_head = _d_model / _n_heads;

  auto Q = ag::matmul(inp, _Wq);
  auto K = ag::matmul(inp, _Wk);
  auto V = ag::matmul(inp, _Wv);

  // Make Q, K , V into (n_head, T, d_head)
  auto q_h = ag::transpose(ag::reshape(Q, {B, T, _n_heads, d_head}), 1, 2);
  auto k_h = ag::transpose(ag::reshape(K, {B, T, _n_heads, d_head}), 1, 2);
  auto v_h = ag::transpose(ag::reshape(V, {B, T, _n_heads, d_head}), 1, 2);

  auto qkt =
      ag::scale(ag::matmul(q_h, ag::transpose(k_h, -1, -2)), 1.0 / d_model);
  // Mask qkt
  qkt->data() =
      torch::add(qkt->data(), _causal_mask.slice(0, 0, T).slice(1, 0, T));
  auto sft = ag::softmax(qkt);
  // {B, n_h, T, d_h}
  auto head_out = ag::matmul(sft, v_h);
  auto merge_back = ag::reshape(ag::transpose(head_out, 1, 2), inp_shape);
  auto res = ag::matmul(merge_back, V);
  return res;
}

} // namespace nn
} // namespace torch
