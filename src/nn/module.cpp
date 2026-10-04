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

  for (auto &[n, m] : _modules) {
    m->to(dtype, dev);
  }
}

void Module::register_module(const std::string &name, Module *module) {
  _modules.emplace_back(std::make_pair(name, module));
}
ag::VarPtr Module::register_param(const std::string &name, ag::VarPtr param) {
  _params.emplace_back(std::make_pair(name, param));
  return param;
}

bool Module::unregister_param(const std::string &name) {
  bool found = false;
  int64_t num_params = _params.size();
  for (int64_t i = 0; i < num_params; i++) {
    if (_params[i].first == name) {
      found = true;
      swap(_params[i], _params[num_params - 1]);
      _params.pop_back();
      break;
    }
  }
  return found;
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

// Embedding layer
Embedding::Embedding(int64_t vocab_sz, int64_t model_dim, DType dtype,
                     Device dev) {
  // Construct a _in by _out
  // so that forward is just: x = [Batch, _in] @ weight
  _weight = register_param(
      "weight",
      ag::Variable::leaf(
          Tensor::randn({vocab_sz, model_dim}, dev, 0, 1).to(dtype, dev)));
}

ag::VarPtr Embedding::forward(ag::VarPtr inp) {
  return ag::index_select(_weight, inp);
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

// MultiHeadAttention
MultiHeadAttention::MultiHeadAttention(int64_t d_model, int64_t n_heads,
                                       int64_t max_context, DType dtype,
                                       Device dev, bool causal)
    : _d_model(d_model),
      _n_heads(n_heads),
      _max_context(max_context),
      _causal(causal) {
  if (d_model % n_heads) {
    throw std::invalid_argument(
        "The Attention model heads must divide model dim");
  }
  _Wq = register_param(
      "Wq", ag::Variable::leaf(torch::Tensor::randn({d_model, d_model}, dev, 0,
                                                    std::sqrt(2.0 / d_model))
                                   .to(dtype, dev)));
  _Wk = register_param(
      "Wk", ag::Variable::leaf(torch::Tensor::randn({d_model, d_model}, dev, 0,
                                                    std::sqrt(2.0 / d_model))
                                   .to(dtype, dev)));
  _Wv = register_param(
      "Wv", ag::Variable::leaf(torch::Tensor::randn({d_model, d_model}, dev, 0,
                                                    std::sqrt(2.0 / d_model))
                                   .to(dtype, dev)));
  _Wo = register_param(
      "Wo", ag::Variable::leaf(torch::Tensor::randn({d_model, d_model}, dev, 0,
                                                    std::sqrt(2.0 / d_model))
                                   .to(dtype, dev)));
}

// Assume we got {T, N}
ag::VarPtr MultiHeadAttention::forward(ag::VarPtr inp) {
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
  // NOTE: Kv Cache goes here

  // Make Q, K , V into (n_head, T, d_head)
  auto q_h = ag::transpose(ag::reshape(Q, {B, T, _n_heads, d_head}), 1, 2);
  auto k_h = ag::transpose(ag::reshape(K, {B, T, _n_heads, d_head}), 1, 2);
  auto v_h = ag::transpose(ag::reshape(V, {B, T, _n_heads, d_head}), 1, 2);

  // Flash is usable only on cuda with float32
  if (inp->data().device() == torch::Device::CUDA &&
      inp->data().dtype() == torch::DType::Float32) {
    return ag::flash_atten(q_h, k_h, v_h, _causal);
  } else {
    auto qkt = ag::scale(ag::matmul(q_h, ag::transpose(k_h, -1, -2)),
                         1.0 / std::sqrt(d_head));

    ag::VarPtr sft;
    if (_causal) {
      // Mask qkt
      if (!_causal_cached) {
        Tensor _causal_mask = Tensor::zeros({_max_context, _max_context});
        for (int64_t i = 0; i < _max_context; i++) {
          for (int64_t j = i + 1; j < _max_context; j++) {
            _causal_mask[i][j].item<float>() = -1e11;
          }
        }
        _causal_cached =
            _causal_mask.to(inp->data().dtype(), inp->data().device());
      }

      auto msk = ag::Variable::leaf(
          _causal_cached->slice(0, 0, T).slice(1, 0, T), false);
      sft = ag::softmax(ag::add(qkt, msk));
    } else {
      sft = ag::softmax(qkt);
    }
    // {B, n_h, T, d_h}
    auto head_out = ag::matmul(sft, v_h);
    auto merge_back = ag::reshape(ag::transpose(head_out, 1, 2), inp_shape);
    auto res = ag::matmul(merge_back, _Wo);
    return res;
  }
}

// FFN
FFN::FFN(int64_t d_model, int64_t d_ff, DType dtype, Device dev) {
  _W1 = register_param(
      "W1", ag::Variable::leaf(
                Tensor::randn({d_model, d_ff}, dev, 0, std::sqrt(2.0 / d_model))
                    .to(dtype, dev)));
  _W2 = register_param("W2",
                       ag::Variable::leaf(Tensor::randn({d_ff, d_model}, dev, 0,
                                                        std::sqrt(2.0 / d_ff))
                                              .to(dtype, dev)));
}

ag::VarPtr FFN::forward(ag::VarPtr inp) {
  return ag::matmul(ag::relu(ag::matmul(inp, _W1)), _W2);
}

// FFN_SwiGLU
FFN_SwiGLU::FFN_SwiGLU(int64_t d_model, int64_t d_ff, DType dtype, Device dev) {
  _Wgate = register_param(
      "W1", ag::Variable::leaf(
                Tensor::randn({d_model, d_ff}, dev, 0, std::sqrt(2.0 / d_model))
                    .to(dtype, dev)));
  _Wup = register_param(
      "W2", ag::Variable::leaf(
                Tensor::randn({d_model, d_ff}, dev, 0, std::sqrt(2.0 / d_model))
                    .to(dtype, dev)));
  _Wdown = register_param(
      "W3", ag::Variable::leaf(
                Tensor::randn({d_ff, d_model}, dev, 0, std::sqrt(2.0 / d_ff))
                    .to(dtype, dev)));
}

ag::VarPtr FFN_SwiGLU::forward(ag::VarPtr inp) {
  return ag::matmul(
      ag::mult(ag::silu(ag::matmul(inp, _Wgate)), ag::matmul(inp, _Wup)),
      _Wdown);
}

// Transformer Block
TransformerBlock::TransformerBlock(int64_t d_model, int64_t d_ff,
                                   int64_t n_heads, int64_t max_context,
                                   DType dtype, Device dev)
    : _atten(d_model, n_heads, max_context, dtype, dev),
      _ff(d_model, d_ff, dtype, dev),
      _n1(d_model, dtype, dev),
      _n2(d_model, dtype, dev) {
  register_module("norm1", &_n1);
  register_module("mha", &_atten);
  register_module("norm2", &_n2);
  register_module("ff", &_ff);
}

ag::VarPtr TransformerBlock::forward(ag::VarPtr inp) {
  auto atten_rich = ag::add(inp, _atten(_n1(inp)));
  auto ffn_rich = ag::add(atten_rich, _ff(_n2(atten_rich)));
  return ffn_rich;
}

// Transformer
Transformer::Transformer(int64_t vocab_size, int64_t d_model, int64_t d_ff,
                         int64_t n_blocks, int64_t n_heads, int64_t max_context,
                         DType dtype, Device dev)
    : _embed(vocab_size, d_model, dtype, dev),
      _unembed(d_model, vocab_size, dtype, dev),
      _max_context(max_context),
      _d_model(d_model),
      _dev(dev) {
  register_module("embed", &_embed);
  for (int i = 0; i < n_blocks; i++) {
    _blocks.emplace_back(std::make_shared<TransformerBlock>(
        d_model, d_ff, n_heads, max_context, dtype, dev));
    register_module("block " + std::to_string(i), _blocks.back().get());
  }
  register_module("unembed", &_unembed);
}

void Transformer::set_pe(std::string type) {
  auto cleanup = [&]() {
    switch (_pe_type) {
    case LEARNED: {
      unregister_param("learned_pe");
      _pe_func = nullptr;
    } break;
    default: {
    } break;
    }
  };

  if (type == "rope") {
    throw std::invalid_argument("rope aint roping yet");
  }
  if (type == "sin") {
    _pe_func = [&](ag::VarPtr inp) {
      // more
      return inp;
    };
    _pe_type = SIN;
  }
  if (type == "nope") {
    if (_pe_type != NOPE) {
      cleanup();
    }
    _pe_type = NOPE;
  }
  if (type == "learned") {
    if (_pe_type != LEARNED) {
      cleanup();
    }
    ag::VarPtr pe =
        ag::Variable::leaf(Tensor::randn({_max_context, _d_model}, _dev, 0, 1));
    Tensor pos = Tensor::iota({_max_context}, 0);

    register_param("learned_pe", pe);

    _pe_func = [pe, pos](ag::VarPtr inp) {
      int64_t T = inp->data().shape()[inp->data().shape().size() - 2];
      ag::VarPtr pos_ptr = ag::Variable::leaf(pos.slice(0, 0, T), false);

      return ag::add(inp, ag::index_select(pe, pos_ptr));
    };
    _pe_type = LEARNED;
  }
}
ag::VarPtr Transformer::forward(ag::VarPtr inp) {
  auto res = _embed(inp);

  if (_pe_type == LEARNED) {
    res = _pe_func(res);
  }
  for (auto &m : _blocks) {
    res = m->forward(res);
  }
  return _unembed(res);
};

} // namespace nn
} // namespace torch
