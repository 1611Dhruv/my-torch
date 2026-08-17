#ifndef NN_MODULE_H
#define NN_MODULE_H
#include "mytorch/autograd.h"
#include "mytorch/tensor.h"
#include <vector>

namespace torch {
namespace nn {

namespace ag = autograd;
// Module holds the params and its submodules
class Module {
public:
  // Module should not be copied or deleted
  Module() = default;
  Module(const Module &) = delete;
  Module(Module &&) = delete;
  Module &operator=(const Module &) = delete;
  Module &operator=(Module &&) = delete;
  virtual ~Module() = default;

  // Virtual forward function
  virtual ag::VarPtr forward(ag::VarPtr inp) = 0;
  ag::VarPtr operator()(ag::VarPtr inp) { return forward(inp); };

  // public functions for things :)
  std::vector<std::pair<std::string, ag::VarPtr>> named_params() const;
  std::vector<ag::VarPtr> params() const;
  void zero_grad();
  void to(DType dtype, Device dev);

protected:
  // Each nn module should be able to either register a param or a module
  void register_module(const std::string &name, Module *module);
  ag::VarPtr register_param(const std::string &name, ag::VarPtr param);

private:
  // Just use raw ptr to submodule this guy might have
  std::vector<std::pair<std::string, Module *>> _modules;
  // Params can be passed in as a shared_ptr, because thats what we do today
  std::vector<std::pair<std::string, ag::VarPtr>> _params;
  void _collect(const std::string &prefix,
                std::vector<std::pair<std::string, ag::VarPtr>> &out) const;
};

class Linear : public Module {
public:
  Linear(int64_t in_dim, int64_t out_dim, DType dtype = DType::Float32,
         Device dev = CPU);
  ag::VarPtr forward(ag::VarPtr inp) override;

private:
  ag::VarPtr _weight;
  ag::VarPtr _bias;
};

class ReLU : public Module {
public:
  ReLU() {}
  ag::VarPtr forward(ag::VarPtr inp) override {
    return torch::autograd::relu(inp);
  };
};

class Sequential : public Module {
public:
  Sequential(std::initializer_list<std::shared_ptr<Module>> layers);
  ag::VarPtr forward(ag::VarPtr inp) override;

private:
  std::vector<std::shared_ptr<Module>> _layers;
};

class LayerNorm : public Module {
public:
  LayerNorm(int64_t in_dim, DType dtype = DType::Float32, Device dev = CPU);
  ag::VarPtr forward(ag::VarPtr inp) override;

private:
  ag::VarPtr _gain;
  ag::VarPtr _bias;
};

class RMSNorm : public Module {
public:
  RMSNorm(int64_t in_dim, DType dtype = DType::Float32, Device dev = CPU);
  ag::VarPtr forward(ag::VarPtr inp) override;

private:
  ag::VarPtr _gain;
};

class MultiHeadSelfAttention : public Module {
public:
  MultiHeadSelfAttention(int64_t d_model, int64_t n_heads, int64_t max_context,
                         DType dtype = DType::Float32, Device dev = CPU);
  ag::VarPtr forward(ag::VarPtr inp) override;

private:
  ag::VarPtr _Wq, _Wk, _Wv, _Wo;
  int64_t _d_model, _n_heads, _max_context;
  Tensor _causal_mask;
};

class FFN : public Module {
public:
  FFN(int64_t d_model, int64_t d_ff, DType dtype = DType::Float32,
      Device dev = CPU);
  ag::VarPtr forward(ag::VarPtr inp) override;

private:
  ag::VarPtr _W1, _W2;
};

class TransformerBlock : public Module {
public:
  TransformerBlock(int64_t d_model, int64_t d_ff, int64_t n_heads,
                   int64_t max_context, DType dtype = DType::Float32,
                   Device dev = CPU);
  ag::VarPtr forward(ag::VarPtr inp) override;

private:
  MultiHeadSelfAttention _atten;
  RMSNorm _n1, _n2;
  FFN _ff;
};

class Transformer : public Module {
public:
  Transformer(int64_t vocab_size, int64_t d_model, int64_t d_ff,
              int64_t n_blocks, int64_t max_context,
              DType dtype = DType::Float32, Device dev = Device::CPU);
  ag::VarPtr forward(ag::VarPtr inp) override;

private:
  ag::VarPtr _pe;
  std::vector<std::shared_ptr<TransformerBlock>> _blocks;
  // Linear _unembed;
};

} // namespace nn
} // namespace torch

#endif
