#ifndef NN_MODULE_H
#define NN_MODULE_H
#include "mytorch/autograd.h"
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

} // namespace nn
} // namespace torch

#endif
