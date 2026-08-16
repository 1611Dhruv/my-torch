#ifndef NN_MODULE_H
#define NN_MODULE_H
#include "mytorch/autograd.h"
#include <vector>

namespace torch {
namespace nn {

using Variable = autograd::Variable;
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
  virtual std::shared_ptr<Variable> forward(std::shared_ptr<Variable> inp) = 0;
  std::shared_ptr<Variable> operator()(std::shared_ptr<Variable> inp) {
    return forward(inp);
  };

  // public functions for things :)
  std::vector<std::pair<std::string, std::shared_ptr<Variable>>>
  named_params() const;
  std::vector<std::shared_ptr<Variable>> params() const;
  void zero_grad();
  void to(DType dtype, Device dev);

protected:
  // Each nn module should be able to either register a param or a module
  void register_module(const std::string &name, Module *module);
  std::shared_ptr<Variable> register_param(const std::string &name,
                                           std::shared_ptr<Variable> param);

private:
  // Just use raw ptr to submodule this guy might have
  std::vector<std::pair<std::string, Module *>> _modules;
  // Params can be passed in as a shared_ptr, because thats what we do today
  std::vector<std::pair<std::string, std::shared_ptr<Variable>>> _params;
  void _collect(const std::string &prefix,
                std::vector<std::pair<std::string, std::shared_ptr<Variable>>>
                    &out) const;
};

class Linear : public Module {
public:
  Linear(int64_t in_dim, int64_t out_dim, DType dtype = DType::Float32,
         Device dev = CPU);
  std::shared_ptr<Variable> forward(std::shared_ptr<Variable> inp) override;

private:
  std::shared_ptr<Variable> _weight;
  std::shared_ptr<Variable> _bias;
};

class ReLU : public Module {
public:
  ReLU() {}
  std::shared_ptr<Variable> forward(std::shared_ptr<Variable> inp) override {
    return torch::autograd::relu(inp);
  };
};

class Sequential : public Module {
public:
  Sequential(std::initializer_list<std::shared_ptr<Module>> modules);
  std::shared_ptr<Variable> forward(std::shared_ptr<Variable> inp) override;

private:
  std::vector<std::shared_ptr<Module>> _modules;
};

} // namespace nn
} // namespace torch

#endif
