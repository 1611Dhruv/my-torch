#include <cmath>
#include <mytorch/nn/module.h>
#include <string>

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

} // namespace nn
} // namespace torch
