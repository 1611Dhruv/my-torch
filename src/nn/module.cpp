#include <cmath>
#include <mytorch/nn/module.h>
#include <string>

namespace torch {
namespace nn {

// Module
void Module::_collect(const std::string &prefix,
                      std::vector<std::pair<std::string, std::shared_ptr<Variable>>> &out) const {
  for (const auto &[param_name, pptr] : _params) {
    out.emplace_back(std::make_pair(prefix + param_name, pptr));
  }
  for (const auto &[module_name, mptr] : _modules) {
    mptr->_collect(prefix + module_name + ".", out);
  }
}

std::vector<std::pair<std::string, std::shared_ptr<Variable>>> Module::named_params() const {
  std::vector<std::pair<std::string, std::shared_ptr<Variable>>> out;
  this->_collect("", out);
  return out;
}

std::vector<std::shared_ptr<Variable>> Module::params() const {
  auto npms = named_params();
  std::vector<std::shared_ptr<Variable>> out;
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

void Module::register_module(const std::string &name, Module *module) {
  _modules.emplace_back(std::make_pair(name, module));
}
std::shared_ptr<Variable> Module::register_param(const std::string &name, std::shared_ptr<Variable> param) {
  _params.emplace_back(std::make_pair(name, param));
  return param;
}

// Linear layer
Linear::Linear(int64_t in_dim, int64_t out_dim, DType dtype, Device dev) {
  // Construct a _in by _out
  // so that forward is just: x = [Batch, _in] @ weight
  _weight = register_param("weight", Variable::leaf(Tensor::randn({in_dim, out_dim}, dev, 0, std::sqrt(2.0 / in_dim))));
  _bias = register_param("bias", Variable::leaf(Tensor::zeros({out_dim}, dtype, dev)));
}

std::shared_ptr<Variable> Linear::forward(std::shared_ptr<Variable> inp) {
  return autograd::add(autograd::matmul(inp, _weight), _bias);
}

} // namespace nn
} // namespace torch
