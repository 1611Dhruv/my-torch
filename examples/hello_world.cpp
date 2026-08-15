#include "mytorch/loss.h"
#include "mytorch/nn/module.h"
#include "mytorch/optim.h"
#include "mytorch/tensor.h"
#include <iostream>
#include <random>

void example_linear_handrolled() {
  std::cout << "Example demonstrating the full training loop by manually doing the loss:\n";
  torch::Tensor x({1024, 1});
  torch::Tensor y({1024, 1});
  std::random_device rd{};

  auto x_ptr = x.data_ptr<float>();
  auto y_ptr = y.data_ptr<float>();
  for (int i = 0; i < 1024; i++) {
    x_ptr[i] = (rd() % 3000) / 3000.0f;
    y_ptr[i] = x_ptr[i] * 40 + 20;
  }

  auto x_var = torch::autograd::Variable::leaf(x, false);
  auto y_var = torch::autograd::Variable::leaf(y, false);

  // 1024 1 D data points
  torch::nn::Linear l1(1, 1);
  torch::Tensor lr({1});
  // Normalize by batch for now as we dont have reduce (too much effort)
  lr.data_ptr<float>()[0] = 0.1 / 1024;

  for (int epoch = 0; epoch < 1000; epoch++) {
    l1.zero_grad();

    auto pred = l1(x_var);
    auto diff = torch::autograd::sub(y_var, pred);
    auto loss = torch::autograd::mult(diff, diff);
    if (epoch % 100 == 0) {
      std::cout << "epoch " << epoch << ", Avg Loss: " << torch::sum(loss->data(), {}, false).item<float>() / 1024.0f
                << std::endl;
    }
    loss->backward();
    for (auto &p : l1.params()) {
      p->data() = torch::sub(p->data(), torch::mult(*p->grad(), lr));
    }
  }

  std::cout << "Final L1: " << std::endl;
  for (auto &p : l1.params()) {
    std::cout << p->data() << std::endl;
  }
}

void example_linear_loop() {
  std::cout << "Example demonstrating the full training loop with optimin and all:\n";
  torch::Tensor x({1024, 1});
  torch::Tensor y({1024, 1});
  std::random_device rd{};

  auto x_ptr = x.data_ptr<float>();
  auto y_ptr = y.data_ptr<float>();
  for (int i = 0; i < 1024; i++) {
    x_ptr[i] = (rd() % 3000) / 3000.0f;
    y_ptr[i] = x_ptr[i] * 40 + 20;
  }

  std::cout << "x: " << x << std::endl;
  std::cout << "y: " << y << std::endl;
  auto x_var = torch::autograd::Variable::leaf(x, false);
  auto y_var = torch::autograd::Variable::leaf(y, false);

  // 1024 1 D data points
  torch::nn::Linear l1(1, 1);
  torch::SGD optim(l1.params(), 0.1);

  for (int epoch = 0; epoch < 1000; epoch++) {
    optim.zero_grad();
    auto pred = l1(x_var);
    auto loss = torch::MSE(pred, y_var);
    if (epoch % 100 == 0) {
      std::cout << "epoch " << epoch << ", Avg Loss: " << loss.loss() << std::endl;
    }
    loss.backward();
    optim.step();
  }

  std::cout << "Final L1: " << std::endl;
  for (auto &p : l1.params()) {
    std::cout << p->data() << std::endl;
  }
}

void example_xor() {
  torch::Tensor x({6767, 2}, torch::DType::UInt8);
  torch::Tensor y({6767, 1}, torch::DType::UInt8);
  std::random_device rd{};
  for (int i = 0; i < 6767; i++) {
    uint8_t x1 = x.data_ptr<uint8_t>()[i * x.strides()[0]] = static_cast<uint8_t>(rd() % 256);
    uint8_t x2 = x.data_ptr<uint8_t>()[i * x.strides()[0] + 1] = static_cast<uint8_t>(rd() % 256);
    y.data_ptr<uint8_t>()[i * y.strides()[0]] = x1 ^ x2;
  }

  using namespace torch::nn;
  using namespace torch::autograd;
  Sequential net{
      std::make_shared<Linear>(2, 3),
      std::make_shared<ReLU>(),
      std::make_shared<Linear>(3, 1),
  };

  VarPtr x_var = Variable::leaf(x, false);
  VarPtr y_var = Variable::leaf(y, false);
  auto opt = torch::SGD(net.params(), 0.01);

  for (int epoch = 0; epoch < 1000; epoch++) {
    opt.zero_grad();
    auto pred = net(x_var);
    auto loss = torch::MSE(pred, y_var);
    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << ": Loss=" << loss.loss();
    }
    loss.backward();
    opt.step();
  }
  std::cout << "Final: ";
  for (auto &[name, param] : net.named_params()) {
    std::cout << name << ": " << param << std::endl;
  }
}

int main() {
  example_linear_handrolled();
  example_linear_loop();
  example_xor();
  return 0;
}
