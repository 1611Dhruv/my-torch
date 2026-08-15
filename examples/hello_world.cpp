#include "mytorch/loss.h"
#include "mytorch/nn/module.h"
#include "mytorch/optim.h"
#include "mytorch/tensor.h"
#include <iostream>
#include <random>

void example_linear_handrolled() {
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

int main() {
  example_linear_loop();
  return 0;
}
