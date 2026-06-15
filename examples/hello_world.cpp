#include "mytorch/tensor.h"
#include <iostream>

int main() {
  torch::Tensor t({2, 3});

  std::cout << t[0] << " " << t[0][1];
}
