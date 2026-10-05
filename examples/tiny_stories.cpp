#include "mytorch/autograd.h"
#include "mytorch/loss.h"
#include "mytorch/nn/module.h"
#include "mytorch/optim.h"
#include "mytorch/tensor.h"
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistdio.h>

template <typename T> class Load {
public:
  Load(const T *tokens_map, size_t num_tokens, size_t batch_sz, size_t seq_len,
       torch::DType dtype, torch::Device dev)
      : _tokens_map(tokens_map),
        _num_tokens(num_tokens),
        _batch_sz(batch_sz),
        _seq_len(seq_len),
        _dtype(dtype),
        _dev(dev),
        _dist(0, num_tokens - seq_len - 1),
        _rg(std::random_device{}()) {
    if (_num_tokens < seq_len + 1) {
      throw std::invalid_argument("Cant complete 1 seq with given tokens");
    }
  }

  std::pair<torch::Tensor, torch::Tensor> batch() {
    torch::Tensor X = torch::Tensor::zeros(
        {_batch_sz, _seq_len}, torch::dtype_of<T>(), torch::Device::CPU);
    torch::Tensor Y = torch::Tensor::zeros(
        {_batch_sz, _seq_len}, torch::dtype_of<T>(), torch::Device::CPU);

    for (size_t i = 0; i < _batch_sz; i++) {
      size_t start = _dist(_rg);
      memcpy(X[i].data_ptr<T>(), _tokens_map + start, sizeof(T) * _seq_len);
      memcpy(Y[i].data_ptr<T>(), _tokens_map + start + 1, sizeof(T) * _seq_len);
    }
    return {X.to(_dtype, _dev), Y.to(_dtype, _dev)};
  }

private:
  const T *_tokens_map;
  size_t _num_tokens;
  size_t _batch_sz;
  size_t _seq_len;
  std::mt19937 _rg;
  std::uniform_int_distribution<size_t> _dist;
  torch::DType _dtype;
  torch::Device _dev;
};

void train_and_save(const std::string &model_file, const char *data,
                    int64_t num_tokens) {
  constexpr int64_t B = 128;
  constexpr int64_t T = 64;

  constexpr int64_t VOCAB = 256;
  constexpr int64_t DMODEL = 128;
  constexpr int64_t NHEADS = 4;
  constexpr int64_t DFF = (8 * DMODEL) / 3;
  constexpr int64_t NBLOCKS = 10;

  Load<uint8_t> loader(reinterpret_cast<const uint8_t *>(data), num_tokens, B,
                       T, torch::DType::Int32, torch::Device::CPU);

  torch::nn::Transformer model(VOCAB, DMODEL, DFF, NBLOCKS, 4, T);
  torch::Adam opt(model.params(), 0.001);

  constexpr int64_t EPOCH = 10;
  for (int64_t epoch = 0; epoch < EPOCH; epoch++) {
    auto [x_data, y_data] = loader.batch();
    auto X = torch::autograd::Variable::leaf(x_data, false);
    auto Y = torch::autograd::Variable::leaf(y_data, false);

    opt.zero_grad();
    auto Y_hat = model(X);
    auto loss = torch::CrossEntropy(Y_hat, Y);
    std::cout << "epoch " << epoch << ", Avg Loss: " << loss.loss()
              << std::endl;

    loss.backward();
    opt.step();
  }
  model.save(model_file);
}

size_t get_file_size(int fd) {
  struct stat s;
  if (fstat(fd, &s) == -1) {
    std::cerr << "Error reading the size of token file\n";
    return 1;
  }

  return s.st_size;
}

int main(int arg, char **argv) {
  std::string token_file;
  std::string model_file;

  std::cout << "Enter the token file name: ";
  std::cin >> token_file;
  std::cout << "Enter the model file name (will try to load or save): ";
  std::cin >> model_file;

  // Map the input file to mmap
  int fd;
  char *tokens_map;

  fd = open(token_file.c_str(), O_RDONLY);
  if (fd == -1) {
    std::cerr << "Error opening the token file: " << token_file << "\n";
    return 1;
  }

  size_t fbytes = get_file_size(fd);
  tokens_map =
      static_cast<char *>(mmap(0, fbytes, PROT_READ, MAP_SHARED, fd, 0));

  train_and_save(model_file, tokens_map, fbytes);
  return 0;
}
