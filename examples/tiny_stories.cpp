#include "mytorch/autograd.h"
#include "mytorch/loss.h"
#include "mytorch/nn/module.h"
#include "mytorch/optim.h"
#include "mytorch/tensor.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Hyperparameters (shared by training and inference so architectures match)
// ---------------------------------------------------------------------------
namespace cfg {
constexpr int64_t B = 128;
constexpr int64_t T = 64;
constexpr int64_t VOCAB = 256;
constexpr int64_t DMODEL = 128;
constexpr int64_t NHEADS = 4;
constexpr int64_t DFF = (8 * DMODEL) / 3;
constexpr int64_t NBLOCKS = 10;
constexpr int64_t EPOCHS = 10;
constexpr float LR = 0.001f;

constexpr int64_t GEN_TOKENS = 200;
constexpr float TEMPERATURE = 0.8f;
} // namespace cfg

// ---------------------------------------------------------------------------
// Data loader
// ---------------------------------------------------------------------------
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
        _rg(std::random_device{}()) {
    // Check BEFORE building the distribution, otherwise the bound underflows.
    if (_num_tokens < seq_len + 1) {
      throw std::invalid_argument("Can't complete 1 seq with given tokens");
    }
    _dist = std::uniform_int_distribution<size_t>(0, num_tokens - seq_len - 1);
  }

  std::pair<torch::Tensor, torch::Tensor> batch() {
    torch::Tensor X = torch::Tensor::zeros(
        {_batch_sz, _seq_len}, torch::dtype_of<T>(), torch::Device::CPU);
    torch::Tensor Y = torch::Tensor::zeros(
        {_batch_sz, _seq_len}, torch::dtype_of<T>(), torch::Device::CPU);

    for (int64_t i = 0; i < _batch_sz; i++) {
      size_t start = _dist(_rg);
      memcpy(X[i].data_ptr<T>(), _tokens_map + start, sizeof(T) * _seq_len);
      memcpy(Y[i].data_ptr<T>(), _tokens_map + start + 1, sizeof(T) * _seq_len);
    }
    return {X.to(_dtype, _dev), Y.to(_dtype, _dev)};
  }

private:
  // Declared in the same order as the init list.
  const T *_tokens_map;
  size_t _num_tokens;
  int64_t _batch_sz;
  int64_t _seq_len;
  torch::DType _dtype;
  torch::Device _dev;
  std::mt19937 _rg;
  std::uniform_int_distribution<size_t> _dist;
};

// ---------------------------------------------------------------------------
// RAII wrapper around an mmapped file
// ---------------------------------------------------------------------------
struct MappedFile {
  const uint8_t *data = nullptr;
  size_t size = 0;
  int fd = -1;

  explicit MappedFile(const std::string &path) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd == -1)
      throw std::runtime_error("Error opening token file: " + path);

    struct stat s;
    if (fstat(fd, &s) == -1) {
      close(fd);
      throw std::runtime_error("Error reading size of token file: " + path);
    }
    size = static_cast<size_t>(s.st_size);

    void *p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("mmap failed for token file: " + path);
    }
    data = static_cast<const uint8_t *>(p);
  }

  ~MappedFile() {
    if (data)
      munmap(const_cast<uint8_t *>(data), size);
    if (fd != -1)
      close(fd);
  }

  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;
};

// ---------------------------------------------------------------------------
// Model construction (heap-allocated so params() pointers stay stable)
// ---------------------------------------------------------------------------
std::unique_ptr<torch::nn::Transformer> make_model() {
  auto model = std::make_unique<torch::nn::Transformer>(
      cfg::VOCAB, cfg::DMODEL, cfg::DFF, cfg::NBLOCKS, cfg::NHEADS, cfg::T,
      torch::DType::Float32, torch::Device::CUDA);
  model->set_pe("learned");
  return model;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------
void train(torch::nn::Transformer &model, const uint8_t *data,
           size_t num_tokens) {
  Load<uint8_t> loader(data, num_tokens, cfg::B, cfg::T, torch::DType::Int32,
                       torch::Device::CUDA);
  torch::Adam opt(model.params(), cfg::LR);

  for (int64_t epoch = 0; epoch < cfg::EPOCHS; epoch++) {
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
}

// ---------------------------------------------------------------------------
// Inference: read prompts from stdin, sample byte-by-byte
// ---------------------------------------------------------------------------
int32_t sample(const float *logits, float temperature, std::mt19937 &rg) {
  float mx = *std::max_element(logits, logits + cfg::VOCAB);
  std::vector<double> w(cfg::VOCAB);
  for (int64_t v = 0; v < cfg::VOCAB; v++)
    w[v] = std::exp((logits[v] - mx) / temperature);
  std::discrete_distribution<int32_t> dist(w.begin(), w.end());
  return dist(rg);
}

void infer(torch::nn::Transformer &model) {
  std::mt19937 rg(std::random_device{}());
  std::string line;

  while (true) {
    std::cout << "\nPrompt (empty line to quit): " << std::flush;
    if (!std::getline(std::cin, line) || line.empty())
      break;

    std::vector<int32_t> ctx;
    for (unsigned char c : line)
      ctx.push_back(c);

    std::cout << line << std::flush;
    for (int64_t step = 0; step < cfg::GEN_TOKENS; step++) {
      // Learned PE only covers T positions, so crop to the last T tokens.
      int64_t len = std::min<int64_t>(ctx.size(), cfg::T);

      torch::Tensor x = torch::Tensor::zeros({1, len}, torch::DType::Int32,
                                             torch::Device::CPU);
      memcpy(x.data_ptr<int32_t>(), ctx.data() + ctx.size() - len,
             sizeof(int32_t) * len);
      auto X = torch::autograd::Variable::leaf(
          x.to(torch::DType::Int32, torch::Device::CUDA), false);

      auto Y_hat = model(X); // [1, len, VOCAB]
      torch::Tensor logits =
          Y_hat->data().to(torch::DType::Float32, torch::Device::CPU);
      const float *last = logits.data_ptr<float>() + (len - 1) * cfg::VOCAB;

      int32_t next = sample(last, cfg::TEMPERATURE, rg);
      ctx.push_back(next);
      std::cout << static_cast<char>(next) << std::flush;
    }
    std::cout << "\n";
  }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
void usage(const char *prog) {
  std::cerr << "Usage:\n"
            << "  " << prog
            << " -t <tokens> -m <model>       train from scratch\n"
            << "  " << prog
            << " -t <tokens> -m <model> -c    continue training\n"
            << "  " << prog << " -i -m <model>                run inference\n"
            << "  " << prog
            << " -t <tokens> -m <model> -i    train, then infer\n"
            << "\nFlags:\n"
            << "  -t <file>  token file (byte-level); enables training\n"
            << "  -m <file>  model file to load and/or save (required)\n"
            << "  -c         continue training from existing model (needs -t)\n"
            << "  -i         inference mode (prompts read from stdin)\n"
            << "  -h         show this help\n";
}

int main(int argc, char **argv) {
  std::string token_file, model_file;
  bool infer_mode = false;
  bool continue_mode = false;

  int opt;
  while ((opt = getopt(argc, argv, "t:m:ich")) != -1) {
    switch (opt) {
    case 't':
      token_file = optarg;
      break;
    case 'm':
      model_file = optarg;
      break;
    case 'i':
      infer_mode = true;
      break;
    case 'c':
      continue_mode = true;
      break;
    case 'h':
      usage(argv[0]);
      return 0;
    default:
      usage(argv[0]);
      return 1;
    }
  }

  if (optind < argc) {
    std::cerr << "Unexpected argument: " << argv[optind] << "\n";
    usage(argv[0]);
    return 1;
  }

  const bool train_mode = !token_file.empty();

  if (model_file.empty()) {
    std::cerr << "Error: -m <model> is required\n";
    usage(argv[0]);
    return 1;
  }
  if (continue_mode && !train_mode) {
    std::cerr << "Error: -c requires -t <tokens>\n";
    return 1;
  }
  if (!train_mode && !infer_mode) {
    std::cerr << "Error: nothing to do; pass -t to train and/or -i to infer\n";
    usage(argv[0]);
    return 1;
  }

  try {
    auto model = make_model();

    // Load when continuing, or when inferring without training first.
    if (continue_mode || (infer_mode && !train_mode)) {
      std::cout << "Loading model from " << model_file << "\n";
      model->load(model_file);
    }

    if (train_mode) {
      MappedFile tokens(token_file);
      train(*model, tokens.data, tokens.size);
      model->save(model_file);
      std::cout << "Saved model to " << model_file << "\n";
    }

    if (infer_mode)
      infer(*model);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
