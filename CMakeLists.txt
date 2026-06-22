cmake_minimum_required(VERSION 3.15)
project(MyTorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # let clangd/IDEs find include paths
set(CMAKE_CUDA_STANDARD 20)
# Pin to the dev GPU's arch (RTX 5050 = Blackwell sm_120). `native` mis-detects to
# sm_75 here, which forces a PTX-JIT fallback that the driver (CUDA 13.2) is too old
# to handle vs nvcc 13.3 ("unsupported toolchain"). Explicit sm_120 emits real SASS.
set(CMAKE_CUDA_ARCHITECTURES 120)
# Emit -I include flags inline in compile_commands.json instead of hiding them in
# an nvcc response file (--options-file), which clangd can't read -> it loses the
# project include path and red-lines every .cu. Keeps the editor happy.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Core library: everything in src/ links into one target the tests + examples use.
# Empty .cpp files are fine -- add real ones here as you implement them.
add_library(mytorch
  src/storage.cpp
  src/tensor.cpp
  src/ops/elementwise.cpp
  src/ops/elementwise.cu
  src/ops/matmul.cu
  src/ops/matmul.cpp
  src/ops/dispatch.cpp
  src/autograd/functional.cpp
  #  src/data/mnist.cpp
)
target_include_directories(mytorch PUBLIC include)
target_compile_features(mytorch PUBLIC cxx_std_20)

# Allow __device__ lambdas defined inside __host__ functions (the binary_kernel op
# functors). CUDA-only: the generator expression keeps this flag off the .cpp files.
target_compile_options(mytorch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(mytorch PUBLIC CUDA::cudart)

# Example
add_executable(hello examples/hello_world.cpp)
target_link_libraries(hello PRIVATE mytorch)

# Tests (build with -DMYTORCH_BUILD_TESTS=ON, on by default)
option(MYTORCH_BUILD_TESTS "Build the test suite" ON)
if(MYTORCH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
