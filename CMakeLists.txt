cmake_minimum_required(VERSION 3.15)
project(MyTorch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # let clangd/IDEs find include paths
set(CMAKE_CUDA_STANDARD 20)
# Default to the dev GPU (RTX 5050/5090 = Blackwell sm_120); override per machine,
# e.g. -DMYTORCH_CUDA_ARCH=89 for an L40S.
set(MYTORCH_CUDA_ARCH 120 CACHE STRING "CUDA arch to build for")
set(CMAKE_CUDA_ARCHITECTURES ${MYTORCH_CUDA_ARCH})
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
  src/ops/reductions.cpp
  src/ops/reductions.cu
  src/ops/flash.cu
  src/autograd/functional.cpp
  src/nn/module.cpp
  src/optim.cpp
  src/loss.cpp
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

add_executable(tiny_stories examples/tiny_stories.cpp)
target_link_libraries(tiny_stories PRIVATE mytorch)

# Tests (build with -DMYTORCH_BUILD_TESTS=ON, on by default)
option(MYTORCH_BUILD_TESTS "Build the test suite" ON)
if(MYTORCH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
