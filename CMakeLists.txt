cmake_minimum_required(VERSION 3.15)
project(MyTorch LANGUAGES CXX)

add_executable(hello examples/hello_world.cpp)
target_compile_features(hello PRIVATE cxx_std_20)
