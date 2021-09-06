cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(test_cuda CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(test_cuda main.cu)

set_target_properties(test_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
