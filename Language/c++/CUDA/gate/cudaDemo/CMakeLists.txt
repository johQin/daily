cmake_minimum_required(VERSION 3.25)
project(cudaDemo CUDA)

set(CMAKE_CUDA_STANDARD 17)
include_directories(/usr/include /usr/local/cuda-12.0/include common)
add_executable(cudaDemo 2_11_cputimer.cu)

set_target_properties(cudaDemo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
