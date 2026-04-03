cmake_minimum_required(VERSION 3.24)
project(pto_gpu_st LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 121)

enable_testing()
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(testcase)
