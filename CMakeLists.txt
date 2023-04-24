cmake_minimum_required(VERSION 3.25)
project(seed_reverser CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(seed_reverser main.cu)

set_target_properties(seed_reverser PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
