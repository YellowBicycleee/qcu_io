cmake_minimum_required(VERSION 3.8)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 89)    # for 4060 laptop
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

project(QCU_IO LANGUAGES C CXX CUDA)
add_library(qcu_io
        SHARED
        src/lattice_desc.cpp
        src/lattice_parse_terminal_param.cpp
        src/lqcd_read_write.cu
)

target_include_directories(qcu_io
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

add_subdirectory(test)
