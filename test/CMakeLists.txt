cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_ARCHITECTURES 89)    # for 4060
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

project(QCU_IO_TESTS LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}

add_executable       (test_io ${PROJECT_SOURCE_DIR}/test_io.cpp)
target_link_libraries(test_io qcu_io)
