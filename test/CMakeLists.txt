cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_ARCHITECTURES 89)    # for 4060
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

project(QCU_IO_TESTS LANGUAGES C CXX CUDA) # project name = ${PROJECT_NAME}

add_executable       (test_output ${PROJECT_SOURCE_DIR}/test_output.cu)
target_link_libraries(test_output qcu_io)

add_executable       (test_input ${PROJECT_SOURCE_DIR}/test_input.cu)
target_link_libraries(test_input qcu_io)