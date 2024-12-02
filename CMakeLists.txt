cmake_minimum_required(VERSION 3.8)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_package(CUDA REQUIRED)
    set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
endif()
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 89)    # for 4060 laptop
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)


project(QCU_IO LANGUAGES CXX CUDA)
# include(cmake/cuda.cmake)

# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
# option(USE_CPM "Use CPM for dependency management" OFF)


aux_source_directory(${PROJECT_SOURCE_DIR}/src/io IO_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/latt_desc LATT_DESC)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/base QCU_BASE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/precondition QCU_PRECONDITION)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/buffer_shape QCU_BUFFER_SHAPE)

add_library(qcu_io SHARED ${IO_SRC} ${LATT_DESC} ${QCU_BASE} ${QCU_PRECONDITION} ${QCU_BUFFER_SHAPE})
add_library(qcu_io_static STATIC ${IO_SRC} ${LATT_DESC} ${QCU_BASE} ${QCU_PRECONDITION} ${QCU_BUFFER_SHAPE})

# target_link_directories(qcu_io PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)
# target_link_libraries(qcu_io PUBLIC cudart)
target_include_directories(qcu_io PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(qcu_io PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(qcu_io_static PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(qcu_io_static PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_subdirectory(test)
