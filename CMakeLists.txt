cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 89)    # for 4060 laptop
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA C CXX)

project(QCU_IO LANGUAGES C CXX CUDA)

# find MPI
find_package(MPI REQUIRED)
if(NOT MPI_FOUND)
    message(FATAL_ERROR "cannot find mpi, please install mpi first")
else()
    message(STATUS "find mpi: ${MPI_CXX_INCLUDE_DIRS}")
endif()

# set hdf5 parallel
set(HDF5_PREFER_PARALLEL TRUE)
set(HDF5_USE_STATIC_LIBRARIES OFF)

# set hdf5 root
# set(HDF5_ROOT "/usr/lib/x86_64-linux-gnu/hdf5/openmpi")
# if use mpich, set hdf5 root
# set(HDF5_ROOT "/usr/lib/x86_64-linux-gnu/hdf5/mpich")

find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

# verify hdf5 parallel
if(HDF5_IS_PARALLEL)
    message(STATUS "find hdf5 parallel")
else()
    message(WARNING "HDF5 is not parallel, may affect parallel function")
endif()

if(NOT HDF5_FOUND)
    message(FATAL_ERROR "cannot find hdf5, please install hdf5 first")
else()
    message(STATUS "find hdf5: ${HDF5_INCLUDE_DIRS}")
    message(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")
    message(STATUS "HDF5_CXX_LIBRARIES: ${HDF5_CXX_LIBRARIES}")
endif()

aux_source_directory(${PROJECT_SOURCE_DIR}/src/io IO_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/base QCU_BASE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/precondition QCU_PRECONDITION)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/buffer_shape QCU_BUFFER_SHAPE)

add_library(qcu_io SHARED ${IO_SRC} ${LATT_DESC} ${QCU_BASE} ${QCU_PRECONDITION} ${QCU_BUFFER_SHAPE})
add_library(qcu_io_static STATIC ${IO_SRC} ${LATT_DESC} ${QCU_BASE} ${QCU_PRECONDITION} ${QCU_BUFFER_SHAPE})


target_include_directories(qcu_io 
    PUBLIC ${PROJECT_SOURCE_DIR}/include 
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    PUBLIC ${MPI_CXX_INCLUDE_DIRS} 
    PUBLIC ${HDF5_INCLUDE_DIRS})

target_include_directories(qcu_io_static 
    PUBLIC ${PROJECT_SOURCE_DIR}/include 
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
    PUBLIC ${MPI_CXX_INCLUDE_DIRS} 
    PUBLIC ${HDF5_INCLUDE_DIRS})

target_link_libraries(qcu_io PUBLIC MPI::MPI_CXX ${HDF5_LIBRARIES})
target_link_libraries(qcu_io_static PUBLIC STATIC MPI::MPI_CXX ${HDF5_LIBRARIES})


add_subdirectory(test)
