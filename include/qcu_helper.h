#pragma once

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define QCU_HOST_DEVICE __forceinline__ __device__ __host__
#define QCU_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define QCU_HOST_DEVICE __forceinline__ __device__
#define QCU_DEVICE __forceinline__ __device__
#else
#define QCU_HOST_DEVICE inline
#define QCU_DEVICE inline
#endif

#define QCU_HOST __host__
#define QCU_GLOBAL __global__ static

namespace qcu {

enum class QcuStatus {
    kSuccess,                    ///< Operation was successful.
    kErrorInternal,              ///< An error within CUTLASS occurred.
    kErrorMemoryAllocation,      ///< Kernel launch failed due to insufficient device memory.
    kInvalid                     ///< Status is unspecified.
};

}