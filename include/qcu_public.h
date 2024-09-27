#pragma once

#include <cstdint>

#define QCU_DEVICE __device__ 
#define QCU_HOST __host__
#define QCU_DEVICE_HOST __host__ __device__
#define QCU_GLOBAL __global__

constexpr int MAGIC_BYTE_SIZE = 4;
constexpr int Nd = 4; // 4维
constexpr int Ns = 4; // nSpinor = 4

// 1 Byte
enum class DataFormat : uint8_t {
    FORMAT_UNKNOWN = 0,
    QUDA_FORMAT,
    QUDA_FORMAT_EO_PRECONDITONED,
    QDP_FORMAT,
};
enum class StoragePrecision : uint8_t {
    PRECISION_UNKNOWN = 0,
    PRECISION_HALF,
    PRECISION_FLOAT,
    PRECISION_DOUBLE
};

enum class StorageType : uint8_t {
    TYPE_UNKNOWN = 0,
    TYPE_FERMION,
    TYPE_GAUGE
};

enum class MrhsShuffled : uint8_t {
    MRHS_SHUFFLED_NO = 0,   // 多个mrhs并揉在一起
    MRHS_SHUFFLED_YES,      // m个向量的元素相邻
};

enum LatticeDimension : int32_t {
    X_DIM = 0,
    Y_DIM,
    Z_DIM,
    T_DIM,
};