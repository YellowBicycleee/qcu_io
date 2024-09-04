#pragma once
#include <cstdint>

enum class DataFormat : int32_t {
    QUDA_FORMAT = 0,
    QUDA_FORMAT_EO_PRECONDITONED,
    QDP_FORMAT,
};

enum class MrhsShuffled : int32_t {
    MRHS_SHUFFLED_NO = 0,   // 多个mrhs并揉在一起
    MRHS_SHUFFLED_YES,      // m个向量的元素相邻
};

enum class ReadWriteFlag : int32_t {
    RW_NO = 0,
    RW_YES
};