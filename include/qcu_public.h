#pragma once

#include <cstdint>

#define NOT_IMPLEMENTED "Not implemented yet\n" 

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

enum class StorageType : uint8_t {
    TYPE_UNKNOWN = 0,
    TYPE_FERMION,
    TYPE_GAUGE
};

enum class MrhsShuffled : uint8_t {
    MRHS_SHUFFLED_NO = 0,   // 多个mrhs并揉在一起
    MRHS_SHUFFLED_YES,      // m个向量的元素相邻
};


// FROM QCU
// clang-format off
enum QcuPrecision {   // precision
    kPrecisionHalf = 0, 
    kPrecisionSingle,
    kPrecisionDouble,
    kPrecisionUndefined
};

enum QCU_PRECONDITION { 
    QCU_NO_PRECONDITION = 0,
    QCU_EO_PC_4D 
};

enum QCU_PARITY {
    EVEN_PARITY = 0,
    ODD_PARITY = 1,
    PARITY = 2
};

enum DIMS : int32_t {
    X_DIM = 0,
    Y_DIM,
    Z_DIM,
    T_DIM,
};

enum DIRS {
    BWD = 0,
    FWD = 1,
    DIRECTIONS
};

// enum QcuDaggerFlag { 
//     kDaggerNo = 0, 
//     kDaggerYes, 
//     kDaggerUndefined 
// };

// enum DslashType {
//     kDslashWilson = 0, 
//     kDslashClover,
//     kDslashUnkown
// };

enum MemoryStorage {
    NON_COALESCED = 0,
    COALESCED = 1,
};

enum ShiftDirection {
    TO_COALESCE = 0,
    TO_NON_COALESCE = 1,
};

enum MatrixOrder {
    ROW_MAJOR = 0,
    COLUMN_MAJOR = 1,
};

enum BufferDataType {
    QCU_BUFFER_DATATYPE_UNKNOWN = 0,
    QCU_FERMION_TZYXSC = 0,
    QCU_FERMION_PTZYXSC,
    QCU_FERMION_PTZYXSC_EVEN,
    QCU_FERMION_PTZYXSC_ODD,
    QCU_FERMION_TZYXSCM,
    QCU_FERMION_PTZYXSCM,
    QCU_FERMION_PTZYXSCM_EVEN,
    QCU_FERMION_PTZYXSCM_ODD,
    // GAUGE
    QCU_GAUGE_DTZYXCC,
    QCU_GAUGE_DPTZYXCC
};

#define errorQcu(msg)                                                                \
    do {                                                                             \
        fprintf(stderr, msg);                                                        \
        fprintf(stderr, "Error happened in file %s, line %d\n", __FILE__, __LINE__); \
        exit(1);                                                                     \
    } while (0)


constexpr int MAX_DIM = 4;
constexpr int WARP_SIZE = 32;
constexpr int WARP_PER_BLOCK = 4;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

#define IDX2D(y, x, lx) ((y) * (lx) + (x))
#define IDX3D(z, y, x, ly, lx) ((((z) * (ly)) + (y)) * (lx) + (x))
#define IDX4D(t, z, y, x, lz, ly, lx) ((((t) * (lz) + (z)) * (ly) + (y)) * (lx) + (x))