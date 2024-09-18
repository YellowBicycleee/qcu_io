#pragma once
#include <cstdint>
#include "qcu_public.h"
struct MPI_Desc {
    static constexpr int32_t MAX_DIM = 4;
    int32_t data[MAX_DIM];  // X_DIM = 0, Y_DIM = 1, Z_DIM = 2, T_DIM = 3
    bool operator== (const MPI_Desc& rhs);
    void detail();
};

struct MPI_Coordinate {
    static constexpr int32_t MAX_DIM = 4;
    int32_t data[MAX_DIM];  // X_DIM = 0, Y_DIM = 1, Z_DIM = 2, T_DIM = 3
    bool operator== (const MPI_Coordinate& rhs);
    void detail();
};