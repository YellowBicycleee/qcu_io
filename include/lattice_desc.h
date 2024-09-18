#pragma once 
#include "qcu_public.h"
#include "qcu_mpi_desc.h"
#include <cstdint>
#include <string>

struct Latt_Desc {
    static constexpr int32_t MAX_DIM = 4;
    int32_t data[MAX_DIM];  // X_DIM = 0, Y_DIM = 1, Z_DIM = 2, T_DIM = 3
    bool operator== (const Latt_Desc& rhs);
    void detail();
};

struct LatticeCoordinate {
    static constexpr int32_t MAX_DIM = 4;
    int32_t data[MAX_DIM];  // X_DIM = 0, Y_DIM = 1, Z_DIM = 2, T_DIM = 3
    bool operator== (const LatticeCoordinate& rhs);
    void detail();
    LatticeCoordinate globalLattCoord_4D(const MPI_Desc& mpi_desc, const MPI_Coordinate& mpi_coord);
    int32_t getIdx1D (const Latt_Desc& latt_desc);

    static LatticeCoordinate getIdx4D (const int32_t idx, const Latt_Desc& latt_desc);
};