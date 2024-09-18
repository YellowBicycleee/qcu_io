#include "lattice_desc.h"
#include <cstdio>

bool Latt_Desc::operator== (const Latt_Desc& rhs) 
{
    bool res = true;
    #pragma unroll
    for (int i = 0; i < MAX_DIM; ++i) {
        if (data[i] != rhs.data[i]) {
            res = false;
            break;
        }
    }
    return res;
}

void Latt_Desc::detail () 
{
    fprintf(stdout, "Lattice information, latt_desc = \n");
    fprintf(stdout, "[Lx, Ly, Lz, Lt] = [%d, %d, %d, %d]\n", 
                    data[X_DIM], data[Y_DIM], data[Z_DIM], data[T_DIM]);
}

bool LatticeCoordinate::operator== (const LatticeCoordinate& rhs) 
{
    bool res = true;
    #pragma unroll
    for (int i = 0; i < MAX_DIM; ++i) {
        if (data[i] != rhs.data[i]) {
            res = false;
            break;
        }
    }
    return res;
}

void LatticeCoordinate::detail () {
    fprintf(stdout, "Lattice information, local coordinate = \n");
    fprintf(stdout, "[x, y, z, t] = [%d, %d, %d, %d]\n", data[X_DIM], data[Y_DIM], data[Z_DIM], data[T_DIM]);
}

LatticeCoordinate LatticeCoordinate::globalLattCoord_4D
    (
        const MPI_Desc& mpi_desc, 
        const MPI_Coordinate& mpi_coord
    ) 
{
    LatticeCoordinate latt_coord;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        latt_coord.data[i] = mpi_desc.data[i] * mpi_coord.data[i] + data[i];
    }
    return latt_coord;
}

int32_t LatticeCoordinate::getIdx1D (const Latt_Desc& latt_desc) 
{
    int32_t idx = 0;
    idx += data[X_DIM];
    idx += data[Y_DIM] * latt_desc.data[X_DIM];
    idx += data[Z_DIM] * latt_desc.data[X_DIM] * latt_desc.data[Y_DIM];
    idx += data[T_DIM] * latt_desc.data[X_DIM] * latt_desc.data[Y_DIM] * latt_desc.data[Z_DIM];
    return idx;
}

LatticeCoordinate LatticeCoordinate::getIdx4D (const int32_t idx, const Latt_Desc& latt_desc) 
{
    LatticeCoordinate latt_coord;
    int32_t Lx = latt_desc.data[X_DIM];
    int32_t Ly = latt_desc.data[Y_DIM];
    int32_t Lz = latt_desc.data[Z_DIM];
    
    latt_coord.data[X_DIM] = idx % Lx;
    latt_coord.data[Y_DIM] = (idx / Lx) % Ly;
    latt_coord.data[Z_DIM] = (idx / (Lx * Ly)) % Lz;
    latt_coord.data[T_DIM] = idx / (Lx * Ly * Lz);
    return latt_coord;
}