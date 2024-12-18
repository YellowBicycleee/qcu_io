#include "qcu_mpi_desc.h"
#include <cstdio>

bool MPI_Desc::operator== (const MPI_Desc& rhs) {
    bool res = true;
    #pragma unroll
    for (int i = 0; i < kMaxDim; ++i) {
        if (data[i] != rhs.data[i]) {
            res = false;
            break;
        }
    }
    return res;
}

void MPI_Desc::detail () {
    fprintf(stdout, "Mpi information, mpi desc \n");
    fprintf(stdout, "[Nx, Ny, Nz, Nt] = [%d, %d, %d, %d]\n", 
                    data[X_DIM], data[Y_DIM], data[Z_DIM], data[T_DIM]);
}


bool MPI_Coordinate::operator== (const MPI_Coordinate& rhs) {
    bool res = true;
    #pragma unroll
    for (int i = 0; i < kMaxDim; ++i) {
        if (data[i] != rhs.data[i]) {
            res = false;
            break;
        }
    }
    return res;
}

void MPI_Coordinate::detail () {
    fprintf(stdout, "Mpi information, coordinate = \n");
    fprintf(stdout, "[x, y, z, t] = [%d, %d, %d, %d]\n", 
                data[X_DIM], data[Y_DIM], data[Z_DIM], data[T_DIM]);
}
