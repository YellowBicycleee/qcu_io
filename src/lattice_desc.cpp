#include "lattice_desc.h"
#include <cstdio>

bool LatticeDescription::operator== (const LatticeDescription& rhs) {
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

void LatticeDescription::detail () {
    fprintf(stdout, "Lattice information, total latt_desc = \n");
    fprintf(stdout, "Lx = %d, Ly = %d, Lz = %d, Lt = %d\n", 
                    data[X_DIM], data[Y_DIM], data[Z_DIM], data[T_DIM]);
}

bool MpiDescription::operator== (const MpiDescription& rhs) {
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

void MpiDescription::detail () {
    fprintf(stdout, "Mpi information, total latt_desc = \n");
    fprintf(stdout, "Nx = %d, Ny = %d, Nz = %d, Nt = %d\n", 
                    data[X_DIM], data[Y_DIM], data[Z_DIM], data[T_DIM]);
}