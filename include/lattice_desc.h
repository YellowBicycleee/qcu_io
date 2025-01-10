#pragma once

#include "qcu_public.h"

#include <cstdint>
#include <string>
#include <sstream>

namespace qcu {

struct FourDimDesc {
    static constexpr int32_t kMaxDim = 4;

    int32_t data[kMaxDim];  // X_DIM = 0, Y_DIM = 1, Z_DIM = 2, T_DIM = 3

    FourDimDesc() : data{0, 0, 0, 0} {}
    FourDimDesc(int32_t x, int32_t y, int32_t z, int32_t t) : data{x, y, z, t} {}

    bool operator== (const FourDimDesc& rhs) {
        #pragma unroll
        for (int32_t i = 0; i < kMaxDim; ++i) {
            if (data[i] != rhs.data[i]) {
                return false;
            }
        }
        return true;
    }
    std::string detail() {
        std::stringstream ss;
        ss << "(X = " << data[X_DIM] << ", Y = " << data[Y_DIM] 
           << ", Z = " << data[Z_DIM] << ", T = " << data[T_DIM] << ")";
        return ss.str();
    }
};


struct FourDimCoordinate {
    static constexpr int32_t kMaxDim = 4;
    int32_t data[kMaxDim];  // X_DIM = 0, Y_DIM = 1, Z_DIM = 2, T_DIM = 3
    
    FourDimCoordinate() : data{0, 0, 0, 0} {}
    FourDimCoordinate(int32_t x, int32_t y, int32_t z, int32_t t) : data{x, y, z, t} {}
    
    bool operator== (const FourDimCoordinate& rhs) {
        #pragma unroll
        for (int32_t i = 0; i < kMaxDim; ++i) {
            if (data[i] != rhs.data[i]) {
                return false;
            }
        }
        return true;
    }
    std::string detail() {
        std::stringstream ss;
        ss << "(X = " << data[X_DIM] << ", Y = " << data[Y_DIM] 
           << ", Z = " << data[Z_DIM] << ", T = " << data[T_DIM] << ")";
        return ss.str();
    }
    int32_t getIdx1D (const FourDimDesc& dim_desc) {
        return data[T_DIM] * dim_desc.data[Z_DIM] * dim_desc.data[Y_DIM] * dim_desc.data[X_DIM] + 
               data[Z_DIM] * dim_desc.data[Y_DIM] * dim_desc.data[X_DIM] + 
               data[Y_DIM] * dim_desc.data[X_DIM] + 
               data[X_DIM];
    }
    int32_t getReversedIdx1D (const FourDimDesc& dim_desc) {
        return 
            data[X_DIM] * dim_desc.data[Y_DIM] * dim_desc.data[Z_DIM] * dim_desc.data[T_DIM] + 
            data[Y_DIM] * dim_desc.data[Z_DIM] * dim_desc.data[T_DIM] + 
            data[Z_DIM] * dim_desc.data[T_DIM] + 
            data[T_DIM];
    }
};

};