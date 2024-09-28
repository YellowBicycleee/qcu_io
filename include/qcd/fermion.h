#pragma once 
#include "lattice_desc.h"
#include "base/buffer.h"
#include <cstdint>
#include <memory>

namespace qcu {
class Fermion  {
public:
    Fermion() = delete;
private:
    BufferDataType buffer_data_type_ = BufferDataType::QCU_BUFFER_DATATYPE_UNKNOWN;    // 描述数据的排布
    int32_t nColor_;    // 颜色数
    int32_t nSpinor_;   // 自旋数
    int32_t mRHS_;      // 多重右手边
    const Latt_Desc& lattice_desc_;   // 描述lattice的维度
    std::shared_ptr<base::Buffer> data_;
};
}