#pragma once 

#include "lattice_desc.h"
#include "qcu_public.h"
#include "base/buffer.h"
#include <memory>

namespace qcu {
class Gauge {
public:
    Gauge(const Gauge& rhs) : use_external_(true), precision_(rhs.precision_), 
                            buffer_data_type_(rhs.buffer_data_type_),
                            lattice_desc_(rhs.lattice_desc_),
                            gauge_buffer_(rhs.gauge_buffer_) {}
    Gauge(  QCU_PRECISION qcu_precision,
            BufferDataType buffer_data_type, 
            const Latt_Desc& latt_desc, 
            std::shared_ptr<base::DeviceAllocator> allocator
          ) {}
private:
    bool use_external_ = false;
    QCU_PRECISION precision_ = QCU_PRECISION::QCU_PRECISION_UNDEFINED;
    // 描述数据的排布
    BufferDataType buffer_data_type_;
    Latt_Desc& lattice_desc_;    // 描述lattice的维度
    std::shared_ptr<base::Buffer> gauge_buffer_;

};
}  // namespace qcu