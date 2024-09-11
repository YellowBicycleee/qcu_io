#pragma once 
#include <cstdint>
#include <string>
// #include "lqcd_format_enum.h"

// 1 Byte
enum class DataFormat : uint8_t {
    FORMAT_UNKNOWN = 0,
    QUDA_FORMAT,
    QUDA_FORMAT_EO_PRECONDITONED,
    QDP_FORMAT,
};
enum class StoragePrecision : uint8_t {
    PRECISION_UNKNOWN = 0,
    PRECISION_HALF,
    PRECISION_FLOAT,
    PRECISION_DOUBLE
};
// 1 Byte，指示是否存放fermionIn, fermionOut, gauge
enum IOPosition : uint8_t {
    IO_FERMION_IN  = 1,     // 000000001
    IO_FERMION_OUT = 2,     // 000000010
    IO_GAUGE       = 4,     // 000000100
};

enum class MrhsShuffled : uint8_t {
    MRHS_SHUFFLED_NO = 0,   // 多个mrhs并揉在一起
    MRHS_SHUFFLED_YES,      // m个向量的元素相邻
};

enum LatticeDimension : int32_t {
    X_DIM = 0,
    Y_DIM,
    Z_DIM,
    T_DIM
};

struct LatticeDescription {
    static constexpr int32_t MAX_DIM = 4;
    int32_t data[MAX_DIM];  // X_DIM = 0, Y_DIM = 1, Z_DIM = 2, T_DIM = 3

    bool operator== (const LatticeDescription& rhs);
    void detail();
};
struct MpiDescription {
    static constexpr int32_t MAX_DIM = 4;
    int32_t data[MAX_DIM];  // X_DIM = 0, Y_DIM = 1, Z_DIM = 2, T_DIM = 3
    bool operator== (const MpiDescription& rhs);
    void detail();
};

struct LatticeConfig
{

    DataFormat       m_data_format       = DataFormat::QUDA_FORMAT;
    StoragePrecision m_storage_precision = StoragePrecision::PRECISION_DOUBLE; // 默认double
    uint8_t          m_io_position       = IO_FERMION_IN | IO_FERMION_OUT | IO_GAUGE;
    MrhsShuffled     m_mrhs_shuffled     = MrhsShuffled::MRHS_SHUFFLED_NO;

    LatticeDescription m_lattice_desc;
    MpiDescription     m_mpi_desc;

    int32_t m_Nc;
    int32_t m_MInput; // m rhs
    int32_t m_Ns;
    int32_t m_Nd;

    bool check_config (const LatticeConfig& rhs) {
        return this->m_lattice_desc == rhs.m_lattice_desc
                && this->m_mpi_desc == rhs.m_mpi_desc
                && this->m_Nc == rhs.m_Nc
                && this->m_MInput == rhs.m_MInput;
    }
    // copy check_config检查范围外的参数
    void copy_info (const LatticeConfig& rhs) {
        m_data_format       = rhs.m_data_format;
        m_storage_precision = rhs.m_storage_precision;
        m_io_position       = rhs.m_io_position;
        m_mrhs_shuffled     = rhs.m_mrhs_shuffled;
        m_Ns                = rhs.m_Ns;
        m_Nd                = rhs.m_Nd;
    }

    int32_t volume() const {
        return m_lattice_desc.data[X_DIM] * m_lattice_desc.data[Y_DIM] * 
               m_lattice_desc.data[Z_DIM] * m_lattice_desc.data[T_DIM];
    }

    int32_t SingleColorSpinorSiteLength () const {
        return m_Ns * m_Nc;
    }
    int32_t MrhsColorSpinorSiteLength () const {
        return m_MInput * SingleColorSpinorSiteLength();
    }

    int32_t GaugeLength () const {
        return m_Nd * volume() * m_Nc * m_Nc; 
    }


    int32_t SingleColorSpinorLength () const {
        return volume() * SingleColorSpinorSiteLength();
    }
    int32_t MrhsColorSpinorLength () const {
        return m_MInput * SingleColorSpinorLength();
    }
};

LatticeConfig get_lattice_config(int argc, char *argv[], std::string& file);