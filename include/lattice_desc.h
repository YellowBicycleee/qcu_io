#pragma once 
#include <cstdint>
#include "lqcd_format_enum.h"

struct LatticeConfig
{
    int32_t m_Lx;
    int32_t m_Ly;
    int32_t m_Lz;
    int32_t m_Lt;
    int32_t m_Ns;
    int32_t m_Nd;
    int32_t m_Nc;
    int32_t m_MInput; // m rhs

    DataFormat   m_data_format    = DataFormat::QUDA_FORMAT;
    MrhsShuffled m_mrhs_shuffled  = MrhsShuffled::MRHS_SHUFFLED_NO;

    ReadWriteFlag m_src_rw_flag   = ReadWriteFlag::RW_YES;  // src读写标志
    ReadWriteFlag m_dst_rw_flag   = ReadWriteFlag::RW_YES;  // dst读写标志
    ReadWriteFlag m_gauge_rw_flag = ReadWriteFlag::RW_YES;  // gauge读写标志

    int32_t volume() const {
        if (m_Nd == 2) {
            return m_Lx * m_Ly;
        }
        else if (m_Nd == 3) {
            return m_Lx * m_Ly * m_Lz;
        }
        else if (m_Nd == 4) {
            return m_Lx * m_Ly * m_Lz * m_Lt;
        }
        else {
            return 0;
        }
    }

    int32_t SingleColorSpinorSiteLength () const {
        return m_Ns * m_Nc;
    }
    int32_t MrhsColorSpinorSiteLength () const {
        return m_MInput * SingleColorSpinorSiteLength();
    }

    int32_t GaugeLenth () const {
        return m_Nd * volume() * m_Nc * m_Nc; 
    }


    int32_t SingleColorSpinorLength () const {
        return volume() * SingleColorSpinorSiteLength();
    }
    int32_t MrhsColorSpinorLength () const {
        return m_MInput * SingleColorSpinorLength();
    }
};

