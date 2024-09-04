//
// Created by wangj on 2024/9/4.
//
#include <iostream>
#include "lattice_desc.h"
#include <complex>
#include <vector>
using namespace std;

template <typename _Float>
void init_complex_vector(complex<_Float>* vec) {

}

int main(int argc, char *argv[]) {

    int Lx = 4;
    int Ly = 4;
    int Lz = 4;
    int Lt = 4;
    int Ns = 4;
    int Nc = 4;
    int Nd = 4;
    int mInput = 1;


    LatticeConfig lattice_config {
        .m_Lx = Lx,
        .m_Ly = Ly,
        .m_Lz = Lz,
        .m_Lt = Lt,
        .m_Ns = Ns,
        .m_Nd = Nd,
        .m_Nc = Nc,
        .m_MInput = mInput,
        .m_data_format    = DataFormat::QUDA_FORMAT,
        .m_mrhs_shuffled  = MrhsShuffled::MRHS_SHUFFLED_NO,
        .m_src_rw_flag   = ReadWriteFlag::RW_YES,
        .m_dst_rw_flag   = ReadWriteFlag::RW_YES,
        .m_gauge_rw_flag = ReadWriteFlag::RW_YES
    };
    const char* file_name = "test_io.txt";
    auto mrhs_colorspinor_len = lattice_config.MrhsColorSpinorLength();
    auto gauge_len = lattice_config.GaugeLenth();

}