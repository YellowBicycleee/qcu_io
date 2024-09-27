#pragma once
#include "qcu_mpi_desc.h"
#include "lattice_desc.h"
#include "lqcd_read_write.h"
#include <string>
#include <cstdint>
#include "qcu_public.h"
struct TerminalConfig {
    uint8_t gauge_in_file_configured = 0;
    uint8_t gauge_out_file_configured = 0;
    uint8_t fermion_in_file_configured = 0;
    uint8_t fermion_out_file_configured = 0;
    uint8_t lattice_configured = 0;
    uint8_t mpi_configured = 0;

    int32_t Nc = 0;
    int32_t mInput = 0;
    
    MPI_Desc mpi_desc;          // 描述MPI的划分
    Latt_Desc lattice_desc;     // 描述格子的大小（整体）
    std::string gauge_in_file;  // 输入的组态文件
    std::string gauge_out_file; // 输出的组态文件
    std::string fermion_in_file; // 输入的费米子文件
    std::string fermion_out_file; // 输出的费米子文件

    void detail();
};

void get_lattice_config(
    int argc,                           // in 
    char *argv[],                       // in
    TerminalConfig &config              // out
);
