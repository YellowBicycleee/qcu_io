#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <complex>
#include "lattice_desc.h"
#include <string>
struct LatticeIOHandler {
    static constexpr uint32_t QCU_READ_MODE  = O_RDONLY;
    static constexpr uint32_t QCU_WRITE_MODE = O_WRONLY | O_CREAT;
    static constexpr uint32_t QCU_READ_WRITE_MODE = O_RDWR | O_CREAT;
    // static constexpr uint32_t QCU_WRITE_CREATE_MODE = O_WRONLY | O_CREAT;

    bool file_opened = false;
    int fd = -1;
    const char *file_path_ = nullptr;

    LatticeIOHandler(const std::string& file_path, const int32_t file_open_mode) 
                            : file_path_(file_path.c_str()) 
    {
        if (file_open_mode != QCU_READ_MODE && file_open_mode != QCU_WRITE_MODE && 
            file_open_mode != QCU_READ_WRITE_MODE) {
            throw std::runtime_error("Invalid file open mode");
        }

        mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH; // rw-rw-rw-
        // fd = open(file_path_, O_RDWR | O_CREAT, mode); // O_RDWR: Read and write
        fd = open(file_path_, file_open_mode, mode); // O_RDWR: Read and write
        if (fd == -1) {
            throw std::runtime_error(std::string("Failed to open file ") + file_path);
        }
        file_opened = true;
    }
    ~LatticeIOHandler() noexcept {
        if (file_opened) {
            close(fd);
            file_opened = false;
        }
    }
};

// 默认双精度读写
template <typename _FloatType = double>
void read_from_file (
        std::complex<_FloatType> *device_src_data,       // 数据指针
        std::complex<_FloatType> *device_dst_data,       // 数据指针
        std::complex<_FloatType> *device_gauge_data,     // 数据指针
        const std::string& file_path,                    // 文件路径
        LatticeConfig& lattice_config
    );

template <typename _FloatType = double>
void write_to_file (
        std::complex<_FloatType> *device_src_data,       // 数据指针
        std::complex<_FloatType> *device_dst_data,       // 数据指针
        std::complex<_FloatType> *device_gauge_data,     // 数据指针
        const std::string& file_path,                          // 文件路径
        LatticeConfig& lattice_config
    );