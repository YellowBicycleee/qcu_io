#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <complex>


struct LatticeIOHandler {
    bool file_opened = false;
    int fd = -1;
    const char *file_path = nullptr;

    LatticeIOHandler(const char *file_path) : file_path(file_path) {
        fd = open(file_path, O_RDWR); // O_RDWR: Read and write
        if (fd == -1) {
            throw std::runtime_error("Failed to open file");
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
        const char *file_path                       // 文件路径
    );

template <typename _FloatType = double>
void write_to_file (
        std::complex<_FloatType> *device_src_data,       // 数据指针
        std::complex<_FloatType> *device_dst_data,       // 数据指针
        std::complex<_FloatType> *device_gauge_data,     // 数据指针
        const char *file_path                       // 文件路径
    );