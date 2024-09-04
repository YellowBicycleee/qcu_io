#include "lattice_desc.h"
#include "lqcd_read_write.h"
#include <cstdio>
#include <complex>
#include <sys/mman.h>

// 默认双精度读写
template <typename _FloatType>
void read_from_file (
        std::complex<_FloatType> *device_src_data,       // 数据指针
        std::complex<_FloatType> *device_dst_data,       // 数据指针
        std::complex<_FloatType> *device_gauge_data,     // 数据指针
        const char *file_path                       // 文件路径
        // LatticeConfig &lattice_config,              // Lattice规格参数
    ) 
{
    // 打开文件
    LatticeIOHandler file_handler(file_path);
    // 读取文件头
    LatticeConfig lattice_config;
    FILE *file = fopen(file_path, "r");
    if (file == nullptr) {
        throw std::runtime_error("Failed to open file");
    }
    // 读取文件头
    fread(&lattice_config, sizeof(LatticeConfig), 1, file);

    // 读取src数据
    fseek(file, 0, SEEK_SET);
    auto file_size = ftell(file);

    void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, file_handler.fd, 0);
    auto mrhs_colorspinor_length = lattice_config.MrhsColorSpinorLength();
    auto gauge_length = lattice_config.GaugeLenth();

    // data前移动，跳过文件头
    data = static_cast<void *>(static_cast<char *>(data) + sizeof(LatticeConfig));
    if (lattice_config.m_src_rw_flag == ReadWriteFlag::RW_YES) {
        cudaMemcpy(device_src_data, data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyHostToDevice);
        data = static_cast<void *>(static_cast<std::complex<_FloatType> *>(data) + mrhs_colorspinor_length);
    }
    // 读取dst数据
    if (lattice_config.m_dst_rw_flag == ReadWriteFlag::RW_YES) {
        cudaMemcpy(device_dst_data, data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyHostToDevice);
        data = static_cast<void *>(static_cast<std::complex<_FloatType> *>(data) + mrhs_colorspinor_length);
    }
    // 读取gauge数据
    if (lattice_config.m_gauge_rw_flag == ReadWriteFlag::RW_YES) {
        cudaMemcpy(device_gauge_data, data, gauge_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyHostToDevice);
    }
    // read over
}

template <typename _FloatType>
void write_to_file (
        std::complex<_FloatType> *device_src_data,       // 数据指针
        std::complex<_FloatType> *device_dst_data,       // 数据指针
        std::complex<_FloatType> *device_gauge_data,     // 数据指针
        const char *file_path                       // 文件路径
    )
{
    // 打开文件
    LatticeIOHandler file_handler(file_path);
    // 读取文件头
    LatticeConfig lattice_config;
    FILE *file = fopen(file_path, "w");
    if (file == nullptr) {
        throw std::runtime_error("Failed to open file");
    }
    // 写文件头
    fwrite(&lattice_config, sizeof(LatticeConfig), 1, file);

    // 读取src数据
    fseek(file, 0, SEEK_SET);
    auto file_size = ftell(file);

    void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, file_handler.fd, 0);
    auto mrhs_colorspinor_length = lattice_config.MrhsColorSpinorLength();
    auto gauge_length = lattice_config.GaugeLenth();

    // data前移动，跳过文件头
    data = static_cast<void *>(static_cast<char *>(data) + sizeof(LatticeConfig));
    if (lattice_config.m_src_rw_flag == ReadWriteFlag::RW_YES) {
        cudaMemcpy(data, device_src_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyDeviceToHost);
        data = static_cast<void *>(static_cast<std::complex<_FloatType> *>(data) + mrhs_colorspinor_length);
    }
    // 读取dst数据
    if (lattice_config.m_dst_rw_flag == ReadWriteFlag::RW_YES) {
        cudaMemcpy(data, device_dst_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyDeviceToHost);
        data = static_cast<void *>(static_cast<std::complex<_FloatType> *>(data) + mrhs_colorspinor_length);
    }
    // 读取gauge数据
    if (lattice_config.m_gauge_rw_flag == ReadWriteFlag::RW_YES) {
        cudaMemcpy(data, device_gauge_data, gauge_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyDeviceToHost);
    }
    // write over
}