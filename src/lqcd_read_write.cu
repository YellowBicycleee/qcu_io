#include "lattice_desc.h"
#include "lqcd_read_write.h"
#include <cstdio>
#include <complex>
#include <sys/mman.h>
#include <string>
#include "check_cuda.h"
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
using std::complex;
using std::cout;
using std::endl;
// #include <fcntl.h>
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
        throw std::runtime_error(std::string("Failed to open file") + file_path);
    }
    else {
        printf("FILE %s opened correctly\n", file_path);
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
        const char *file_path,                           // 文件路径
        LatticeConfig& lattice_config
    )
{
    auto mrhs_colorspinor_length = lattice_config.MrhsColorSpinorLength();
    auto gauge_length = lattice_config.GaugeLenth();

    complex<_FloatType>* host_src_data = new complex<_FloatType>[mrhs_colorspinor_length];
    complex<_FloatType>* host_dst_data = new complex<_FloatType>[mrhs_colorspinor_length];
    complex<_FloatType>* host_gauge    = new complex<_FloatType>[gauge_length];  
    CHECK_CUDA(cudaMemcpy(host_src_data, device_src_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyDeviceToHost)); 
    CHECK_CUDA(cudaMemcpy(host_dst_data, device_dst_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyDeviceToHost)); 
    CHECK_CUDA(cudaMemcpy(host_gauge, device_gauge_data, gauge_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyDeviceToHost));

    cout << "DEBUG, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>) = " << 
                    mrhs_colorspinor_length * sizeof(std::complex<_FloatType>) 
                    << " " << host_gauge[0] << endl;


    // 打开文件
    LatticeIOHandler file_handler(file_path);
    // 读取文件头
    // LatticeConfig lattice_config;
    FILE *file = fopen(file_path, "w");
    if (file == nullptr) {
        throw std::runtime_error(std::string("Failed to open file") + file_path);
    }
    else {
        printf("FILE %s opened correctly\n", file_path);
    }
    // 写文件头
    fwrite(&lattice_config, sizeof(LatticeConfig), 1, file);

    fclose(file);

    //获取文件属性
    struct stat sb;
    if(fstat(file_handler.fd, &sb) == -1)  perror("fstat");
    auto appendSize = sizeof(LatticeConfig) + 
            (2 * mrhs_colorspinor_length + gauge_length) * sizeof (complex<_FloatType>);

    ftruncate(file_handler.fd, sb.st_size + appendSize);
    fsync(file_handler.fd);

    printf("======fd = %d, sb.st_size = %ld, sb.st_size + appendSize = %ld\n", 
                    file_handler.fd, sb.st_size, sb.st_size + appendSize);
    
    void* origin_data = mmap(nullptr, sb.st_size + appendSize, PROT_WRITE | PROT_READ, 
                            MAP_SHARED, file_handler.fd, 0);
    void* data = origin_data;
    if (data == MAP_FAILED || data == nullptr) {
        throw std::runtime_error("MMAP failed\n");
    } 
    else {
        printf("MMAP correct\n");
    }

    // data前移动，跳过文件头
    data = static_cast<void *>(static_cast<char *>(data) + sizeof(LatticeConfig));

    // cout << "DEBUG, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>) = " << 
    //                 mrhs_colorspinor_length * sizeof(std::complex<_FloatType>) 
    //                 << " " << host_dst_data[0] << endl;

    if (lattice_config.m_src_rw_flag == ReadWriteFlag::RW_YES) {
        memcpy((char*)data, host_src_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>));
        // CHECK_CUDA(cudaMemcpy(data, device_src_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
        //             cudaMemcpyDeviceToHost));
        data = static_cast<void *>(static_cast<std::complex<_FloatType> *>(data) + mrhs_colorspinor_length);  
    }
    // 读取dst数据
    if (lattice_config.m_dst_rw_flag == ReadWriteFlag::RW_YES) {

        memcpy((char*)data, host_dst_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>));
        // CHECK_CUDA(cudaMemcpy(data, device_dst_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
        //             cudaMemcpyDeviceToHost));
        data = static_cast<void *>(static_cast<std::complex<_FloatType> *>(data) + mrhs_colorspinor_length);
    }
    // 读取gauge数据
    if (lattice_config.m_gauge_rw_flag == ReadWriteFlag::RW_YES) {
        memcpy((char*)data, host_gauge, gauge_length * sizeof(std::complex<_FloatType>));
        // CHECK_CUDA(cudaMemcpy(data, device_gauge_data, gauge_length * sizeof(std::complex<_FloatType>),
        //             cudaMemcpyDeviceToHost));
    }
    if((msync((void*)origin_data, sb.st_size + appendSize, MS_SYNC)) == -1)  perror("msync");
    if((munmap((void *)origin_data, sb.st_size + appendSize)) == -1)  perror("munmap\n");

    // write over
    delete[] host_dst_data;
    delete[] host_src_data;
    delete[] host_gauge;
}

template void read_from_file(
        std::complex<double> *device_src_data,       // 数据指针
        std::complex<double> *device_dst_data,       // 数据指针
        std::complex<double> *device_gauge_data,     // 数据指针
        const char *file_path                       // 文件路径
    );  

template void write_to_file(
        std::complex<double> *device_src_data,       // 数据指针
        std::complex<double> *device_dst_data,       // 数据指针
        std::complex<double> *device_gauge_data,     // 数据指针
        const char *file_path,                       // 文件路径
        LatticeConfig& lattice_config
    );  