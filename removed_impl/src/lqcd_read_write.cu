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
#include <cstring>
#include <cuda_runtime_api.h>
using std::complex;
using std::cout;
using std::endl;

template <typename _FloatType>
static void write_to_file_memcpy_GPU (
        void* file_ptr,                           // 文件映射指针
        std::complex<_FloatType> *device_src_data,       // 数据指针
        std::complex<_FloatType> *device_dst_data,       // 数据指针
        std::complex<_FloatType> *device_gauge_data,      // 数据指针
        const QcuHeader& lattice_config
    )
{
    auto mrhs_colorspinor_length = lattice_config.MrhsColorSpinorLength();
    auto gauge_length = lattice_config.GaugeLength();

    if (lattice_config.m_io_position & IOPosition::IO_FERMION_IN) {
        CHECK_CUDA(cudaMemcpy(file_ptr, device_src_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyDeviceToHost));
        file_ptr = static_cast<void *>(static_cast<std::complex<_FloatType> *>(file_ptr) + mrhs_colorspinor_length);  
    }
    // 写入dst数据
    if (lattice_config.m_io_position & IOPosition::IO_FERMION_OUT) {
        CHECK_CUDA(cudaMemcpy(file_ptr, device_dst_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyDeviceToHost));
        file_ptr = static_cast<void *>(static_cast<std::complex<_FloatType> *>(file_ptr) + mrhs_colorspinor_length);
    }
    // 写入gauge数据
    if (lattice_config.m_io_position & IOPosition::IO_GAUGE) {
        CHECK_CUDA(cudaMemcpy(file_ptr, device_gauge_data, gauge_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyDeviceToHost));
    }
}

template <typename _FloatType>
static void write_to_file_memcpy_CPU (
        void* file_ptr,                                // 文件映射指针
        std::complex<_FloatType> *host_src_data,       // 数据指针
        std::complex<_FloatType> *host_dst_data,       // 数据指针
        std::complex<_FloatType> *host_gauge_data,     // 数据指针
        const QcuHeader& lattice_config
    )
{
    auto mrhs_colorspinor_length = lattice_config.MrhsColorSpinorLength();
    auto gauge_length = lattice_config.GaugeLength();

    if (lattice_config.m_io_position & IOPosition::IO_FERMION_IN) {
        memcpy(file_ptr, host_src_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>));
        file_ptr = static_cast<void *>(static_cast<std::complex<_FloatType> *>(file_ptr) + mrhs_colorspinor_length);  
    }
    // 写入dst数据
    if (lattice_config.m_io_position & IOPosition::IO_FERMION_OUT) {
        memcpy(file_ptr, host_dst_data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>));
        file_ptr = static_cast<void *>(static_cast<std::complex<_FloatType> *>(file_ptr) + mrhs_colorspinor_length);
    }
    // 写入gauge数据
    if (lattice_config.m_io_position & IOPosition::IO_GAUGE) {
        memcpy(file_ptr, host_gauge_data, gauge_length * sizeof(std::complex<_FloatType>));
    }
}


template <typename _FloatType>
void read_from_file (
        std::complex<_FloatType> *device_src_data,       // 数据指针
        std::complex<_FloatType> *device_dst_data,       // 数据指针
        std::complex<_FloatType> *device_gauge_data,     // 数据指针
        const std::string& file_path,                    // 文件路径
        QcuHeader& lattice_config_input
    ) 
{
    // 打开文件
    LatticeIOHandler file_handler(file_path, LatticeIOHandler::QCU_READ_MODE);
    struct stat st;
    if(fstat(file_handler.fd, &st) == -1)  {
        perror("fstat");
    }
    auto file_size = st.st_size;

    // mmap 映射
    void* origin_data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, file_handler.fd, 0);
    void* data = origin_data;
    if (data == MAP_FAILED || data == nullptr) {
        throw std::runtime_error("MMAP failed\n");
    } 
    else {
        printf("MMAP correct\n");
    }

    // 读取文件头
    QcuHeader lattice_config_file;
    memcpy(&lattice_config_file, data, sizeof(QcuHeader));
    auto mrhs_colorspinor_length = lattice_config_file.MrhsColorSpinorLength();
    auto gauge_length = lattice_config_file.GaugeLength();
    // 检查mpi参数、lattice参数和Nc，mIn参数是否一致
    if (!lattice_config_input.check_config(lattice_config_file)) {
        throw std::runtime_error("input config is not the same as file config\n");
    } else {
        fprintf(stdout, "terminal paramters matched\n");
    }
    // 接下来拷贝文件头的剩余不检查部分
    lattice_config_input.copy_info(lattice_config_file);

    // data前移动，跳过文件头
    data = static_cast<void *>(static_cast<char *>(data) + sizeof(QcuHeader));
    if (lattice_config_file.m_io_position & IOPosition::IO_FERMION_IN) {
        CHECK_CUDA(cudaMemcpy(device_src_data, data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyHostToDevice));
        data = static_cast<void *>(static_cast<std::complex<_FloatType> *>(data) + mrhs_colorspinor_length);
    }
    // 读取dst数据
    if (lattice_config_file.m_io_position & IOPosition::IO_FERMION_OUT) {
        CHECK_CUDA(cudaMemcpy(device_dst_data, data, mrhs_colorspinor_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyHostToDevice));
        data = static_cast<void *>(static_cast<std::complex<_FloatType> *>(data) + mrhs_colorspinor_length);
    }
    // 读取gauge数据
    if (lattice_config_file.m_io_position & IOPosition::IO_GAUGE) {
        CHECK_CUDA(cudaMemcpy(device_gauge_data, data, gauge_length * sizeof(std::complex<_FloatType>),
                    cudaMemcpyHostToDevice));
    }
    // read over
    if((msync((void*)origin_data,   file_size, MS_SYNC)) == -1)  { perror("msync");   }
    if((munmap((void *)origin_data, file_size         )) == -1)  { perror("munmap\n");}
}


template <typename _FloatType>
void write_to_file (
        std::complex<_FloatType> *device_src_data,       // 数据指针
        std::complex<_FloatType> *device_dst_data,       // 数据指针
        std::complex<_FloatType> *device_gauge_data,     // 数据指针
        const std::string& file_path,                    // 文件路径
        QcuHeader& lattice_config
    )
{
    auto mrhs_colorspinor_length = lattice_config.MrhsColorSpinorLength();
    auto gauge_length = lattice_config.GaugeLength();

    // 打开文件
    LatticeIOHandler file_handler(file_path, LatticeIOHandler::QCU_READ_WRITE_CREATE_MODE);

    auto newFileSize = sizeof(QcuHeader) + 
            (2 * mrhs_colorspinor_length + gauge_length) * sizeof (complex<_FloatType>);

    ftruncate(file_handler.fd, newFileSize);
    fsync(file_handler.fd);

    printf("===== DEBUG: fd = %d, newfilesize = %ld\n", file_handler.fd, newFileSize);
    void* origin_data = mmap(nullptr, newFileSize, PROT_WRITE | PROT_READ, MAP_SHARED, file_handler.fd, 0);
    void* data = origin_data;
    if (data == MAP_FAILED || data == nullptr) {
        throw std::runtime_error("MMAP failed\n");
    } 
    else {
        printf("MMAP correct\n");
    }

    // 写文件头
    memcpy(data, &lattice_config, sizeof(QcuHeader));
    // data前移动，跳过文件头
    data = static_cast<void *>(static_cast<char *>(data) + sizeof(QcuHeader));

    write_to_file_memcpy_GPU<_FloatType>(data, device_src_data, device_dst_data, device_gauge_data, lattice_config);

    if((msync((void*)origin_data, newFileSize, MS_SYNC)) == -1)  perror("msync");
    if((munmap((void *)origin_data, newFileSize)) == -1)  perror("munmap\n");
}

// 目前只实例化了双精度实例
template void read_from_file(
        std::complex<double> *device_src_data,       // 数据指针
        std::complex<double> *device_dst_data,       // 数据指针
        std::complex<double> *device_gauge_data,     // 数据指针
        const std::string& file_path,                // 文件路径
        QcuHeader& lattice_config_input          // 命令行文件头，和文件存储文件头进行对比
    );  

template void write_to_file(
        std::complex<double> *device_src_data,       // 数据指针
        std::complex<double> *device_dst_data,       // 数据指针
        std::complex<double> *device_gauge_data,     // 数据指针
        const std::string& file_path,                // 文件路径
        QcuHeader& lattice_config
    );