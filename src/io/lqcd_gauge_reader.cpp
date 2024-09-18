#include "lattice_desc.h"
#include "lqcd_read_write.h"
#include <cstdio>
#include <complex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

// 单进程实现
template <typename _FloatType>
void GaugeReader<_FloatType>::read_gauge_kernel (std::complex<_FloatType>* memory_gauge, 
                                                  std::complex<_FloatType>* disk_gauge) 
{
    auto gauge_length = header_.GaugeLength();
    memcpy(memory_gauge, disk_gauge, gauge_length * sizeof(std::complex<_FloatType>));
}

// 这个文件头是未初始化的
template <typename _FloatType>
GaugeReader<_FloatType>::GaugeReader(const std::string& file_path, 
                                     QcuHeader& header,
                                     const MPI_Desc& mpi_desc,
                                     const MPI_Coordinate& mpi_coord) 
          : header_(header),
            mpi_desc_(mpi_desc),
            mpi_coord_(mpi_coord)
{
    // 打开文件
    file_handler_ = LatticeIOHandler(file_path, LatticeIOHandler::QCU_READ_MODE);
    struct stat st;
    if(fstat(file_handler_.fd, &st) == -1)  {
        perror("fstat");
    }
    file_size_ = st.st_size;

    // mmap 映射
    disk_mapped_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, file_handler_.fd, 0);
    if (disk_mapped_ptr_ == MAP_FAILED || disk_mapped_ptr_ == nullptr) {
        throw std::runtime_error("MMAP failed\n");
    } 

    // 读取文件头
    memcpy(&header_, disk_mapped_ptr_, sizeof(QcuHeader));

    if (header_.m_storage_type != StorageType::TYPE_GAUGE) {
        close(file_handler_.fd);
        if((msync((void*)disk_mapped_ptr_, file_size_, MS_SYNC)) == -1) { perror("msync");}
        if((munmap((void *)disk_mapped_ptr_, file_size_)) == -1)        { perror("munmap\n");}
        throw std::runtime_error("StorageType is not TYPE_GAUGE\n");
    }
}

template <typename _FloatType>
GaugeReader<_FloatType>::~GaugeReader() noexcept 
{
    if (disk_mapped_ptr_ != nullptr) {
        if((msync((void*)disk_mapped_ptr_, file_size_, MS_SYNC)) == -1) { perror("msync");}
        if((munmap((void *)disk_mapped_ptr_, file_size_)) == -1)        { perror("munmap\n");}
        close(file_handler_.fd);
    }
}


template <typename _FloatType>
void GaugeReader<_FloatType>::read_gauge (std::complex<_FloatType>* memory_gauge, int gauge_pos) 
{
    auto gauge_num    = header_.m_Ngauge;
    auto gauge_length = header_.GaugeLength();

    if (gauge_pos >= gauge_num) {
        throw std::runtime_error("gauge_pos >= gauge_num");
    }

    // 定位起始写入位置
    std::complex<_FloatType>* disk_gauge 
            = reinterpret_cast<std::complex<_FloatType>*>(
                static_cast<char *>(disk_mapped_ptr_) + sizeof(QcuHeader)
              ) + gauge_pos * gauge_length;

    read_gauge_kernel(memory_gauge, disk_gauge);
}


template class GaugeReader<double>;
template class GaugeReader<float>;