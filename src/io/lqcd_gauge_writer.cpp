#include "lattice_desc.h"
#include "lqcd_read_write.h"
#include <cstdio>
#include <complex>
#include <sys/mman.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

// 单进程实现
template <typename _FloatType>
void GaugeWriter<_FloatType>::write_gauge_kernel (std::complex<_FloatType>* disk_gauge, 
                                                  std::complex<_FloatType>* memory_gauge) 
{
    auto gauge_length = header_.GaugeLength();
    memcpy(disk_gauge, memory_gauge, gauge_length * sizeof(std::complex<_FloatType>));
}

template <typename _FloatType>
GaugeWriter<_FloatType>::GaugeWriter(const std::string& file_path, 
                                     const QcuHeader& header,
                                     const MPI_Desc& mpi_desc,
                                     const MPI_Coordinate& mpi_coord
                                    ) 
          : header_(header),
            mpi_desc_(mpi_desc),
            mpi_coord_(mpi_coord)
{
    auto gauge_num    = header_.m_Ngauge;
    auto gauge_length = header_.GaugeLength();

    // 打开文件, 写文件头预先留出文件长度
    LatticeIOHandler file_handler(file_path, LatticeIOHandler::QCU_READ_WRITE_CREATE_MODE);
    file_handler = file_handler;
    file_size_ = sizeof(QcuHeader) + (gauge_num * gauge_length) * sizeof (std::complex<_FloatType>);
    
    std::cout << "file_path = " << file_path << ", gauge file_size_ = " << file_size_ << std::endl;
    int ret = ftruncate(file_handler.fd, file_size_);
    if (ret == -1) {
        perror("ftruncate");
    }
    ret = fsync(file_handler.fd);
    if (ret == -1) {
        perror("fsync");
    }

    disk_mapped_ptr_ = mmap(nullptr, file_size_, PROT_WRITE | PROT_READ, MAP_SHARED, file_handler.fd, 0);
    if (disk_mapped_ptr_ == MAP_FAILED || disk_mapped_ptr_ == nullptr) {
        throw std::runtime_error("GaugeWriter MMAP failed\n");
    }

    // 写文件头
    memcpy(disk_mapped_ptr_, &header, sizeof(QcuHeader));
}

template <typename _FloatType>
GaugeWriter<_FloatType>::~GaugeWriter() noexcept {
    if (gauge_pos_ != header_.m_Ngauge) {
        fprintf(stderr, "WARNING: gauge_pos != header_.m_Ngauge, in file %s, line %d\n", __FILE__, __LINE__);
        fprintf(stderr, "gauge_pos = %d, header_.m_Ngauge = %d\n", gauge_pos_, header_.m_Ngauge);
    }
    if (disk_mapped_ptr_ != nullptr) {
        if((msync((void*)disk_mapped_ptr_, file_size_, MS_SYNC)) == -1) { perror("msync");}
        if((munmap((void *)disk_mapped_ptr_, file_size_)) == -1)        { perror("munmap\n");}
    }
}


template <typename _FloatType>
void GaugeWriter<_FloatType>::write_gauge (std::complex<_FloatType>* memory_gauge) 
{
    auto gauge_num    = header_.m_Ngauge;
    auto gauge_length = header_.GaugeLength();

    if (gauge_pos_ >= gauge_num) {
        throw std::runtime_error("gauge_pos >= gauge_num");
    }

    // 定位起始写入位置
    std::complex<_FloatType>* disk_gauge 
            = reinterpret_cast<std::complex<_FloatType>*>(
                static_cast<char *>(disk_mapped_ptr_) + sizeof(QcuHeader)
              ) + gauge_pos_ * gauge_length;

    write_gauge_kernel(disk_gauge, memory_gauge);
    gauge_pos_++;
}

template class GaugeWriter<double>;
template class GaugeWriter<float>;