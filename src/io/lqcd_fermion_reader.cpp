#include "lqcd_read_write.h"
#include <cstdio>
#include <complex>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring>

template <typename _FloatType>  
void FermionReader<_FloatType>::read_fermion_kernel (std::complex<_FloatType>* memory_fermion, 
                                                     std::complex<_FloatType>* disk_fermion) 
{
    auto fermion_length = header_.SingleColorSpinorLength();
    memcpy(memory_fermion, disk_fermion, fermion_length * sizeof(std::complex<_FloatType>));
}

template <typename _FloatType>
FermionReader<_FloatType>::FermionReader(const std::string& file_path, 
                                         QcuHeader& header,
                                         const MPI_Desc& mpi_desc,
                                         const MPI_Coordinate& mpi_coord) 
          : header_(header),
            mpi_desc_(mpi_desc),
            mpi_coord_(mpi_coord)
{
    // 打开文件
    LatticeIOHandler file_handler(file_path, LatticeIOHandler::QCU_READ_MODE);
    struct stat st;
    if(fstat(file_handler.fd, &st) == -1)  {
        perror("fstat");
    }
    file_size_ = st.st_size;

    // mmap 映射
    disk_mapped_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, file_handler.fd, 0);
    if (disk_mapped_ptr_ == MAP_FAILED || disk_mapped_ptr_ == nullptr) {
        throw std::runtime_error("MMAP failed\n");
    } 

    // 读取文件头
    memcpy(&header_, disk_mapped_ptr_, sizeof(QcuHeader));

    if (header_.m_storage_type != StorageType::TYPE_FERMION) {
        if((msync((void*)disk_mapped_ptr_, file_size_, MS_SYNC)) == -1) { perror("msync");}
        if((munmap((void *)disk_mapped_ptr_, file_size_)) == -1)        { perror("munmap\n");}
        throw std::runtime_error("StorageType is not TYPE_FERMION\n");
    }
}

template <typename _FloatType>
FermionReader<_FloatType>::~FermionReader() noexcept 
{
    if (disk_mapped_ptr_ != nullptr) {
        if((msync((void*)disk_mapped_ptr_, file_size_, MS_SYNC)) == -1) { perror("msync");}
        if((munmap((void *)disk_mapped_ptr_, file_size_)) == -1)        { perror("munmap\n");}
    }
}

template <typename _FloatType>
void FermionReader<_FloatType>::read_fermion (std::complex<_FloatType>* memory_fermion, int fermion_pos)
{
    int fermion_length = header_.SingleColorSpinorLength();
    std::complex<_FloatType>* disk_fermion = 
        reinterpret_cast<std::complex<_FloatType>*>(static_cast<char*>(disk_mapped_ptr_) + sizeof(QcuHeader))
             + fermion_pos * fermion_length;
    read_fermion_kernel(memory_fermion, disk_fermion);
}

template class FermionReader<double>;
template class FermionReader<float>;