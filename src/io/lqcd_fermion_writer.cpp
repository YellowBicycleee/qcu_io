#include "lqcd_read_write.h"
#include <cstdio>
#include <complex>
#include <sys/mman.h>
#include <string>
#include <cstring>

template <typename _FloatType>
void FermionWriter<_FloatType>::write_fermion_kernel 
    (std::complex<_FloatType>* disk_fermion, std::complex<_FloatType>* memory_fermion)
{
    auto fermion_length = header_.SingleColorSpinorLength();
    memcpy(disk_fermion, memory_fermion, fermion_length * sizeof(std::complex<_FloatType>));
}

template <typename _FloatType>
FermionWriter<_FloatType>::FermionWriter 
    (   const std::string& file_path, 
        const QcuHeader& header, 
        const MPI_Desc& mpi_desc, 
        const MPI_Coordinate& mpi_coord
    ) 
    : header_(header),
      mpi_desc_(mpi_desc),
      mpi_coord_(mpi_coord)
{
    auto fermion_num    = header_.m_MInput;
    auto fermion_length = header_.SingleColorSpinorLength();

    // 打开文件, 写文件头预先留出文件长度
    LatticeIOHandler file_handler(file_path, LatticeIOHandler::QCU_READ_WRITE_CREATE_MODE);
    file_handler_ = file_handler;
    file_size_ = sizeof(QcuHeader) + (fermion_num * fermion_length) * sizeof (std::complex<_FloatType>);
    
    ftruncate(file_handler_.fd, file_size_);
    fsync(file_handler_.fd);
    disk_mapped_ptr_ = mmap(nullptr, file_size_, PROT_WRITE | PROT_READ, MAP_SHARED, file_handler_.fd, 0);
    if (disk_mapped_ptr_ == MAP_FAILED || disk_mapped_ptr_ == nullptr) {
        throw std::runtime_error("FermionWriter MMAP failed\n");
    }

    // 写文件头
    memcpy(disk_mapped_ptr_, &header, sizeof(QcuHeader));
}

template <typename _FloatType>
FermionWriter<_FloatType>::~FermionWriter() noexcept
{
    if (fermion_pos_ != header_.m_MInput) {
        fprintf(stderr, "WARNING: fermion_pos != header_.m_Nfermion, in file %s, line %d\n", __FILE__, __LINE__);
        fprintf(stderr, "fermion_pos = %d, header_.m_MInput = %d\n", fermion_pos_, header_.m_MInput);
    }
    if (disk_mapped_ptr_ != nullptr) {
        if((msync((void*)disk_mapped_ptr_, file_size_, MS_SYNC)) == -1) { perror("msync");}
        if((munmap((void *)disk_mapped_ptr_, file_size_)) == -1)        { perror("munmap\n");}
        close(file_handler_.fd);
    }
}

template <typename _FloatType>
void FermionWriter<_FloatType>::write_fermion (std::complex<_FloatType>* memory_fermion)
{
    int fermion_length = header_.SingleColorSpinorLength();
    std::complex<_FloatType>* disk_fermion = 
            reinterpret_cast<std::complex<_FloatType>*>(static_cast<char*>(disk_mapped_ptr_) + sizeof(QcuHeader)) 
            + fermion_pos_ * fermion_length;
    write_fermion_kernel(disk_fermion, memory_fermion);
    fermion_pos_++;
}

template class FermionWriter<double>;
template class FermionWriter<float>;