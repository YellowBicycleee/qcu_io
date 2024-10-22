#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <complex>
#include "lattice_desc.h"
#include "qcu_mpi_desc.h"
#include <string>
struct LatticeIOHandler {
    static constexpr uint32_t QCU_READ_MODE              = O_RDONLY;
    static constexpr uint32_t QCU_WRITE_CREATE_MODE      = O_WRONLY | O_CREAT;
    static constexpr uint32_t QCU_READ_WRITE_CREATE_MODE = O_RDWR | O_CREAT;
    static constexpr uint32_t QCU_READ_WRITE_MODE        = O_RDWR;
    // static constexpr uint32_t QCU_WRITE_CREATE_MODE = O_WRONLY | O_CREAT;

    bool file_opened = false;
    int fd = -1;
    const char *file_path_ = nullptr;

    LatticeIOHandler() = default;
    LatticeIOHandler(const std::string& file_path, const int32_t file_open_mode) 
                            : file_path_(file_path.c_str()) 
    {
        if (    file_open_mode != QCU_READ_MODE 
            &&  file_open_mode != QCU_WRITE_CREATE_MODE 
            &&  file_open_mode != QCU_READ_WRITE_CREATE_MODE
            &&  file_open_mode != QCU_READ_WRITE_MODE) 
        {
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

struct QcuHeader
{
    unsigned char  m_magic[MAGIC_BYTE_SIZE] = {0x7f, 'Q', 'C', 'U'};
    DataFormat       m_data_format       = DataFormat::QUDA_FORMAT;
    QcuPrecision m_storage_precision = QcuPrecision::kPrecisionDouble; // 默认double
    MrhsShuffled     m_mrhs_shuffled     = MrhsShuffled::MRHS_SHUFFLED_NO;
    StorageType      m_storage_type      = StorageType::TYPE_UNKNOWN;
    Latt_Desc        m_lattice_desc;

    int32_t m_Nc;
    union {
        int32_t m_MInput;   // m lhs
        int32_t m_Ngauge;   // 组态个数
    };

    union {
        int32_t m_Ns = 4;
        int32_t m_Nd;
    };

    // 检查硬参数是否一致
    bool check_config (const QcuHeader& rhs) {
        return     this->m_lattice_desc == rhs.m_lattice_desc
                && this->m_Nc           == rhs.m_Nc
                && this->m_MInput       == rhs.m_MInput;
    }
    // copy check_config检查范围外的参数
    void copy_info (const QcuHeader& rhs) {
        m_data_format       = rhs.m_data_format;
        m_storage_precision = rhs.m_storage_precision;
        m_mrhs_shuffled     = rhs.m_mrhs_shuffled;
        m_Ns                = rhs.m_Ns;
        m_Nd                = rhs.m_Nd;
    }

    int32_t volume4D() const {
        return m_lattice_desc.data[X_DIM] * m_lattice_desc.data[Y_DIM] * 
               m_lattice_desc.data[Z_DIM] * m_lattice_desc.data[T_DIM];
    }

    int32_t SingleColorSpinorSiteLength () const {
        return m_Ns * m_Nc;
    }
    int32_t MrhsColorSpinorSiteLength () const {
        return m_MInput * SingleColorSpinorSiteLength();
    }
    int32_t GaugeSiteLength () const {
        return m_Nc * m_Nc;
    }

    int32_t GaugeLength () const {
        return m_Nd * volume4D() * GaugeSiteLength(); 
    }

    int32_t SingleColorSpinorLength () const {
        return volume4D() * SingleColorSpinorSiteLength();
    }

    int32_t MrhsColorSpinorLength () const {
        return m_MInput * SingleColorSpinorLength();
    }
};

template <typename _FloatType = double>
class GaugeWriter {
    int                   gauge_pos_    = 0;  // 第几个组态
    size_t                file_size_ = 0;
    const MPI_Desc&       mpi_desc_;
    const MPI_Coordinate& mpi_coord_;
    const QcuHeader&      header_;
    void*                 disk_mapped_ptr_ = nullptr;
    void write_gauge_kernel (std::complex<_FloatType>* disk_gauge, std::complex<_FloatType>* memory_gauge);
public:
    GaugeWriter(const std::string& file_path, const QcuHeader& header, const MPI_Desc& mpi_desc, const MPI_Coordinate& mpi_coord);
    ~GaugeWriter() noexcept;
    void write_gauge (std::complex<_FloatType>* memory_gauge);
};

template <typename _FloatType = double>
class GaugeReader {
    size_t                  file_size_ = 0;
    const MPI_Desc&         mpi_desc_;
    const MPI_Coordinate&   mpi_coord_;
    QcuHeader&              header_;
    void*                   disk_mapped_ptr_ = nullptr;
    const Latt_Desc &       lattice_desc_;

    void read_gauge_kernel (std::complex<_FloatType>* disk_gauge, std::complex<_FloatType>* memory_gauge);
public:
    GaugeReader(const std::string& file_path, QcuHeader& header, const MPI_Desc& mpi_desc, 
                const MPI_Coordinate& mpi_coord, const Latt_Desc& lattice_desc);
    ~GaugeReader() noexcept;
    void read_gauge (std::complex<_FloatType>* memory_gauge, int gauge_pos);
};

template <typename _FloatType = double>
class FermionWriter {
    int                   fermion_pos_  = 0;  // 第几个fermion
    size_t                file_size_ = 0;
    const MPI_Desc&       mpi_desc_;
    const MPI_Coordinate& mpi_coord_;
    const QcuHeader&      header_;
    void*                 disk_mapped_ptr_ = nullptr;
    void write_fermion_kernel (std::complex<_FloatType>* disk_fermion, std::complex<_FloatType>* memory_fermion);
public:
    FermionWriter(const std::string& file_path, const QcuHeader& header, const MPI_Desc& mpi_desc, const MPI_Coordinate& mpi_coord);
    ~FermionWriter() noexcept;
    void write_fermion (std::complex<_FloatType>* memory_fermion);
};

template <typename _FloatType = double>
class FermionReader {
    size_t                  file_size_ = 0;
    const MPI_Desc&         mpi_desc_;
    const MPI_Coordinate&   mpi_coord_;
    QcuHeader&              header_;
    void*                   disk_mapped_ptr_ = nullptr;

    void read_fermion_kernel (std::complex<_FloatType>* disk_fermion, std::complex<_FloatType>* memory_fermion);
public:
    FermionReader(const std::string& file_path, QcuHeader& header, const MPI_Desc& mpi_desc, const MPI_Coordinate& mpi_coord);
    ~FermionReader() noexcept;
    void read_fermion (std::complex<_FloatType>* memory_fermion, int fermion_pos);
};