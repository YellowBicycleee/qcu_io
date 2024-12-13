#pragma once

#include <stdexcept>
#include <complex>
#include "lattice_desc.h"
#include "qcd/gauge.h"
#include <string>
#include <vector>

namespace qcu::io {

template <typename Real_ = double>
class GaugeWriter{
public:
    GaugeWriter (const int mpi_rank, const qcu::FourDimDesc& mpi_desc) : mpi_rank_(mpi_rank), mpi_desc_(mpi_desc) {}
    void write(std::string file_path, std::vector<int> dims, qcu::io::Gauge4Dim<std::complex<Real_>>& gauge_in);
protected:
    const int mpi_rank_;
    const qcu::FourDimDesc& mpi_desc_;
};

template <typename Real_ = double>
class GaugeReader{
public:
    GaugeReader (const int mpi_rank, const qcu::FourDimDesc& mpi_desc) : mpi_rank_(mpi_rank), mpi_desc_(mpi_desc) {}
    void read(std::string file_path, std::vector<int> dims, qcu::io::Gauge4Dim<std::complex<Real_>>& gauge_out);
protected:
    const int mpi_rank_;
    const qcu::FourDimDesc& mpi_desc_;
};

} // namespace qcu::io
