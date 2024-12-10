#include "io/lqcd_read_write.h"
#include <iostream>
#include <complex>
#include <H5Cpp.h>
#include <cassert>
#include <exception>
#include <iomanip>

namespace qcu::io {
template <typename Real_>    
void GaugeReader<Real_>::read(std::string file_path, std::vector<int> dims_required, qcu::io::Gauge4Dim<std::complex<Real_>>& gauge) {
    assert(mpi_desc_.data[T_DIM] > 0 && mpi_desc_.data[Z_DIM] > 0 && mpi_desc_.data[Y_DIM] > 0 && mpi_desc_.data[X_DIM] > 0);
    try {
        H5::PredType storage_data_type = H5::PredType::NATIVE_DOUBLE;
        if constexpr (std::is_same_v<Real_, double>) {
            storage_data_type = H5::PredType::NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<Real_, float>) {
            storage_data_type = H5::PredType::NATIVE_FLOAT;
        } else {
            throw std::runtime_error("Unsupported data type, Only double and float are supported");
        }

        const std::string dataset_name = "LatticeMatrix";

        qcu::FourDimCoordinate mpi_coord;
        
        int remainder;
        // T
        mpi_coord.data[T_DIM] = mpi_rank_ / (mpi_desc_.data[X_DIM] * mpi_desc_.data[Y_DIM] * mpi_desc_.data[Z_DIM]);
        remainder = mpi_rank_ % (mpi_desc_.data[X_DIM] * mpi_desc_.data[Y_DIM] * mpi_desc_.data[Z_DIM]); // t
        
        // Z
        mpi_coord.data[Z_DIM] = remainder / (mpi_desc_.data[X_DIM] * mpi_desc_.data[Y_DIM]);
        remainder = remainder % (mpi_desc_.data[X_DIM] * mpi_desc_.data[Y_DIM]); // z
        
        // Y
        mpi_coord.data[Y_DIM] = remainder / mpi_desc_.data[X_DIM];
        remainder = remainder % mpi_desc_.data[X_DIM]; // y
        
        // X
        mpi_coord.data[X_DIM] = remainder; // x

        // parallel write file  
        {
            // set parallel access property
            H5::FileAccPropList plist;
            plist.copy(H5::FileAccPropList::DEFAULT);
            H5Pset_fapl_mpio(plist.getId(), MPI_COMM_WORLD, MPI_INFO_NULL);

            // open HDF5 file
            H5::H5File file(file_path, H5F_ACC_RDONLY, H5::FileCreatPropList::DEFAULT, plist);
            H5::DataSet dataset = file.openDataSet(dataset_name);

            // get data space and dimension information
            H5::DataSpace dataspace = dataset.getSpace();
            const int ndims = dataspace.getSimpleExtentNdims();
            std::vector<hsize_t> dims(ndims);
            dataspace.getSimpleExtentDims(dims.data(), nullptr);

            // CHECK dims
            if (dims.size() != dims_required.size()) {
                throw std::runtime_error("Dimension mismatch");
            }
            std::string tags[] = {"Nd", "Nt", "Nz", "Ny", "Nx", "Nc", "Nc * 2"};
            for (int i = 0; i < dims_required.size(); ++i) {
                if (dims[i] != dims_required[i]) {
                    
                    std::ostringstream error_msg;
                    error_msg << "." << std::string(32, '-') << ".\n";
                    error_msg << "|" << std::setw(6) << "" 
                              << " | "
                              << std::setw(10) << std::left << "IN_FILE"
                              << " | "
                              << std::setw(10) << std::left << "REQUIRED"
                              << "|\n";
                    error_msg << "|" << std::string(32, '-') << "|\n";
                    for (int j = 0; j < dims_required.size(); ++j) {
                        error_msg << "|" << std::setw(6) << std::left << tags[j]
                                  << " | "
                                  << std::setw(10) << std::left << dims[j]
                                  << " | "
                                  << std::setw(10)  << std::left << dims_required[j] << "|\n";
                    }
                    error_msg << "." << std::string(32, '-') << ".\n";
                    throw std::runtime_error("Dimension mismatch \n" + error_msg.str());
                }
            }


            // calculate local size
            const size_t local_lt = dims[1] / mpi_desc_.data[T_DIM];
            const size_t local_lz = dims[2] / mpi_desc_.data[Z_DIM];
            const size_t local_ly = dims[3] / mpi_desc_.data[Y_DIM];
            const size_t local_lx = dims[4] / mpi_desc_.data[X_DIM];
            const size_t Nc = dims[5];

            // calculate data offset
            qcu::FourDimCoordinate data_offset;
            data_offset.data[T_DIM] = mpi_coord.data[T_DIM] * local_lt;
            data_offset.data[Z_DIM] = mpi_coord.data[Z_DIM] * local_lz;
            data_offset.data[Y_DIM] = mpi_coord.data[Y_DIM] * local_ly;
            data_offset.data[X_DIM] = mpi_coord.data[X_DIM] * local_lx;

            // create local array
            qcu::io::Gauge4Dim<std::complex<Real_>> local_gauge(local_lt, local_lz, local_ly, local_lx, Nc);

            // set local data space
            std::vector<hsize_t> local_dims = {
                static_cast<hsize_t>(local_gauge.get_Ndim()), 
                static_cast<hsize_t>(local_lt), 
                static_cast<hsize_t>(local_lz), 
                static_cast<hsize_t>(local_ly), 
                static_cast<hsize_t>(local_lx), 
                static_cast<hsize_t>(Nc), 
                static_cast<hsize_t>(Nc * 2)
            };
            std::vector<hsize_t> offset = {
                0, 
                static_cast<hsize_t>(data_offset.data[T_DIM]), 
                static_cast<hsize_t>(data_offset.data[Z_DIM]), 
                static_cast<hsize_t>(data_offset.data[Y_DIM]), 
                static_cast<hsize_t>(data_offset.data[X_DIM]), 
                0, 
                0
            };

            H5::DataSpace memspace(local_dims.size(), local_dims.data());
            dataspace.selectHyperslab(H5S_SELECT_SET, local_dims.data(), offset.data());

            // set collective read property
            H5::DSetMemXferPropList xfer_plist;
            xfer_plist.copy(H5::DSetMemXferPropList::DEFAULT);
            H5Pset_dxpl_mpio(xfer_plist.getId(), H5FD_MPIO_COLLECTIVE);

            // read data
            dataset.read(reinterpret_cast<Real_*>(local_gauge.data_ptr()), storage_data_type, memspace, dataspace, xfer_plist);

            gauge = std::move(local_gauge);
        }
    } catch (const H5::Exception& e) {
        if (mpi_rank_ == 0) {
            std::cerr << "HDF5 exception: Error," << e.getCDetailMsg() 
                      << "in file " << __FILE__ << " at line " << __LINE__ << '\n';
        }
        MPI_Finalize();
        exit(-1);
    } catch (const std::exception& e) {
        if (mpi_rank_ == 0) {
            std::cerr << "Error: " << e.what() << 
                      "in file " << __FILE__ << " at line " << __LINE__ << '\n';
        }
        MPI_Finalize();
        exit(-1);
    }

}

template class GaugeReader<double>;
template class GaugeReader<float>;
} // namespace qcu::io