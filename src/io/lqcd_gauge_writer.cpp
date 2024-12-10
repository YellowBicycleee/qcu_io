#include "io/lqcd_read_write.h"
#include <iostream>
#include <complex>
#include <H5Cpp.h>
#include <cassert>
#include <exception>

namespace qcu::io {
template <typename Real_>    
void GaugeWriter<Real_>::write(std::string file_path, std::vector<int> dims, qcu::io::Gauge4Dim<std::complex<Real_>>& gauge) {
    assert(mpi_desc_.data[T_DIM] > 0 && mpi_desc_.data[Z_DIM] > 0 && mpi_desc_.data[Y_DIM] > 0 && mpi_desc_.data[X_DIM] > 0);
    try  {
        H5::PredType storage_data_type = H5::PredType::NATIVE_DOUBLE;
        if constexpr (std::is_same_v<Real_, double>) {
            storage_data_type = H5::PredType::NATIVE_DOUBLE;
        } else if constexpr (std::is_same_v<Real_, float>) {
            storage_data_type = H5::PredType::NATIVE_FLOAT;
        } else {
            throw std::runtime_error("Unsupported data type, Only double and float are supported");
        }
        
        // vector [Nd, Lt, Lz, Ly, Lx, Nc, Nc]
        const size_t Nd = dims[0];
        const size_t Lt = dims[1];
        const size_t Lz = dims[2];
        const size_t Ly = dims[3];
        const size_t Lx = dims[4];
        const size_t Nc = dims[5];

        // const int Nc_double = dims[6];

        const int local_lt = Lt / mpi_desc_.data[T_DIM];
        const int local_lz = Lz / mpi_desc_.data[Z_DIM];
        const int local_ly = Ly / mpi_desc_.data[Y_DIM];
        const int local_lx = Lx / mpi_desc_.data[X_DIM];

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

        qcu::FourDimCoordinate data_offset;
        data_offset.data[T_DIM] = mpi_coord.data[T_DIM] * dims[1];
        data_offset.data[Z_DIM] = mpi_coord.data[Z_DIM] * dims[2];
        data_offset.data[Y_DIM] = mpi_coord.data[Y_DIM] * dims[3];
        data_offset.data[X_DIM] = mpi_coord.data[X_DIM] * dims[4];

        // parallel write file  
        {
            // set parallel access property
            H5::FileAccPropList plist;
            plist.copy(H5::FileAccPropList::DEFAULT);
            H5Pset_fapl_mpio(plist.getId(), MPI_COMM_WORLD, MPI_INFO_NULL);

            // create file
            H5::H5File file(file_path, H5F_ACC_TRUNC, H5::FileCreatPropList::DEFAULT, plist);

            // create global data space
            std::vector<hsize_t> dims = { Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2};
            H5::DataSpace filespace(dims.size(), dims.data());

            // create dataset
            H5::DataSet dataset = file.createDataSet(dataset_name, storage_data_type, filespace);

            // set local data space
            std::vector<hsize_t> local_dims = { 
                static_cast<hsize_t>(Nd), 
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
            filespace.selectHyperslab(H5S_SELECT_SET, local_dims.data(), offset.data());

            // set collective write property
            H5::DSetMemXferPropList xfer_plist;
            xfer_plist.copy(H5::DSetMemXferPropList::DEFAULT);
            H5Pset_dxpl_mpio(xfer_plist.getId(), H5FD_MPIO_COLLECTIVE);

            // write data
            dataset.write(gauge.data_ptr(), storage_data_type, memspace, filespace, xfer_plist);
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

template class GaugeWriter<double>;
template class GaugeWriter<float>;
} // namespace qcu::io