#include "io/lqcd_read_write.h"
#include <iostream>
#include <complex>
#include <H5Cpp.h>
#include <cassert>
#include <exception>
#include <iomanip>

namespace qcu::io {

template <typename Real_>    
void GaugeReader<Real_>::read(std::string file_path, qcu::io::GaugeStorage<std::complex<Real_>>& gauge_in) {
    int num_dims = mpi_desc_.size();
    int gauge_dims = gauge_in.get_dim_num();
    assert(num_dims == gauge_dims);
    for (int i = 0; i < num_dims; ++i) {
        assert(mpi_desc_[i] > 0);
    }

    try {
        hid_t complex_id = H5Tcreate (H5T_COMPOUND, 2 * sizeof(Real_));
        if constexpr (std::is_same_v<Real_, double>) {  
            H5Tinsert (complex_id, "r", 0, H5T_NATIVE_DOUBLE);
            H5Tinsert (complex_id, "i", sizeof(Real_), H5T_NATIVE_DOUBLE);
        } else if constexpr (std::is_same_v<Real_, float>) {
            H5Tinsert (complex_id, "r", 0, H5T_NATIVE_FLOAT);
            H5Tinsert (complex_id, "i", sizeof(Real_), H5T_NATIVE_FLOAT);
        } else {
            throw std::runtime_error("Unsupported data type, Only double and float are supported");
        }

        const size_t Nd = num_dims;
        const size_t Nc = gauge_in.get_n_color();

        // const int Nc_double = dims[6];
        std::vector<int> lattice_total_dims = gauge_in.get_global_lattice_desc();
        std::vector<int> lattice_local_dims(num_dims, 0);
        for (int i = 0; i < num_dims; ++i) {
            lattice_local_dims[i] = lattice_total_dims[i] / mpi_desc_[i];
        }

        const std::string dataset_name = "LatticeMatrix";

        // qcu::FourDimCoordinate mpi_coord;
        std::vector<int> mpi_coord(num_dims, 0);
        int remainder = mpi_rank_;
        for (int i = num_dims - 1; i >= 0; --i) {
            mpi_coord[i] = remainder % mpi_desc_[i];
            remainder = remainder / mpi_desc_[i];
        }

        std::vector<int> data_offset(num_dims, 0);
        for (int i = 0; i < num_dims; ++i) {
            data_offset[i] = mpi_coord[i] * lattice_local_dims[i];
        }


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

            std::string tags[] = {"Nd", "Nt", "Nz", "Ny", "Nx", "Nc", "Nc * 2"};
            for (int i = 0; i < dims.size(); ++i) {
                if (dims[i] != lattice_total_dims[i]) {
                    
                    std::ostringstream error_msg;
                    error_msg << "." << std::string(32, '-') << ".\n";
                    error_msg << "|" << std::setw(6) << "" 
                        << " | "
                        << std::setw(10) << std::left << "IN_FILE"
                        << " | "
                        << std::setw(10) << std::left << "REQUIRED"
                        << "|\n";
                    error_msg << "|" << std::string(32, '-') << "|\n";
                    for (int j = 0; j < dims.size(); ++j) {
                        error_msg << "|" << std::setw(6) << std::left << tags[j]
                            << " | "
                            << std::setw(10) << std::left << dims[j]
                            << " | "
                            << std::setw(10)  << std::left << lattice_total_dims[j] << "|\n";
                    }
                    error_msg << "." << std::string(32, '-') << ".\n";
                    throw std::runtime_error("Dimension mismatch \n" + error_msg.str());
                }
            }

            const size_t Nc = dims[dims.size() - 1];
            // set local data space
            std::vector<hsize_t> local_dims;
            local_dims.push_back(Nd);
            for (int i = 0; i < num_dims; ++i) {
                local_dims.push_back(lattice_local_dims[i]);
            }
            local_dims.push_back(Nc);
            local_dims.push_back(Nc);
            std::vector<hsize_t> offset;
            offset.push_back(0);
            for (int i = 0; i < num_dims; ++i) {
                offset.push_back(data_offset[i]);
            }
            offset.push_back(0);
            offset.push_back(0);

            H5::DataSpace memspace(local_dims.size(), local_dims.data());
            dataspace.selectHyperslab(H5S_SELECT_SET, local_dims.data(), offset.data());

            // set collective read property
            H5::DSetMemXferPropList xfer_plist;
            xfer_plist.copy(H5::DSetMemXferPropList::DEFAULT);
            H5Pset_dxpl_mpio(xfer_plist.getId(), H5FD_MPIO_COLLECTIVE);

            // read data
            dataset.read(reinterpret_cast<Real_*>(gauge_in.data_ptr()), complex_id, memspace, dataspace, xfer_plist);

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