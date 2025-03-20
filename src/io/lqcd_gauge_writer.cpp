#include "io/lqcd_read_write.h"
#include <cstddef>
#include <iostream>
#include <complex>
#include <H5Cpp.h>
#include <cassert>
#include <exception>
#include <type_traits>
#include <vector>
#include <complex>
namespace qcu::io {
template <typename Real_>    
void GaugeWriter<Real_>::write(std::string file_path, qcu::io::GaugeStorage<std::complex<Real_>>& gauge_out) {
    int num_dims = mpi_desc_.size();
    int gauge_dims = gauge_out.get_dim_num();
    assert(num_dims == gauge_dims);
    for (int i = 0; i < num_dims; ++i) {
        assert(mpi_desc_[i] > 0);
    }
    
    try  {
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
        
        // vector [Nd, dims, Nc, Nc]
        const size_t Nd = num_dims;
        const size_t Nc = gauge_out.get_n_color();

        // const int Nc_double = dims[6];
        std::vector<int> lattice_total_dims = gauge_out.get_global_lattice_desc();
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

            // create file
            H5::H5File file(file_path, H5F_ACC_TRUNC, H5::FileCreatPropList::DEFAULT, plist);

            // create global data space
            std::vector<hsize_t> dims; //  = { Nd, Lt, Lz, Ly, Lx, Nc, Nc * 2};
            dims.push_back(Nd);
            for (int i = 0; i < num_dims; ++i) { // 注意反向
                dims.push_back(lattice_total_dims[gauge_dims - i - 1]);
            }
            dims.push_back(Nc);
            dims.push_back(Nc);
            H5::DataSpace filespace(dims.size(), dims.data());

            // create dataset
            H5::DataSet dataset = file.createDataSet(dataset_name, complex_id, filespace);

            // set local data space
            std::vector<hsize_t> local_dims;
            local_dims.push_back(Nd);
            for (int i = 0; i < num_dims; ++i) { // lattice_desc顺序为xyzt，序列化为tzyx
                local_dims.push_back(lattice_local_dims[gauge_dims - i - 1]);
            }
            local_dims.push_back(Nc);
            local_dims.push_back(Nc);

            std::vector<hsize_t> offset;
            offset.push_back(0);
            for (int i = 0; i < num_dims; ++i) {
                offset.push_back(data_offset[gauge_dims - i - 1]);
            }
            offset.push_back(0);
            offset.push_back(0);
            
            H5::DataSpace memspace(local_dims.size(), local_dims.data());
            filespace.selectHyperslab(H5S_SELECT_SET, local_dims.data(), offset.data());

            // set collective write property
            H5::DSetMemXferPropList xfer_plist;
            xfer_plist.copy(H5::DSetMemXferPropList::DEFAULT);
            H5Pset_dxpl_mpio(xfer_plist.getId(), H5FD_MPIO_COLLECTIVE);

            // write data
            dataset.write(gauge_out.data_ptr(), complex_id, memspace, filespace, xfer_plist);
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