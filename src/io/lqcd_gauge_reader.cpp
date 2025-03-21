#include "io/lqcd_read_write.h"
#include <iostream>
#include <complex>
#include <H5Cpp.h>
#include <cassert>
#include <exception>
#include <iomanip>
#include <numeric>

namespace qcu::io {
// 方向标签定义
const char high_dim_tag[] = {'X', 'Y', 'Z', 'T'};

template <typename Real_>    
void GaugeReader<Real_>::read(std::string file_path, qcu::io::GaugeStorage<std::complex<Real_>>& gauge_in) {
    int mpi_dims = mpi_desc_.size(); // mpi维度
    int gauge_dims = gauge_in.get_dim_num(); // 格子维度
    assert(mpi_dims == gauge_dims);
    for (int i = 0; i < gauge_dims; ++i) {
        assert(mpi_desc_[i] > 0);
    }
    
    // 计算MPI进程坐标
    std::vector<int> mpi_coord(gauge_dims, 0); 
    int remainder = mpi_rank_;
    for (int i = gauge_dims - 1; i >= 0; --i) {
        mpi_coord[i] = remainder % mpi_desc_[i];
        remainder = remainder / mpi_desc_[i];
    }

    // 计算格点大小
    std::vector<int> lattice_total_dims = gauge_in.get_global_lattice_desc();
    std::vector<int> lattice_local_dims(gauge_dims, 0);
    for (int i = 0; i < gauge_dims; ++i) {
        lattice_local_dims[i] = lattice_total_dims[i] / mpi_desc_[i];
    }

    // 计算每个进程的 时空维度的数据偏移 （其他维度都是0，不需要计算）
    std::vector<int> data_offset(gauge_dims, 0);
    for (int i = 0; i < gauge_dims; ++i) {
        data_offset[i] = mpi_coord[i] * lattice_local_dims[i];
    }

    const size_t Nd = gauge_dims;
    const size_t Nc = gauge_in.get_n_color();

    // 计算本地格点体积
    size_t local_volume = 1;
    for (int i = 0; i < gauge_dims; ++i) {
        local_volume *= lattice_local_dims[i];
    }
    
    // 预先计算HDF5相关的数据结构
    // 1. 创建本地数据空间维度
    std::vector<hsize_t> local_dims;
    for (int i = 0; i < gauge_dims; ++i) {
        local_dims.push_back(lattice_local_dims[gauge_dims - i - 1]);
    }
    local_dims.push_back(Nc);
    local_dims.push_back(Nc);
    
    // 2. 创建偏移量
    std::vector<hsize_t> offset;
    for (int i = 0; i < gauge_dims; ++i) {
        offset.push_back(data_offset[gauge_dims - i - 1]);
    }
    offset.push_back(0); // Nc
    offset.push_back(0); // Nc
    
    try {
        // 创建复数数据类型
        hid_t complex_id = H5Tcreate(H5T_COMPOUND, 2 * sizeof(Real_));
        if constexpr (std::is_same_v<Real_, double>) {  
            H5Tinsert(complex_id, "r", 0, H5T_NATIVE_DOUBLE);
            H5Tinsert(complex_id, "i", sizeof(Real_), H5T_NATIVE_DOUBLE);
        } else if constexpr (std::is_same_v<Real_, float>) {
            H5Tinsert(complex_id, "r", 0, H5T_NATIVE_FLOAT);
            H5Tinsert(complex_id, "i", sizeof(Real_), H5T_NATIVE_FLOAT);
        } else {
            throw std::runtime_error("Unsupported data type, Only double and float are supported");
        }
        
        // 设置并行访问属性
        H5::FileAccPropList plist;
        plist.copy(H5::FileAccPropList::DEFAULT);
        H5Pset_fapl_mpio(plist.getId(), MPI_COMM_WORLD, MPI_INFO_NULL);

        // 打开HDF5文件
        H5::H5File file(file_path, H5F_ACC_RDONLY, H5::FileCreatPropList::DEFAULT, plist);
        
        // 打开LatticeColorMatrix组
        H5::Group mainGroup = file.openGroup("LatticeColorMatrix");
        
        // 设置集体读取属性
        H5::DSetMemXferPropList xfer_plist;
        xfer_plist.copy(H5::DSetMemXferPropList::DEFAULT);
        H5Pset_dxpl_mpio(xfer_plist.getId(), H5FD_MPIO_COLLECTIVE);
        
        // 预先创建内存空间，对所有方向都是相同的
        H5::DataSpace memspace(local_dims.size(), local_dims.data());
        
        // 对每个方向读取数据
        for (int dir = 0; dir < Nd; ++dir) {
            // 确定方向名称
            std::string dir_name;
            if (dir < 4) {
                dir_name = std::string(1, high_dim_tag[dir]);
            } else {
                dir_name = std::to_string(dir);
            }
            
            // 打开对应方向的数据集
            H5::DataSet dataset = mainGroup.openDataSet(dir_name);
            
            // 获取文件中的数据空间和维度信息
            H5::DataSpace dataspace = dataset.getSpace();
            const int ndims = dataspace.getSimpleExtentNdims();
            
            // 检查维度 - 应该是 [L....L] [Nc] [Nc]
            assert(ndims == gauge_dims + 2); 
            
            std::vector<hsize_t> dims(ndims);
            dataspace.getSimpleExtentDims(dims.data(), nullptr);
            
            // 验证维度匹配
            for (int i = 0; i < gauge_dims; ++i) {
                if (dims[i] != lattice_total_dims[gauge_dims - i - 1]) {
                    std::ostringstream error_msg;
                    error_msg << "Dimension mismatch for direction " << dir_name;
                    throw std::runtime_error(error_msg.str());
                }
            }
            
            // 设置超块选择
            dataspace.selectHyperslab(H5S_SELECT_SET, local_dims.data(), offset.data());
            
            // 计算当前方向数据在内存中的偏移量
            size_t dir_offset = dir * local_volume * Nc * Nc;
            
            // 读取数据到指定偏移位置
            dataset.read(
                reinterpret_cast<Real_*>(gauge_in.data_ptr()) + dir_offset * 2, // *2因为复数占用2个Real_
                complex_id,
                memspace,
                dataspace,
                xfer_plist
            );
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