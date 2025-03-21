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
#include <numeric>

namespace qcu::io {
// 方向标签定义
const char high_dim_tag[] = {'X', 'Y', 'Z', 'T'};

template <typename Real_>    
void GaugeWriter<Real_>::write(std::string file_path, qcu::io::GaugeStorage<std::complex<Real_>>& gauge_out) {
    int num_dims = mpi_desc_.size();
    int gauge_dims = gauge_out.get_dim_num();
    assert(num_dims == gauge_dims);
    for (int i = 0; i < num_dims; ++i) {
        assert(mpi_desc_[i] > 0);
    }

    // 计算MPI进程坐标
    std::vector<int> mpi_coord(num_dims, 0);
    int remainder = mpi_rank_;
    for (int i = num_dims - 1; i >= 0; --i) {
        mpi_coord[i] = remainder % mpi_desc_[i];
        remainder = remainder / mpi_desc_[i];
    }
    
    // 计算格点大小
    std::vector<int> lattice_total_dims = gauge_out.get_global_lattice_desc();
    std::vector<int> lattice_local_dims(num_dims, 0);
    for (int i = 0; i < num_dims; ++i) {
        lattice_local_dims[i] = lattice_total_dims[i] / mpi_desc_[i];
    }
    
    // 计算数据偏移
    std::vector<int> data_offset(num_dims, 0);
    for (int i = 0; i < num_dims; ++i) {
        data_offset[i] = mpi_coord[i] * lattice_local_dims[i];
    }
    
    const size_t Nd = num_dims;
    const size_t Nc = gauge_out.get_n_color();

    // 计算本地格点体积
    size_t local_volume = 1;
    for (int i = 0; i < num_dims; ++i) {
        local_volume *= lattice_local_dims[i];
    }
    
    // 预先计算HDF5相关的数据结构
    // 1. 创建全局数据空间维度
    std::vector<hsize_t> global_dims;
    for (int i = 0; i < num_dims; ++i) {
        global_dims.push_back(lattice_total_dims[gauge_dims - i - 1]);
    }
    global_dims.push_back(Nc);
    global_dims.push_back(Nc);
    
    // 2. 创建本地数据空间维度
    std::vector<hsize_t> local_dims;
    for (int i = 0; i < num_dims; ++i) {
        local_dims.push_back(lattice_local_dims[gauge_dims - i - 1]);
    }
    local_dims.push_back(Nc);
    local_dims.push_back(Nc);
    
    // 3. 创建偏移量
    std::vector<hsize_t> offset;
    for (int i = 0; i < num_dims; ++i) {
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

        // 创建文件
        H5::H5File file(file_path, H5F_ACC_TRUNC, H5::FileCreatPropList::DEFAULT, plist);
        
        // 创建主组 LatticeColorMatrix
        H5::Group mainGroup = file.createGroup("LatticeColorMatrix");

        // 设置集体写入属性
        H5::DSetMemXferPropList xfer_plist;
        xfer_plist.copy(H5::DSetMemXferPropList::DEFAULT);
        H5Pset_dxpl_mpio(xfer_plist.getId(), H5FD_MPIO_COLLECTIVE);
        
        // 可以预先创建文件空间和内存空间，它们对所有方向都是一样的
        H5::DataSpace filespace(global_dims.size(), global_dims.data());
        H5::DataSpace memspace(local_dims.size(), local_dims.data());
        memspace.selectAll();
        filespace.selectHyperslab(H5S_SELECT_SET, local_dims.data(), offset.data());
        
        // 为每个方向创建单独的数据集
        for (int dir = 0; dir < Nd; ++dir) {
            // 确保方向名称有效
            std::string dir_name;
            if (dir < 4) {
                dir_name = std::string(1, high_dim_tag[dir]);
            } else {
                dir_name = std::to_string(dir);
            }
            
            // 在主组下创建数据集
            H5::DataSet dataset = mainGroup.createDataSet(dir_name, complex_id, filespace);
            
            // 计算这个方向的数据在gauge_out中的偏移量
            size_t dir_offset = dir * local_volume * Nc * Nc;
            
            // 写入数据 - 只写入当前方向的部分
            dataset.write(
                reinterpret_cast<Real_*>(gauge_out.data_ptr()) + dir_offset * 2, // *2因为复数占用2个Real_
                complex_id, 
                memspace, 
                filespace, 
                xfer_plist
            );
            
            // 添加额外的属性(metadata)，标记方向信息
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::Attribute attr = dataset.createAttribute("direction", H5::PredType::NATIVE_INT, attr_space);
            attr.write(H5::PredType::NATIVE_INT, &dir);
        }
        
        // 添加附加信息到主组
        {
            // 添加维度信息
            std::vector<int> dims_info(num_dims);
            for (int i = 0; i < num_dims; ++i) {
                dims_info[i] = lattice_total_dims[i];
            }
            
            hsize_t attr_dims[1] = {static_cast<hsize_t>(num_dims)};
            H5::DataSpace attr_space(1, attr_dims);
            H5::Attribute attr = mainGroup.createAttribute("lattice_dims", H5::PredType::NATIVE_INT, attr_space);
            attr.write(H5::PredType::NATIVE_INT, dims_info.data());
            
            // 添加颜色数信息
            H5::DataSpace nc_attr_space(H5S_SCALAR);
            H5::Attribute nc_attr = mainGroup.createAttribute("n_colors", H5::PredType::NATIVE_INT, nc_attr_space);
            int nc_val = static_cast<int>(Nc);
            nc_attr.write(H5::PredType::NATIVE_INT, &nc_val);
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