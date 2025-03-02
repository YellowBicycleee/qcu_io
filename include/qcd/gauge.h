#pragma once 

#include <memory>
#include <vector>
#include <numeric>
#include "lattice_desc.h"
#include "qcu_public.h"
#include <cstdio>
namespace qcu::io {

// template <typename T>
// class Gauge4Dim {
// private:
//     constexpr static int Ndim = 4;  
//     std::vector<T> data;
//     size_t Lt, Lz, Ly, Lx, Nc;
    
//     // 私有的索引计算方法
//     size_t index(size_t dim, size_t t, size_t z, size_t y, 
//                  size_t x, size_t c1, size_t c2) const {
//         return ((((((dim * Lt + t) * Lz + z) * Ly + y) * Lx + x) * Nc + c1) * Nc + c2);
//     }
    
// public:
//     Gauge4Dim(size_t t_size, size_t z_size, size_t y_size, 
//             size_t x_size, size_t color_size) 
//         : Lt(t_size), Lz(z_size), Ly(y_size), Lx(x_size), Nc(color_size) {
//         data.resize(Ndim * Lt * Lz * Ly * Lx * Nc * Nc);
//     }
//     Gauge4Dim(const Gauge4Dim& other) = delete;
//     Gauge4Dim(Gauge4Dim&& other) = default;
//     Gauge4Dim& operator=(const Gauge4Dim& other) = delete;
//     Gauge4Dim& operator=(Gauge4Dim&& other) = default;

//     T& operator()(size_t dim, size_t t, size_t z, size_t y, 
//                   size_t x, size_t c1, size_t c2) {
//         return data[index(dim, t, z, y, x, c1, c2)];
//     }

//     const T& operator()(size_t dim, size_t t, size_t z, size_t y, 
//                        size_t x, size_t c1, size_t c2) const {
//         return data[index(dim, t, z, y, x, c1, c2)];
//     }

//     T* data_ptr() { return data.data(); }
//     const T* data_ptr() const { return data.data(); }
    
//     size_t get_Lt() const { return Lt; }
//     size_t get_Lz() const { return Lz; }
//     size_t get_Ly() const { return Ly; }
//     size_t get_Lx() const { return Lx; }
//     size_t get_Nc() const { return Nc; }
//     static constexpr size_t get_Ndim() { return Ndim; }
// };

template <typename T>
class GaugeStorage {

public:
    GaugeStorage(const std::vector<int>& global_lattice_desc, int n_color) 
        : global_lattice_desc_(global_lattice_desc), n_color_(n_color) 
    {
        int dims = global_lattice_desc_.size();
        auto volume = std::accumulate(global_lattice_desc_.begin(), global_lattice_desc_.end(), 1, std::multiplies<int>());
        data_.resize(dims * volume * n_color_ * n_color_);
        // printf("volume * n_color_ * n_color_ =  %ld\n", volume * n_color_ * n_color_);
        // printf("n_color_ = %d, volume = %d\n", n_color_, volume);
        // for (int i = 0; i < global_lattice_desc_.size(); ++i) {
        //     printf("global_lattice_desc_[%d] = %d\n", i, global_lattice_desc_[i]);
        // }
    }
    GaugeStorage(const GaugeStorage& other) = delete;
    GaugeStorage(GaugeStorage&& other) = default;
    GaugeStorage& operator=(const GaugeStorage& other) = delete;
    GaugeStorage& operator=(GaugeStorage&& other) = default;

    T* data_ptr() { return data_.data(); }
    const T* data_ptr() const { return data_.data(); }  
    int get_dim_num() const { return global_lattice_desc_.size(); }
    int get_n_color() const { return n_color_; }
    std::vector<int> get_global_lattice_desc() const { return global_lattice_desc_; }
private: 
    std::vector<T> data_;
    const std::vector<int> global_lattice_desc_;
    const int n_color_;    
};


}  // namespace qcu