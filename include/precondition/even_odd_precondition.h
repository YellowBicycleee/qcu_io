#pragma once
#include "base/datatype/qcu_complex.cuh"
#include "lattice_desc.h"
#include <vector>
namespace qcu {

// clang-format off
template <typename _Float>
class EOPreconditioner {
 public:
  virtual void apply (  Complex<_Float>* __restrict__ output, 
                        Complex<_Float>* __restrict__ input, 
                        const std::vector<int>& local_lattice_desc,
                        int site_vec_len, 
                        [[maybe_unused]] int Nd = 4, 
                        void* stream = nullptr);
  virtual void reverse( Complex<_Float>* __restrict__ output, 
                        Complex<_Float>* __restrict__ input, 
                        const std::vector<int>& local_lattice_desc,
                        int site_vec_len,
                        [[maybe_unused]]int Nd = 4,
                        void* stream = nullptr);
};

template <typename _Float>
class GaugeEOPreconditioner : public EOPreconditioner<_Float> {
 public:
  void apply(   Complex<_Float>* __restrict__ output, 
                Complex<_Float>* __restrict__ input, 
                const std::vector<int>& local_lattice_desc, 
                int site_vec_len,
                [[maybe_unused]] int Nd = 4, 
                void* stream = nullptr) override;
  void reverse( Complex<_Float>* __restrict__ output, 
                Complex<_Float>* __restrict__ input, 
                const std::vector<int>& local_lattice_desc,
                int site_vec_len, 
                [[maybe_unused]] int Nd = 4, 
                void* stream = nullptr) override;
};
}