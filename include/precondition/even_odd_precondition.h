#pragma once
#include "base/datatype/qcu_complex.cuh"
#include "lattice_desc.h"
namespace qcu {

template <typename _Float>
class EOPreconditioner {
 public:
  virtual void apply(Complex<_Float>* output, Complex<_Float>* input, const Latt_Desc& desc,
                     int site_vec_len, int Nd = 4, void* stream = nullptr);
  virtual void reverse(Complex<_Float>* output, Complex<_Float>* input, const Latt_Desc& desc,
                       int site_vec_len, int Nd = 4, void* stream = nullptr);
};

template <typename _Float>
class GaugeEOPreconditioner : public EOPreconditioner<_Float> {
 public:
  void apply(Complex<_Float>* output, Complex<_Float>* input, const Latt_Desc& desc, int site_vec_len,
             int Nd = 4, void* stream = nullptr) override;
  void reverse(Complex<_Float>* output, Complex<_Float>* input, const Latt_Desc& desc,
               int site_vec_len, int Nd = 4, void* stream = nullptr) override;
};
}