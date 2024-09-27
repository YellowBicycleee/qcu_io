#include "base/datatype/qcu_complex.cuh"
#include "check_error/check_cuda.cuh"
#include "kernel/precondition/eo_precondition.cuh"
#include "lattice_desc.h"
#include "precondition/even_odd_precondition.h"

namespace qcu {

template <typename _Float>
void EOPreconditioner<_Float>::apply(Complex<float>* output, Complex<float>* input,
                                     const Latt_Desc& desc, int site_vec_len, int Nd,
                                     void* stream) {
  int Lx = desc.data[X_DIM];
  int Ly = desc.data[Y_DIM];
  int Lz = desc.data[Z_DIM];
  int Lt = desc.data[T_DIM];

  int threads_per_block = 256;
  int blocks = (Lx * Ly * Lz * Lt + threads_per_block - 1) / threads_per_block;

  kernel::eo_precondition_4D<_Float><<<blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(
      output, input, Lx, Ly, Lz, Lt, site_vec_len);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}
template <typename _Float>
void EOPreconditioner<_Float>::reverse(Complex<float>* output, Complex<float>* input,
                                       const Latt_Desc& desc, int site_vec_len, int Nd,
                                       void* stream) {
  int Lx = desc.data[X_DIM];
  int Ly = desc.data[Y_DIM];
  int Lz = desc.data[Z_DIM];
  int Lt = desc.data[T_DIM];

  int threads_per_block = 256;
  int blocks = (Lx * Ly * Lz * Lt + threads_per_block - 1) / threads_per_block;

  kernel::reverse_eo_precondition_4D<_Float>
      <<<blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(output, input, Lx, Ly,
                                                                            Lz, Lt, site_vec_len);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
}
}  // namespace qcu