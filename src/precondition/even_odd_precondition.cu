#include "base/datatype/qcu_complex.cuh"
#include "check_error/check_cuda.cuh"
#include "kernel/precondition/eo_precondition.cuh"
#include "lattice_desc.h"
#include "precondition/even_odd_precondition.h"
#include <cassert>
namespace qcu {

template <typename _Float>
void EOPreconditioner<_Float>::apply(Complex<_Float>* __restrict__ output,
                                     Complex<_Float>* __restrict__ input, 
                                     const std::vector<int>& local_lattice_desc,
                                     int site_vec_len, 
                                     [[maybe_unused]] int Nd, 
                                     void* stream) 
{
    assert(local_lattice_desc.size() <= 4);
    int Lx = 1;
    int Ly = 1;
    int Lz = 1;
    int Lt = 1;
    for (int i = 0; i < local_lattice_desc.size(); ++i) {
        if (i == X_DIM) Lx = local_lattice_desc[i];
        if (i == Y_DIM) Ly = local_lattice_desc[i];
        if (i == Z_DIM) Lz = local_lattice_desc[i];
        if (i == T_DIM) Lt = local_lattice_desc[i];
    }

    int threads_per_block = 256;
    int blocks = (Lx * Ly * Lz * Lt + threads_per_block - 1) / threads_per_block;

    kernel::eo_precondition_4D<_Float>
        <<<blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(output, input, Lx, Ly,
                                                                                Lz, Lt, site_vec_len);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}
template <typename _Float>
void EOPreconditioner<_Float>::reverse(Complex<_Float>* __restrict__ output,
                                       Complex<_Float>* __restrict__ input, 
                                       const std::vector<int>& local_lattice_desc,
                                       int site_vec_len, [[maybe_unused]] int Nd, void* stream) 
{
    assert(local_lattice_desc.size() <= 4);
    int Lx = 1;
    int Ly = 1;
    int Lz = 1;
    int Lt = 1;
    for (int i = 0; i < local_lattice_desc.size(); ++i) {
        if (i == X_DIM) Lx = local_lattice_desc[i];
        if (i == Y_DIM) Ly = local_lattice_desc[i];
        if (i == Z_DIM) Lz = local_lattice_desc[i];
        if (i == T_DIM) Lt = local_lattice_desc[i];
    }

    int threads_per_block = 256;
    int blocks = (Lx * Ly * Lz * Lt + threads_per_block - 1) / threads_per_block;

    kernel::reverse_eo_precondition_4D<_Float>
        <<<blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(output, input, Lx, Ly,
                                                                                Lz, Lt, site_vec_len);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

template <typename _Float>
void GaugeEOPreconditioner<_Float>::apply(Complex<_Float>* __restrict__ output,
                                          Complex<_Float>* __restrict__ input,
                                          const std::vector<int>& local_lattice_desc, 
                                          int site_vec_len,
                                          [[maybe_unused]] int Nd, 
                                          void* stream) 
{
    assert(local_lattice_desc.size() <= 4);
    int Lx = 1;
    int Ly = 1;
    int Lz = 1;
    int Lt = 1;
    for (int i = 0; i < local_lattice_desc.size(); ++i) {
        if (i == X_DIM) Lx = local_lattice_desc[i];
        if (i == Y_DIM) Ly = local_lattice_desc[i];
        if (i == Z_DIM) Lz = local_lattice_desc[i];
        if (i == T_DIM) Lt = local_lattice_desc[i];
    }

  int threads_per_block = 256;
  int blocks = (Lx * Ly * Lz * Lt + threads_per_block - 1) / threads_per_block;
  for (int i = 0; i < 4; ++i) {
    Complex<_Float>* mu_output = output + i * Lx * Ly * Lz * Lt * site_vec_len;
    Complex<_Float>* mu_input = input + i * Lx * Ly * Lz * Lt * site_vec_len;
    kernel::eo_precondition_4D<_Float>
        <<<blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(
            mu_output, mu_input, Lx, Ly, Lz, Lt, site_vec_len);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

template <typename _Float>
void GaugeEOPreconditioner<_Float>::reverse(Complex<_Float>* __restrict__ output,
                                            Complex<_Float>* __restrict__ input,
                                            const std::vector<int>& local_lattice_desc, 
                                            int site_vec_len,
                                            [[maybe_unused]] int Nd, 
                                            void* stream) {
    assert(local_lattice_desc.size() <= 4);
    int Lx = 1;
    int Ly = 1;
    int Lz = 1;
    int Lt = 1;
    for (int i = 0; i < local_lattice_desc.size(); ++i) {
        if (i == X_DIM) Lx = local_lattice_desc[i];
        if (i == Y_DIM) Ly = local_lattice_desc[i];
        if (i == Z_DIM) Lz = local_lattice_desc[i];
        if (i == T_DIM) Lt = local_lattice_desc[i];
    }

  int threads_per_block = 256;
  int blocks = (Lx * Ly * Lz * Lt + threads_per_block - 1) / threads_per_block;

  for (int i = 0; i < 4; ++i) {
    Complex<_Float>* mu_output = output + i * Lx * Ly * Lz * Lt * site_vec_len;
    Complex<_Float>* mu_input = input + i * Lx * Ly * Lz * Lt * site_vec_len;
    kernel::reverse_eo_precondition_4D<_Float>
        <<<blocks, threads_per_block, 0, static_cast<cudaStream_t>(stream)>>>(
            mu_output, mu_input, Lx, Ly, Lz, Lt, site_vec_len);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

template class EOPreconditioner<float>;
template class EOPreconditioner<double>;
template class EOPreconditioner<half>;

template class GaugeEOPreconditioner<float>;
template class GaugeEOPreconditioner<double>;
template class GaugeEOPreconditioner<half>;
}  // namespace qcu