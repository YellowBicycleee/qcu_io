#pragma once

#include "base/datatype/qcu_complex.cuh"
#include "qcu_public.h"
namespace kernel {

template <typename _Float>
__forceinline__ __device__ void eo_precondition_4D_operator(Complex<_Float>* __restrict__ output,
                                                            Complex<_Float>* __restrict__ input,
                                                            int Lx, int Ly, int Lz, int Lt,
                                                            int site_vec_len, int cell_id) {
  // coord without precondition
  int x = cell_id % Lx;  // 不进行预处理的坐标x
  int x_prec = x / 2;    // 进行预处理的坐标x
  int y = (cell_id / Lx) % Ly;
  int z = (cell_id / Lx / Ly) % Lz;
  int t = cell_id / Lx / Ly / Lz;

  int parity = (x + y + z + t) % 2;

  Complex<_Float>* dst =
      output +
      (parity * Lx / 2 * Ly * Lz * Lt + (IDX4D(t, z, y, x_prec, Lz, Ly, Lx))) * site_vec_len;
  Complex<_Float>* src = input + cell_id * site_vec_len;

  for (int i = 0; i < site_vec_len; i++) {
    dst[i] = src[i];
  }
}

template <typename _Float>
__global__ void eo_precondition_4D(Complex<_Float>* __restrict__ output,
                                   Complex<_Float>* __restrict__ input, int Lx, int Ly, int Lz,
                                   int Lt, int site_vec_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int cell_id = tid; cell_id < Lx * Ly * Lz * Lt; cell_id += stride) {
    eo_precondition_4D_operator<_Float>(output, input, Lx, Ly, Lz, Lt, site_vec_len, cell_id);
  }
}

// reverse the even-odd preconditioning
template <typename _Float>
__forceinline__ __device__ void reverse_eo_precondition_4D_operator(
    Complex<_Float>* __restrict__ output, Complex<_Float>* __restrict__ input, int Lx, int Ly,
    int Lz, int Lt, int site_vec_len, int cell_id) {
  // coord without precondition
  int x = cell_id % Lx;  // 不进行预处理的坐标x
  int x_prec = x / 2;    // 进行预处理的坐标x
  int y = (cell_id / Lx) % Ly;
  int z = (cell_id / Lx / Ly) % Lz;
  int t = cell_id / Lx / Ly / Lz;

  int parity = (x + y + z + t) % 2;

  Complex<_Float>* src =
      input +
      (parity * Lx / 2 * Ly * Lz * Lt + (IDX4D(t, z, y, x_prec, Lz, Ly, Lx))) * site_vec_len;
  Complex<_Float>* dst = input + cell_id * site_vec_len;

  for (int i = 0; i < site_vec_len; i++) {
    dst[i] = src[i];
  }
}

template <typename _Float>
__global__ void reverse_eo_precondition_4D(Complex<_Float>* __restrict__ output,
                                           Complex<_Float>* __restrict__ input, int Lx, int Ly,
                                           int Lz, int Lt, int site_vec_len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int cell_id = tid; cell_id < Lx * Ly * Lz * Lt; cell_id += stride) {
    reverse_eo_precondition_4D_operator<_Float>(output, input, Lx, Ly, Lz, Lt, site_vec_len,
                                                cell_id);
  }
}

}  // namespace kernel