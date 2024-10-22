#include "buffer_shape/data_shift.h"
#include "check_error/check_cuda.cuh"
namespace qcu {

static constexpr int M_TILE = 32;
static constexpr int N_TILE = 32;

static inline __device__ __host__ int div_ceil(int a, int b) { return (a + b - 1) / b; }
// constexpr int

template <typename _Tp>
// transpose input(m * n) to output(n * m)
__global__ void transpose2D_kernel (_Tp* output, const _Tp* input, int m, int n) {

  int block_x_stride = gridDim.x;
  int block_y_stride = gridDim.y;

  int total_logic_blocks_x = (n + blockDim.x - 1) / blockDim.x;
  int total_logic_blocks_y = (m + blockDim.y - 1) / blockDim.y;

  __shared__ _Tp tile[M_TILE][N_TILE + 1];  // erase bank conflict

  for (int i = blockIdx.x; i < total_logic_blocks_x; i += block_x_stride) {
    for (int j = blockIdx.y; j < total_logic_blocks_y; j += block_y_stride) {
      int logic_in_x = i * blockDim.x + threadIdx.x;
      int logic_in_y = j * blockDim.y + threadIdx.y;

      if (logic_in_y < m && logic_in_x < n) {
        tile[threadIdx.y][threadIdx.x] = input[logic_in_y * n + logic_in_x];
      }
      __syncthreads();

      int logic_out_x = j * blockDim.y + threadIdx.x;
      int logic_out_y = i * blockDim.x + threadIdx.y;
  
      if (logic_out_y < n && logic_out_x < m) {
        output[logic_out_y * m + logic_out_x] = tile[threadIdx.x][threadIdx.y];
      }
      __syncthreads();
    }
  }
}

template <typename _Float>
void gauge_from_sunwEO_to_qudaEO(Complex<_Float>* qudaEO_guage, Complex<_Float>* sunwEO_gauge,
                                 int nc_square, int quard_vol, void* stream) {
  int max_block_x = 1 << 16;
  int max_block_y = (1 << 16) - 1;

  int threads_per_block_x = N_TILE;
  int threads_per_block_y = M_TILE;
  int blocks_per_grid_y = std::min(div_ceil(nc_square, threads_per_block_y), max_block_y);
  int blocks_per_grid_x =
      std::min(div_ceil(quard_vol, threads_per_block_x), max_block_x);

  dim3 block_size(threads_per_block_x, threads_per_block_y);
  dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);

  transpose2D_kernel<Complex<_Float>>
      <<<grid_size, block_size, 0, static_cast<cudaStream_t>(stream)>>>(qudaEO_guage, sunwEO_gauge,
                                                                        nc_square, quard_vol);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

template <typename _Float>
void from_qudaEO_to_sunwEO(Complex<_Float>* sunwEO_gauge, Complex<_Float>* qudaEO_gauge,
                           int quard_vol, int nc_square, void* stream) {
  int max_block_x = 1 << 16;      
  int max_block_y = (1 << 16) - 1;

  int threads_per_block_x = N_TILE;
  int threads_per_block_y = M_TILE;
  int blocks_per_grid_y = std::min(div_ceil(quard_vol, threads_per_block_y), max_block_y);
  int blocks_per_grid_x =
      std::min(div_ceil(nc_square, threads_per_block_x), max_block_x);

  dim3 block_size(threads_per_block_x, threads_per_block_y);
  dim3 grid_size(blocks_per_grid_x, blocks_per_grid_y);

  transpose2D_kernel<Complex<_Float>>
      <<<grid_size, block_size, 0, static_cast<cudaStream_t>(stream)>>>(sunwEO_gauge, qudaEO_gauge,
                                                                        quard_vol, nc_square);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

template void gauge_from_sunwEO_to_qudaEO<float>(Complex<float>*, Complex<float>*, int, int, void*);
template void gauge_from_sunwEO_to_qudaEO<double>(Complex<double>*, Complex<double>*, int, int,
                                                  void*);
template void gauge_from_sunwEO_to_qudaEO<half>(Complex<half>*, Complex<half>*, int, int, void*);

template void from_qudaEO_to_sunwEO<float>(Complex<float>*, Complex<float>*, int, int, void*);
template void from_qudaEO_to_sunwEO<double>(Complex<double>*, Complex<double>*, int, int, void*);
template void from_qudaEO_to_sunwEO<half>(Complex<half>*, Complex<half>*, int, int, void*);
}  // namespace qcu