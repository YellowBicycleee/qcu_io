#include "base/alloc.h"
#include <cstring>
#include <cassert>
#include "check_error/check_cuda.cuh"
namespace base {
void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
  assert(src_ptr != nullptr);
  assert(dest_ptr != nullptr);
  if (!byte_size) {
    return;
  }

  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = static_cast<CUstream_st*>(stream);
  }

  if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
    std::memcpy(dest_ptr, src_ptr, byte_size);
  } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
    if (!stream_) {
      CHECK_CUDA(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice));
    } else {
      CHECK_CUDA(cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_));
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
    if (!stream_) {
      CHECK_CUDA(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost));
    } else {
      CHECK_CUDA(cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_));
    }
  } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
    if (!stream_) {
      CHECK_CUDA(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice));
    } else {
      CHECK_CUDA(cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_));
    }
  } else {
    fprintf(stderr, "Unknown memcpy kind: %d\n",  int(memcpy_kind));
    exit(-1);
  }
  if (need_sync) {
     cudaDeviceSynchronize();
  }
}


}  // namespace base