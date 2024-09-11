#pragma once
#include <cuda_runtime_api.h>
#define CHECK_CUDA(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while (0) 

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}