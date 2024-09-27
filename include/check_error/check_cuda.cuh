#pragma once
#include <cstdio>
#define DEBUG
#ifdef DEBUG

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
#else
#define CHECK_CUDA(ans) ans
#endif

