#pragma once
// #include <nccl.h>
#include <cstdio>

#define CHECK_NCCL(cmd)                                                   \
    do {                                                                  \
        ncclResult_t err = cmd;                                           \
        if (err != ncclSuccess) {                                         \
            fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(err)); \
            exit(1);                                                      \
        }                                                                 \
    } while (0)
